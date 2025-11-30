# model/Predictor.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType


class Projector(nn.Module):
    def __init__(self, src_dim: int, tgt_dim: int, tokens_per_elem: int = 2):
        super().__init__()
        self.tokens_per_elem = int(tokens_per_elem)
        self.tgt_dim = int(tgt_dim)
        self.proj = nn.Sequential(
            nn.Linear(src_dim, tgt_dim * self.tokens_per_elem),
            nn.GELU(),
            nn.Linear(tgt_dim * self.tokens_per_elem, tgt_dim * self.tokens_per_elem),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, _ = x.shape
        if K == 0:
            return x.new_zeros(B, 0, self.tgt_dim)
        y = self.proj(x).view(B, K * self.tokens_per_elem, self.tgt_dim)
        return y


class NextCatePrefix(nn.Module):
    def __init__(self, llm_hidden: int, tgt_dim: int, tokens_per_elem: int = 2):
        super().__init__()
        self.tokens_per_elem = int(tokens_per_elem)
        self.tgt_dim = int(tgt_dim)
        self.mlp = nn.Sequential(
            nn.Linear(llm_hidden, tgt_dim * self.tokens_per_elem),
            nn.GELU(),
            nn.Linear(tgt_dim * self.tokens_per_elem, tgt_dim * self.tokens_per_elem),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        y = self.mlp(x).view(B, self.tokens_per_elem, self.tgt_dim)
        return y

class _FFNBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        return x + self.ffn(self.norm(x))

class Predictor(nn.Module):
    """
    共享 LLM 的两阶段 Prefix 流程（保持原有接口/变量不变）：
      Phase A：P_cate + P_st + P_nextst + 指令 -> 预测 next_cate（辅助头）并产出 P_nextcate
      Phase B：仅 P_nextcate + P_hlong + P_hshort + 指令 -> 预测 app_id / traffic
    """
    def __init__(
        self,
        model_path: str,
        num_app: int,
        num_cate: int,
        num_traffic_bin: int,
        tokens_per_elem: int = 2,
        max_elem_text_len: int = 96,
        inst_max_len: int = 64,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        # ===== QLoRA 4-bit 量化加载（不改动外部 .to(device) 习惯：不指定 device_map）=====
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,   # 或 torch.float16
            bnb_4bit_use_double_quant=True,
        )
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_cfg,
        )
        self.llm.config.use_cache = False
        # 省显存：激活重算
        self.llm.gradient_checkpointing_enable()

        # ===== LoRA 微调（保持参数名/行为不变）=====
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_target_modules,
        )
        self.llm = get_peft_model(self.llm, peft_cfg)
        if freeze_backbone:
            for n, p in self.llm.named_parameters():
                p.requires_grad_(("lora_" in n))

        self.max_elem_text_len = int(max_elem_text_len)
        self.inst_max_len = int(inst_max_len)

        self.hidden = self.llm.config.hidden_size
        self.emb_dim = self.llm.get_input_embeddings().embedding_dim

        # embedding → 软前缀
        self.proj_cate   = Projector(self.hidden, self.emb_dim, tokens_per_elem)
        self.proj_st     = Projector(self.hidden, self.emb_dim, tokens_per_elem)
        self.proj_hlong  = Projector(self.hidden, self.emb_dim, tokens_per_elem)
        self.proj_hshort = Projector(self.hidden, self.emb_dim, tokens_per_elem)
        self.proj_nextst = Projector(self.hidden, self.emb_dim, tokens_per_elem)

        # Phase-A 输出 → next_cate 前缀
        self.proj_nextcate = NextCatePrefix(self.hidden, self.emb_dim, tokens_per_elem)

        # 预测头（保持命名和用途不变）
        self.poolA = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.hidden))
        self.head_next_cate = nn.Linear(self.hidden, num_cate)

        self.poolB = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.hidden))
        self.head_app = nn.Linear(self.hidden, num_app)
        # self.head_traffic  = nn.Linear(self.hidden, 1)
        self.head_traffic = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),   # 先做一个投影/稳定维度
            _FFNBlock(self.hidden, self.hidden * 4),
            _FFNBlock(self.hidden, self.hidden * 4),
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, 1),
        )


    # ---------- 逐元素编码：不参与反向图，显存大降 ----------
    def _encode_flat(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, self.hidden, device=self.head_app.weight.device)
        enc = self.tok(
            texts, max_length=self.max_elem_text_len, truncation=True, padding=True, return_tensors="pt"
        ).to(self.head_app.weight.device)
        # with torch.no_grad():  # 关键：不反传这一路
        out = self.llm.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
        hs = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb.detach()  # [N,H]

    def _embed_lists(self, lists: List[List[str]]) -> torch.Tensor:
        B = len(lists)
        flat, counts = [], []
        for li in lists:
            counts.append(len(li))
            flat.extend(li)
        if len(flat) == 0:
            return torch.zeros(B, 0, self.hidden, device=self.head_app.weight.device)
        emb = self._encode_flat(flat)  # [sumK,H]
        out_list, off = [], 0
        Kmax = max(counts + [1])
        for k in counts:
            if k == 0:
                out_list.append(torch.zeros(Kmax, self.hidden, device=emb.device))
            else:
                cur = emb[off:off+k]
                pad = torch.zeros(Kmax-k, self.hidden, device=emb.device)
                out_list.append(torch.cat([cur, pad], dim=0))
                off += k
        return torch.stack(out_list, dim=0)  # [B,Kmax,H]

    def _inst_embeds(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(texts, max_length=self.inst_max_len, truncation=True, padding=True, return_tensors="pt"
        ).to(self.head_app.weight.device)
        tok_emb = self.llm.get_input_embeddings()(enc["input_ids"])   # [B,T,Emb]
        attn = enc["attention_mask"]                                   # [B,T]
        return tok_emb, attn

    def forward(
        self,
        texts_cate_long: List[List[str]],
        texts_st_long: List[List[str]],
        texts_hlong: List[List[str]],
        texts_hshort: List[List[str]],
        texts_next_st: List[str],
        instA_texts: List[str],
        instB_texts: List[str],
    ) -> Dict[str, torch.Tensor]:

        device = self.head_app.weight.device
        # 1) 元素级编码
        E_cate_long = self._embed_lists(texts_cate_long)  # [B,Kc,H]
        E_st_long   = self._embed_lists(texts_st_long)    # [B,Ks,H]
        E_hlong     = self._embed_lists(texts_hlong)      # [B,KhL,H]
        E_hshort    = self._embed_lists(texts_hshort)     # [B,KhS,H]
        E_next_st   = self._encode_flat(texts_next_st)    # [B,H]

        # 2) embedding → 软前缀
        P_cate   = self.proj_cate(E_cate_long)               # [B,Kc*P,Emb]
        P_st     = self.proj_st(E_st_long)                   # [B,Ks*P,Emb]
        P_hlong  = self.proj_hlong(E_hlong)                  # [B,KhL*P,Emb]
        P_hshort = self.proj_hshort(E_hshort)                # [B,KhS*P,Emb]
        P_nextst = self.proj_nextst(E_next_st.unsqueeze(1))  # [B,P,Emb]

        # ===== Phase A =====
        instA_tok, instA_attn = self._inst_embeds(instA_texts)
        B = instA_tok.size(0)
        prefA = torch.cat([P_cate, P_st, P_nextst], dim=1)
        attnA_pref = torch.ones(B, prefA.size(1), dtype=torch.long, device=device)
        inputsA = torch.cat([prefA, instA_tok], dim=1)
        attnA   = torch.cat([attnA_pref, instA_attn], dim=1)

        outA = self.llm(inputs_embeds=inputsA, attention_mask=attnA,
                        output_hidden_states=True, return_dict=True)
        hA = outA.hidden_states[-1][:, -1, :]        # [B,H]
        logits_next_cate = self.head_next_cate(self.poolA(hA))  # [B,V_cate]
        P_nextcate = self.proj_nextcate(hA)          # [B,P,Emb]

        # ===== Phase B =====
        instB_tok, instB_attn = self._inst_embeds(instB_texts)
        prefB = torch.cat([P_nextcate, P_hlong, P_hshort], dim=1)
        attnB_pref = torch.ones(B, prefB.size(1), dtype=torch.long, device=device)
        inputsB = torch.cat([prefB, instB_tok], dim=1)
        attnB   = torch.cat([attnB_pref, instB_attn], dim=1)

        outB = self.llm(inputs_embeds=inputsB, attention_mask=attnB,
                        output_hidden_states=True, return_dict=True)
        hB = outB.hidden_states[-1][:, -1, :]        # [B,H]
        pooledB = self.poolB(hB)
        logits_app = self.head_app(pooledB)          # [B,V_app]
        traffic_raw = self.head_traffic(pooledB)     # [B,1]
        pred_traffic_kb = F.softplus(traffic_raw).squeeze(-1)  # [B]  (>=0)

        return {
            "logits_app": logits_app,
            "pred_traffic_kb": pred_traffic_kb,
            "logits_next_cate": logits_next_cate,
        }
