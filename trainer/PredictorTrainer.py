# trainer/PredictorTrainer.py
from __future__ import annotations
from typing import Dict, Any, List
import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset.MyDataset import MyDataset
from model.Selector import Selector

from util import (
    build_prompt_cate_traffic_elem,
    build_prompt_loc_time_elem,
    build_prompt_app_traffic_elem,
    build_cosine_warmup_decay
)

REQUIRED = ["app_ids","time","loc_ids","app_cates","traffic_bins","traffic"]

# ---------------- 资源加载 ----------------
def _load_cate_map(cate_csv: str) -> Dict[int, str]:
    df = pd.read_csv(cate_csv)
    need = {"app_type_id","en","cn"}
    if not set(need).issubset(df.columns):
        raise KeyError(f"Category CSV must include columns: {need}")
    return {int(r.app_type_id): str(r.en) for _, r in df.iterrows()}

def _load_location_poi(loc_csv: str) -> Dict[int, Dict[str, int]]:
    df = pd.read_csv(loc_csv)
    if "base_id" not in df.columns:
        raise KeyError("Location CSV must contain 'base_id'.")
    poi_cols = [c for c in df.columns if c.startswith("poi_")]
    table: Dict[int, Dict[str,int]] = {}
    for _, r in df.iterrows():
        table[int(r["base_id"])] = {c.replace("poi_",""): int(r[c]) for c in poi_cols}
    return table

def _collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: torch.stack([b[k] for b in batch_list], dim=0) for k in REQUIRED}

@torch.no_grad()
def _to_device(batch: Dict[str, Any], device: str):
    return {k:(v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}

def _indices_from_mask(mask: torch.Tensor, topk: int, recent_first: bool=True) -> List[List[int]]:
    """
    mask: [B, L] 0/1 或分数。返回每个样本的索引列表（递增）。
    """
    B, L = mask.shape
    out = []
    for i in range(B):
        row = mask[i]
        if ((row.max() - row.min()) > 1e-6):
            k = min(topk, L)
            idx = torch.topk(row, k=k, largest=True, sorted=True).indices.tolist()
            idx = sorted(idx)
        else:
            idx = (row > 0.5).nonzero(as_tuple=True)[0].tolist()
            if len(idx) > topk:
                idx = idx[-topk:] if recent_first else idx[:topk]
            if len(idx) < topk:
                need = topk - len(idx)
                filler = list(range(max(0, L-need), L))
                idx = sorted(list(dict.fromkeys(idx + filler)))
        out.append(idx)
    return out

def _split_views(batch: Dict[str, torch.Tensor], long_L: int, short_L: int):
    """
    使用总长度 L = long_L + short_L + 1
      - hist_len = L-1
      - short:   [hist_len - short_L, hist_len)
      - long:    [hist_len - (long_L + short_L), hist_len)   （允许与 short 重叠）
      - label:   index = hist_len
    """
    B, L = batch["app_ids"].shape
    hist_len = L - 1
    assert L >= long_L + short_L + 1, f"L={L} < long+short+1"

    sl_short = slice(hist_len - short_L, hist_len)
    sl_long  = slice(hist_len - (long_L + short_L), hist_len)

    view = {
        "long_app":   batch["app_ids"][:, sl_long],
        "long_time":  batch["time"][:, sl_long],
        "long_loc":   batch["loc_ids"][:, sl_long],
        "long_cate":  batch["app_cates"][:, sl_long],
        "long_tb":    batch["traffic_bins"][:, sl_long],
        "long_traffic":  batch["traffic"][:, sl_long],     # +++

        "short_app":  batch["app_ids"][:, sl_short],
        "short_time": batch["time"][:, sl_short],
        "short_loc":  batch["loc_ids"][:, sl_short],
        "short_tb":   batch["traffic_bins"][:, sl_short],
        "short_traffic": batch["traffic"][:, sl_short],    # +++

        "label_app":  batch["app_ids"][:, hist_len],
        "label_tb":   batch["traffic_bins"][:, hist_len],
        "label_cate": batch["app_cates"][:, hist_len],
        "label_loc":  batch["loc_ids"][:, hist_len],   # 用于 next_st prompt
        "label_time": batch["time"][:, hist_len],
        "label_traffic": batch["traffic"][:, hist_len],  # 连续值监督
    }
    return view

# ---------------- 指标（对 app_id） ----------------
def _acc_at_k(logits: torch.Tensor, labels: torch.Tensor, ks: List[int]) -> Dict[int, float]:
    # logits:[N,V], labels:[N]
    res = {}
    topk_vals, topk_idx = torch.topk(logits, k=max(ks), dim=-1)
    for k in ks:
        hit = (topk_idx[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        res[k] = hit
    return res

def _mrr_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
    # rank = 1/(pos) if found in topk else 0
    B = labels.size(0)
    ranks = torch.zeros(B, device=logits.device)
    for i in range(B):
        pos = (topk_idx[i] == labels[i]).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            ranks[i] = 1.0 / float(pos.item() + 1)
        else:
            ranks[i] = 0.0
    return ranks.mean().item()

def _ndcg_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    # 单一相关项的 NDCG：若在位置 p (1-based)，DCG = 1/log2(p+1), IDCG=1/log2(1+1)=1
    topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
    B = labels.size(0)
    scores = torch.zeros(B, device=logits.device)
    log2 = torch.log2
    for i in range(B):
        pos = (topk_idx[i] == labels[i]).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            p = pos.item() + 1
            scores[i] = 1.0 / log2(torch.tensor(p+1.0, device=logits.device))
        else:
            scores[i] = 0.0
    return scores.mean().item()

# ---------------- 主训练 ----------------
def trainPredictor(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    # ===== 数据 & 切分（保持不变）=====
    long_L  = int(getattr(args, "long_seq_len", 128))
    short_L = int(getattr(args, "short_seq_len", 32))
    total_L = long_L + short_L + 1

    dataset = MyDataset(
        csv_file=getattr(args, "predictor_csv_file", None) or args.csv_file,
        seq_len=total_L,
        stride=getattr(args, "stride", None),
        pad_id=getattr(args, "pad_id", 0),
        keep_in_memory=getattr(args, "keep_in_memory", True),
    )

    val_ratio = getattr(args, "val_ratio", 0.05)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.predictor_batch_size, shuffle=True,
        num_workers=getattr(args, "num_workers", 4), pin_memory=True, drop_last=True,
        collate_fn=_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=args.predictor_batch_size, shuffle=False,
        num_workers=getattr(args, "num_workers", 4), pin_memory=True, drop_last=False,
        collate_fn=_collate
    )

    # ===== 资源（保持不变）=====
    cate_id2en = _load_cate_map(args.app_category_csv)
    base_id2poi = _load_location_poi(args.location_csv)
    traffic_labels = getattr(args, "traffic_bin_labels", [
        "ultra-low","very-low","low","lower-mid","mid","upper-mid","high","very-high","extreme","burst"
    ])

    # ===== 模型（共享 LLM + LoRA + QLoRA）=====
    if not args.ablation:
        from model.Predictor import Predictor
        predictor = Predictor(
            model_path=getattr(args, "model_path", getattr(args, "llm_name", "Qwen2-1.5B-Instruct")),
            num_app=args.app_vocab_size,
            num_cate=args.num_app_cate,
            num_traffic_bin=args.num_traffic_bin,
            tokens_per_elem=getattr(args, "tokens_per_elem", 2),
            max_elem_text_len=getattr(args, "max_elem_text_len", 96),
            inst_max_len=getattr(args, "inst_max_len", 64),
            lora_r=getattr(args, "lora_r", 64),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.1),
            lora_target_modules=getattr(args, "lora_target_modules", None),
            freeze_backbone=getattr(args, "freeze_backbone", True),
        ).to(device)
    else:
        from model.Predictor_ablation import Predictor
        predictor = Predictor(
            model_path=getattr(args, "model_path", getattr(args, "llm_name", "Qwen2-1.5B-Instruct")),
            num_app=args.app_vocab_size,
            num_cate=args.num_app_cate,
            num_traffic_bin=args.num_traffic_bin,
            use_nextcate_in_phaseB=args.use_nextcate_in_phaseB, use_hlong_in_phaseB=args.use_hlong_in_phaseB,
            tokens_per_elem=getattr(args, "tokens_per_elem", 2),
            max_elem_text_len=getattr(args, "max_elem_text_len", 96),
            inst_max_len=getattr(args, "inst_max_len", 64),
            lora_r=getattr(args, "lora_r", 64),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.1),
            lora_target_modules=getattr(args, "lora_target_modules", None),
            freeze_backbone=getattr(args, "freeze_backbone", True),
        ).to(device)

    continue_the_best = (getattr(args, "continue_the_best", None))
    if continue_the_best and os.path.exists(continue_the_best):
        ckpt = torch.load(continue_the_best, map_location="cpu")
        # 1) 加载模型权重
        sd = ckpt.get("predictor", ckpt)  # 兼容纯 state_dict / dict 包裹两种
        missing, unexpected = predictor.load_state_dict(sd, strict=False)
        print(f"[Resume] Loaded predictor from {continue_the_best} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ===== Selector（保持不变）=====
    selector = Selector(
        num_apps=args.app_vocab_size,
        num_locs=args.location_vocab_size,
        num_app_cate=args.num_app_cate,
        num_traffic_bin=args.num_traffic_bin,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        key_pattern_len=args.key_pattern_len,
    ).to(device)
    sel_ckpt = getattr(args, "selector_ckpt", None) or getattr(args, "selector_save_path", None)
    if sel_ckpt and os.path.exists(sel_ckpt):
        state = torch.load(sel_ckpt, map_location="cpu")
        sd = state.get("selector", state) if isinstance(state, dict) else state
        selector.load_state_dict(sd, strict=False)
        print(f"[Predictor] Loaded Selector from {sel_ckpt}")
    else:
        print("[Predictor] WARNING: selector_ckpt not found; using randomly initialized selector.")
    selector.eval()
    for p in selector.parameters():
        p.requires_grad = False

    # ===== 优化器（保持不变）=====
    """原版：所有可训练参数同等对待"""
    # trainable = [p for p in predictor.parameters() if p.requires_grad]
    # optim = torch.optim.AdamW(
    #     trainable, lr=getattr(args, "lr", 1e-4), weight_decay=getattr(args, "weight_decay", 0.01)
    # )

    """针对性"""
    # # 替换你现在的 optimizer 构造
    lora_params, head_params, other_params = [], [], []
    for n, p in predictor.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(p)  # LoRA 低 wd
        elif any(k in n for k in ["head_app", "head_tb", "head_traffic", "head_next_cate",
                                "proj_", "poolA", "poolB"]):
            head_params.append(p)  # 新头/投影较大学习率
        else:
            other_params.append(p) # 理论上很少（大多已冻结）

    optim = torch.optim.AdamW(
        [
            {"params": lora_params,  "lr": getattr(args, "lr", 1e-4),      "weight_decay": 0.0},
            {"params": head_params,  "lr": getattr(args, "lr", 1e-4) * 5,  "weight_decay": 0.01},
            {"params": other_params, "lr": getattr(args, "lr", 1e-4),      "weight_decay": 0.01},
        ],
        betas=(0.9, 0.95), eps=1e-8,
        fused=torch.cuda.is_available()  # torch>=2.0
    )
    """新增：学习率调度（线性或余弦）"""
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(getattr(args, "epochs_predictor", getattr(args, "predictor_epochs", 5)))
    warmup_steps = int(getattr(args, "warmup_ratio", 0.03) * total_steps)
    scheduler = build_cosine_warmup_decay(optim, total_steps=total_steps,
                                      warmup_steps=warmup_steps, min_ratio=0.1)

    #================================================================================================

    topk = int(getattr(args, "selector_topk", 5))
    epochs = int(getattr(args, "epochs_predictor", getattr(args, "predictor_epochs", 5)))
    cate_aux_lambda = float(getattr(args, "cate_aux_lambda", 1.0))
    traf_aux_lambda = float(getattr(args, "traf_aux_lambda", 1.0))  # +++
    eval_topk = list(getattr(args, "eval_topk", [1, 5, 10]))

    # ===== 新增：AMP & 梯度累积（不改名，不改业务逻辑）=====
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    grad_accum = int(getattr(args, "grad_accum", 1))  # 可不在 config 里显式加

    best_val = float("inf")

    for ep in range(1, epochs+1):
        predictor.train()
        tot, nstep = 0.0, 0

        if args.train:
            for step, batch in enumerate(tqdm(train_loader, desc=f"[Predictor Train] Ep{ep:03d}")):
                batch = _to_device(batch, device)
                view = _split_views(batch, long_L, short_L)

                # selector on long（保持不变）
                with torch.no_grad():
                    m = selector(view["long_app"], view["long_time"], view["long_loc"], view["long_cate"], view["long_tb"])
                    if m.dim() == 3 and m.size(-1) == 1:
                        m = m.squeeze(-1)
                    idx_list = _indices_from_mask(m, topk=topk, recent_first=True)

                # 逐元素 prompt（保持不变）
                B = view["long_app"].size(0)
                texts_cate_long, texts_st_long = [], []
                texts_hlong, texts_hshort = [], []
                texts_next_st = []
                instA_texts, instB_texts = [], []

                for i in range(B):
                    li_cate, li_st, li_hlong = [], [], []
                    for j in idx_list[i]:
                        li_cate.append(build_prompt_cate_traffic_elem(
                            cate_id=int(view["long_cate"][i, j].item()),
                            traffic_bin=int(view["long_tb"][i, j].item()),
                            cate_id2en=cate_id2en,
                            traffic_labels=traffic_labels
                        ))
                        li_st.append(build_prompt_loc_time_elem(
                            loc_id=int(view["long_loc"][i, j].item()),
                            time_val=int(view["long_time"][i, j].item()),
                            base_id2poi=base_id2poi, topk=5
                        ))
                        li_hlong.append(build_prompt_app_traffic_elem(
                            app_id=int(view["long_app"][i, j].item()),
                            traffic=float(view["long_traffic"][i, j].item())   # +++
                        ))
                    texts_cate_long.append(li_cate)
                    texts_st_long.append(li_st)
                    texts_hlong.append(li_hlong)

                    sj = []
                    for j in range(view["short_app"].size(1)):
                        sj.append(build_prompt_app_traffic_elem(
                            app_id=int(view["short_app"][i, j].item()),
                            traffic=float(view["short_traffic"][i, j].item())  # +++
                        ))
                    texts_hshort.append(sj)

                    next_loc = int(view["label_loc"][i].item())
                    next_time = int(view["label_time"][i].item())
                    texts_next_st.append(build_prompt_loc_time_elem(
                        loc_id=next_loc, time_val=next_time, base_id2poi=base_id2poi, topk=5
                    ))

                    instA_texts.append("Predict the next app category.")
                    instB_texts.append("Predict the next app and traffic volume.")

                # ====== AMP 前向 + 累积 ======
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out = predictor(
                        texts_cate_long=texts_cate_long,
                        texts_st_long=texts_st_long,
                        texts_hlong=texts_hlong,
                        texts_hshort=texts_hshort,
                        texts_next_st=texts_next_st,
                        instA_texts=instA_texts,
                        instB_texts=instB_texts,
                    )
                    # === loss_app===
                    loss_app = F.cross_entropy(out["logits_app"], view["label_app"]-1, label_smoothing=0.05)  # << 我顺手加了label_smoothing，见下文


                    # === loss_trf（traffic）===
                    pred_kb = out["pred_traffic_kb"]                        # [B]  真实 KB（非负）
                    y_kb    = view["label_traffic"].float().clamp_min(0)
                    # 训练期对 y 做轻量裁剪（p99），抑制极端值对梯度的主导
                    clip_max = torch.quantile(y_kb, q=0.95)           # 也可固定为 5e5
                    y_clip   = y_kb.clamp_max(clip_max)
                    # 主损失：Huber on log1p（对长尾更稳）
                    pred_log = torch.log1p(pred_kb)
                    y_log    = torch.log1p(y_clip)
                    loss_traf_log = F.smooth_l1_loss(pred_log, y_log, beta=0.2)
                    # 线性空间 MAE（小权重）——让模型别完全忽略大值
                    loss_traf_lin = (pred_kb - y_clip).abs().mean()
                    loss_trf = loss_traf_log + 0.2 * loss_traf_lin
                    
                    
                    # === loss_cate===
                    loss_cate = F.cross_entropy(out["logits_next_cate"], view["label_cate"]-1, label_smoothing=0.05)

                    # === 打印用的 total（未 / grad_accum） ===
                    total_loss_raw = loss_app + traf_aux_lambda * loss_trf + cate_aux_lambda * loss_cate
                    # === 反向传播的 loss（考虑梯度累积） ===
                    loss = total_loss_raw / max(1, grad_accum)

                log_every = int(getattr(args, "log_every", 1))
                if step % log_every == 0:
                    print_line = (f"[Ep{ep:03d} it{step:05d}] "
                                f"total={float(total_loss_raw.detach().cpu()):.4f} "
                                f"app={float(loss_app.detach().cpu()):.4f} "
                                f"traffic={float(loss_traf_lin.detach().cpu()):.4f} "
                                f"next_cate={float(loss_cate.detach().cpu()):.4f}")
                    print(print_line, flush=True)  # 或 tqdm.write(print_line)
                    save_path = getattr(args, "predictor_save_path", None)
                    if save_path and total_loss_raw < best_val:
                        best_val = total_loss_raw
                        torch.save(
                            {"predictor": predictor.state_dict(), "args": vars(args), "epoch": ep, "val_loss": total_loss_raw},
                            save_path
                        )
                        print(f"  ✓ Saved predictor to {save_path}")

                scaler.scale(loss).backward()
                
                if (step + 1) % max(1, grad_accum) == 0:
                    scaler.step(optim)
                    scaler.update()
                    scheduler.step()     # <<<< 新增
                    optim.zero_grad(set_to_none=True)

                tot += float(total_loss_raw.detach().cpu())
                nstep += 1

            tr_loss = tot / max(1, nstep)

        # ==================== 验证（autocast 省显存）====================
        predictor.eval()
        tot_v, n_v = 0.0, 0
        hits_at_k = {k: 0.0 for k in eval_topk}
        mrr_at_k  = {k: 0.0 for k in eval_topk}
        ndcg_at_k = {k: 0.0 for k in eval_topk}
        N_eval = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Predictor Valid] Ep{ep:03d}"):
                batch = _to_device(batch, device)
                view = _split_views(batch, long_L, short_L)

                m = selector(view["long_app"], view["long_time"], view["long_loc"], view["long_cate"], view["long_tb"])
                if m.dim() == 3 and m.size(-1) == 1:
                    m = m.squeeze(-1)
                idx_list = _indices_from_mask(m, topk=topk, recent_first=True)

                B = view["long_app"].size(0)
                texts_cate_long, texts_st_long = [], []
                texts_hlong, texts_hshort = [], []
                texts_next_st = []
                instA_texts, instB_texts = [], []

                for i in range(B):
                    li_cate, li_st, li_hlong = [], [], []
                    for j in idx_list[i]:
                        li_cate.append(build_prompt_cate_traffic_elem(
                            cate_id=int(view["long_cate"][i, j].item()),
                            traffic_bin=int(view["long_tb"][i, j].item()),
                            cate_id2en=cate_id2en,
                            traffic_labels=traffic_labels
                        ))
                        li_st.append(build_prompt_loc_time_elem(
                            loc_id=int(view["long_loc"][i, j].item()),
                            time_val=int(view["long_time"][i, j].item()),
                            base_id2poi=base_id2poi, topk=5
                        ))
                        li_hlong.append(build_prompt_app_traffic_elem(
                            app_id=int(view["long_app"][i, j].item()),
                            traffic=float(view["long_traffic"][i, j].item())   # +++
                        ))
                    texts_cate_long.append(li_cate)
                    texts_st_long.append(li_st)
                    texts_hlong.append(li_hlong)

                    sj = []
                    for j in range(view["short_app"].size(1)):
                        sj.append(build_prompt_app_traffic_elem(
                            app_id=int(view["short_app"][i, j].item()),
                            traffic=float(view["short_traffic"][i, j].item())  # +++
                        ))
                    texts_hshort.append(sj)

                    next_loc = int(view["label_loc"][i].item())
                    next_time = int(view["label_time"][i].item())
                    texts_next_st.append(build_prompt_loc_time_elem(
                        loc_id=next_loc, time_val=next_time, base_id2poi=base_id2poi, topk=5
                    ))

                    instA_texts.append("Predict the next app category.")
                    instB_texts.append("Predict the next app and traffic level.")

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out = predictor(
                        texts_cate_long=texts_cate_long,
                        texts_st_long=texts_st_long,
                        texts_hlong=texts_hlong,
                        texts_hshort=texts_hshort,
                        texts_next_st=texts_next_st,
                        instA_texts=instA_texts,
                        instB_texts=instB_texts,
                    )

                    # === traffic 验证损失（与训练一致）===
                    pred_kb = out["pred_traffic_kb"]                        # [B]  真实 KB（非负）
                    y_kb    = view["label_traffic"].float().clamp_min(0)
                    # 训练期对 y 做轻量裁剪（p99），抑制极端值对梯度的主导
                    clip_max = torch.quantile(y_kb, q=0.95)           # 也可固定为 5e5
                    y_clip   = y_kb.clamp_max(clip_max)
                    # 主损失：Huber on log1p（对长尾更稳）
                    pred_log = torch.log1p(pred_kb)
                    y_log    = torch.log1p(y_clip)
                    loss_traf_log = F.smooth_l1_loss(pred_log, y_log, beta=0.2)
                    # 线性空间 MAE（小权重）——让模型别完全忽略大值
                    loss_traf_lin = (pred_kb - y_clip).abs().mean()
                    loss_trf = loss_traf_log + 0.2 * loss_traf_lin

                    # 分类损失（可带轻微 label smoothing，和训练保持一致）
                    loss_app  = F.cross_entropy(out["logits_app"],       view["label_app"]-1,  label_smoothing=0.05)
                    loss_cate = F.cross_entropy(out["logits_next_cate"], view["label_cate"]-1, label_smoothing=0.05)

                    v_loss = loss_app + traf_aux_lambda * loss_trf + cate_aux_lambda * loss_cate

            B = view["label_app"].size(0)
            tot_v += float(v_loss.detach().cpu()) * B   # sample-weighted
            n_v   += B

            N_eval += B
            accs = _acc_at_k(out["logits_app"], view["label_app"]-1, eval_topk)
            for k in eval_topk:
                hits_at_k[k] += accs[k] * B
                mrr_at_k[k]  += _mrr_at_k(out["logits_app"], view["label_app"]-1, k) * B
                ndcg_at_k[k] += _ndcg_at_k(out["logits_app"], view["label_app"]-1, k) * B


        val_loss = tot_v / max(1, n_v)
        acc_res  = {k: hits_at_k[k] / max(1, N_eval) for k in eval_topk}
        mrr_res  = {k: mrr_at_k[k]  / max(1, N_eval) for k in eval_topk}
        ndcg_res = {k: ndcg_at_k[k] / max(1, N_eval) for k in eval_topk}
        # 计算流量的 MAE,MSE,RMSE
        mae = (pred_kb - y_kb).abs().mean()
        mse = F.mse_loss(pred_kb, y_kb)
        rmse = torch.sqrt(mse)
        r2 = 1 - (mse / (y_kb.var() + 1e-6))
        print(f"Traffic Prediction - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

        def _fmt1510(d):  # 打印 acc1510/mrr1510/ndcg1510
            return "/".join(f"{d.get(k, float('nan')):.4f}" for k in (1,5,10))

        print(
            f"[Predictor {ep:03d}] val_loss={val_loss:.4f}  "
            f"acc1510={_fmt1510(acc_res)}  "
            f"mrr1510={_fmt1510(mrr_res)}  "
            f"ndcg1510={_fmt1510(ndcg_res)}"
        )
