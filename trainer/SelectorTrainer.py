# trainer.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset.MyDataset import MyDataset
from model.Selector import Selector, Imputer

REQUIRED_BATCH_KEYS = [
    "app_ids", "time", "loc_ids", "app_cates", "traffic_bins"
]

@torch.no_grad()
def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch

def _ensure_batch_keys(batch):
    missing = [k for k in REQUIRED_BATCH_KEYS if k not in batch]
    if missing:
        raise KeyError(f"Batch is missing required keys: {missing}. "
                       f"Expected keys: {REQUIRED_BATCH_KEYS}")


def _sample_recon_mask(app_ids, sel_mask, pad_id=0, mlm_prob=0.15):
    """
    生成 BERT-like 的重建mask：随机挑选未被 selector 选中的位置进行遮蔽。
    - 不遮蔽 pad 位
    - 不遮蔽 sel_mask==1 的关键pattern位
    """
    with torch.no_grad():
        B, L = app_ids.shape
        cand = (app_ids != pad_id) & (sel_mask == 0)  # 只在非pad且非关键pattern处候选
        rand = torch.rand(B, L, device=app_ids.device)
        recon_mask = (rand < mlm_prob) & cand
    return recon_mask.float()  # [B,L]，1表示被遮蔽

def train_one_epoch(selector: Selector,
                    imputer: Imputer,
                    dataloader,
                    optimizer,
                    device: str,
                    mask_on: str = "masked_only",
                    mlm_prob: float = 0.15):
    selector.train()
    imputer.train()
    total_loss_sum = 0.0
    total_valid = 0

    for batch in tqdm(dataloader, desc="Train"):
        _ensure_batch_keys(batch)
        batch = _move_to_device(batch, device)

        # 1) selector 产生关键pattern位置
        sel_mask = selector(
            batch["app_ids"], batch["time"], batch["loc_ids"],
            batch["app_cates"], batch["traffic_bins"]
        )  # [B,L]
        if sel_mask.dim() == 3:
            sel_mask = sel_mask.squeeze(-1)

        # 2) 生成 BERT-like recon_mask（只mask非关键pattern）
        recon_mask = _sample_recon_mask(batch["app_ids"], sel_mask,
                                        pad_id=0, mlm_prob=mlm_prob)  # [B,L]

        # 3) imputer：输入完整序列 + sel_mask + recon_mask
        logits = imputer(
            batch["app_ids"], batch["time"], batch["loc_ids"],
            batch["app_cates"], batch["traffic_bins"],
            sel_mask, recon_mask
        )  # [B,L,V]

        # 4) 只在被遮蔽的位置监督（BERT-like）
        labels = batch["app_ids"].clone()
        # labels整体-1
        labels -= 1
        labels[recon_mask == 0] = -100  # 只训被mask位置

        # 统计有效监督的 token 数
        valid = (labels != -100)
        num_valid = int(valid.sum().item())
        if num_valid == 0:
            # 保险：若这一批意外未采到mask，就跳过（或强制采1个）
            continue

        # 把 loss 设为 sum，按 token 计
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum"
        )

        optimizer.zero_grad(set_to_none=True)
        (loss_sum / num_valid).backward()  # 梯度仍可用平均的尺度
        optimizer.step()

        total_loss_sum += float(loss_sum.item())
        total_valid += num_valid

    return total_loss_sum / max(1, total_valid)


@torch.no_grad()
def validate(selector: Selector,
             imputer: Imputer,
             dataloader,
             device: str,
             mask_on: str = "masked_only",
             mlm_prob: float = 0.15):
    selector.eval()
    imputer.eval()
    total_loss_sum = 0.0
    total_valid = 0

    for batch in tqdm(dataloader, desc="Valid"):
        _ensure_batch_keys(batch)
        batch = _move_to_device(batch, device)

        sel_mask = selector(
            batch["app_ids"], batch["time"], batch["loc_ids"],
            batch["app_cates"], batch["traffic_bins"]
        )
        if sel_mask.dim() == 3:
            sel_mask = sel_mask.squeeze(-1)

        recon_mask = _sample_recon_mask(batch["app_ids"], sel_mask,
                                        pad_id=0, mlm_prob=mlm_prob)

        logits = imputer(
            batch["app_ids"], batch["time"], batch["loc_ids"],
            batch["app_cates"], batch["traffic_bins"],
            sel_mask, recon_mask
        )

        labels = batch["app_ids"].clone()
        labels -= 1
        labels[recon_mask == 0] = -100

        valid = (labels != -100)
        num_valid = int(valid.sum().item())
        if num_valid == 0:
            continue

        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum"
        )
        total_loss_sum += float(loss_sum.item())
        total_valid += num_valid

    return total_loss_sum / max(1, total_valid)

def trainSelector(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    dataset = MyDataset(
        csv_file=args.csv_file,
        seq_len=args.long_seq_len,
        stride=getattr(args, "stride", None),
        pad_id=args.pad_id,
        keep_in_memory=getattr(args, "keep_in_memory", True),
    )

    val_ratio = getattr(args, "val_ratio", 0.05)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.selector_batch_size, shuffle=True,
        num_workers=getattr(args, "num_workers", 4), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.selector_batch_size, shuffle=False,
        num_workers=getattr(args, "num_workers", 4), pin_memory=True, drop_last=False,
    )

    selector = Selector(
        num_apps=args.app_vocab_size,
        num_locs=args.location_vocab_size,
        num_app_cate=args.num_app_cate,
        num_traffic_bin=args.num_traffic_bin,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        key_pattern_len=args.key_pattern_len,
    ).to(device)

    imputer = Imputer(
        num_apps=args.app_vocab_size,
        num_locs=args.location_vocab_size,
        num_app_cate=args.num_app_cate,
        num_traffic_bin=args.num_traffic_bin,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(selector.parameters()) + list(imputer.parameters()),
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0)
    )

    best_val = float("inf")
    for epoch in range(1, args.selector_epochs + 1):
        train_loss = train_one_epoch(selector, imputer, train_loader, optimizer, device, args.mlm_prob)
        val_loss = validate(selector, imputer, val_loader, device)
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            if getattr(args, "selector_save_path", None):
                torch.save(
                    {
                        "selector": selector.state_dict(),
                        "imputer": imputer.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    args.selector_save_path,
                )
                print(f"  ✓ Saved checkpoint to {args.selector_save_path}")
