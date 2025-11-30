# util.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
from datetime import datetime

import math
from torch.optim.lr_scheduler import LambdaLR

WEEKDAY_EN = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def parse_time_to_dow_hour(ts_raw: Any) -> Tuple[str, int]:
    s = str(int(ts_raw))
    try:
        if len(s) == 12:
            dt = datetime.strptime(s, "%y%m%d%H%M%S")
        elif len(s) == 14:
            if s[:2] in ("19","20"):
                dt = datetime.strptime(s, "%Y%m%d%H%M%S")
            else:
                dt = datetime.strptime(s, "%y%m%d%H%M%S")
        else:
            iv = int(s)
            dt = datetime.fromtimestamp(iv/1000.0 if iv > 10**12 else iv)
        return WEEKDAY_EN[dt.weekday()], dt.hour
    except Exception:
        return "Mon", 0

def topk_poi_inline(poi_counter: Dict[str, int], k: int = 5) -> str:
    if not poi_counter:
        return "POI:none"
    items = sorted(poi_counter.items(), key=lambda x: x[1], reverse=True)[:k]
    parts = [f"{name}:{cnt}" for name, cnt in items if cnt > 0]
    return ", ".join(parts) if parts else "POI:few"

# ---------- 逐元素语义提取用 prompt（两类） ----------
def build_prompt_cate_traffic_elem(cate_id: int,
                                   traffic_bin: int,
                                   cate_id2en: Dict[int, str],
                                   traffic_labels: List[str]) -> str:
    cname = cate_id2en.get(int(cate_id), f"cate#{cate_id}")
    tname = traffic_labels[traffic_bin] if 0 <= traffic_bin < len(traffic_labels) else str(traffic_bin)
    return f"Describe semantic meaning of one mobile application usage: (app#{cname}, traffic:{tname})."

def build_prompt_loc_time_elem(loc_id: int,
                               time_val: Any,
                               base_id2poi: Dict[int, Dict[str, int]],
                               topk: int = 5) -> str:
    dow, hour = parse_time_to_dow_hour(time_val)
    poi_str = topk_poi_inline(base_id2poi.get(int(loc_id), {}), k=topk)
    return f"Explain spatio-temporal context: [{dow} {hour:02d}h, {poi_str}]."

# ---------- 用户习惯（长期/近期）逐元素 prompt（app_id+traffic） ----------
# 现在：build_prompt_app_traffic_elem(app_id, traffic_bin, traffic_labels)
# 改成：多一个 traffic=None
def build_prompt_app_traffic_elem(app_id: int,
                                  traffic: float = None) -> str:
    return f"(app#{app_id}, traffic:{float(traffic):.4f})"


# ---------- 最终分类阶段的“指令”文本 ----------
def build_instruction(next_loc_id: int,
                      next_time_val: Any,
                      base_id2poi: Dict[int, Dict[str, int]],
                      include_time: bool = True,
                      topk: int = 5) -> str:
    dow, hour = parse_time_to_dow_hour(next_time_val)
    poi_str = topk_poi_inline(base_id2poi.get(int(next_loc_id), {}), k=topk)
    if include_time:
        return f"Predict the NEXT app category and traffic level for context [{dow} {hour:02d}h, {poi_str}]."
    else:
        return f"Predict the NEXT app category and traffic level for context [{poi_str}]."


# ---------- 学习率调度器（cosine decay with warmup） ----------
# ====== 损失权重：线性衰减 ======
def make_linear_decay_weight(start: float = 1.0,
                             end: float = 0.0,
                             total_steps: int = 1000,
                             warmup_steps: int = 0):
    """
    返回一个 fn(step)->weight：
      - 前 warmup_steps 线性升到 start
      - 之后从 start 线性衰减到 end（到 total_steps 为止）
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    def fn(step: int) -> float:
        step = max(0, step)
        if step < warmup_steps:
            return (start * step) / max(1, warmup_steps)
        t = min(step - warmup_steps, decay_steps)
        return end + (start - end) * (1.0 - t / decay_steps)
    return fn

# ====== 损失权重：余弦衰减 ======
def make_cosine_decay_weight(start: float = 1.0,
                             end: float = 0.0,
                             total_steps: int = 1000,
                             warmup_steps: int = 0):
    """
    返回一个 fn(step)->weight：
      - 前 warmup_steps 线性升到 start
      - 之后使用半周期余弦从 start 衰减到 end
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    def fn(step: int) -> float:
        step = max(0, step)
        if step < warmup_steps:
            return (start * step) / max(1, warmup_steps)
        t = min(step - warmup_steps, decay_steps) / decay_steps  # 0→1
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))
    return fn

# ====== 学习率调度：线性（带 warmup）======
def build_linear_warmup_decay(optimizer, total_steps: int, warmup_steps: int = 0):
    """
    用 LambdaLR 实现：warmup 线性上升，随后线性下降到 0
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        t = min(current_step - warmup_steps, decay_steps)
        return max(0.0, 1.0 - t / decay_steps)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# ====== 学习率调度：余弦（带 warmup）======
def build_cosine_warmup_decay(optimizer, total_steps: int, warmup_steps: int = 0, min_ratio: float = 0.0):
    """
    经典余弦退火到 min_ratio * base_lr
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    min_ratio = float(min_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        t = min(current_step - warmup_steps, decay_steps) / decay_steps  # 0→1
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * t))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)