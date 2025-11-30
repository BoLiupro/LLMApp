# dataset.py
# -*- coding: utf-8 -*-
import os
import glob
from typing import List, Union, Optional, Sequence, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# -------- 基础工具 --------
def _list_csv_files(src: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(src, (list, tuple)):
        files = []
        for s in src:
            files += _list_csv_files(s)
        return sorted(list(dict.fromkeys(files)))
    if os.path.isdir(src):
        return sorted(glob.glob(os.path.join(src, "*.csv")))
    if any(ch in src for ch in ["*", "?", "[", "]"]):
        return sorted(glob.glob(src))
    if os.path.isfile(src):
        return [src]
    raise ValueError(f"Invalid csv_file argument: {src}")


# -------- 常量：严格列名与顺序 --------
COLUMN_ORDER = [
    "user_id",       # int
    "time",          # int(YYYYMMDDhhmmss 或 Unix秒)
    "location",      # int
    "app_id",        # int
    "app_cate_id",   # int
    "traffic",       # int/float
    "traffic_bin",   # int  (若出现重复列名，会自动去重)
]

DTYPES_DEFAULT = {
    "user_id": "int64",
    "time": "int64",
    "location": "int64",
    "app_id": "int64",
    "app_cate_id": "int64",
    "traffic": "float64",
    "traffic_bin": "int64",
}


class MyDataset(Dataset):
    """
    从一个或多个 CSV 读取，严格使用如下列名（并按此顺序）：
      user_id,time,location,app_id,app_cate_id,traffic,traffic_bin
    产出：
      app_ids, loc_ids, time, app_cates, traffic_bins, dayofweek, hour, minute, attn_mask
    """
    REQUIRED = COLUMN_ORDER  # 全部必需

    def __init__(
        self,
        csv_file: Union[str, Sequence[str]],
        seq_len: int = 64,
        stride: Optional[int] = None,
        pad_id: int = 0,
        dtypes: Optional[Dict[str, str]] = None,
        keep_in_memory: bool = True,
    ):
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else int(seq_len)
        self.pad_id = int(pad_id)
        self.keep_in_memory = bool(keep_in_memory)

        paths = _list_csv_files(csv_file)
        if not paths:
            raise FileNotFoundError(f"No CSV files found for {csv_file}")

        # 合并 dtype
        read_dtypes = DTYPES_DEFAULT.copy()
        if dtypes:
            read_dtypes.update(dtypes)

        dfs = []
        for p in paths:
            df = pd.read_csv(p, dtype=read_dtypes)
            # 处理可能重复的 traffic_bin 列（如 'traffic_bin.1'）
            if "traffic_bin" not in df.columns:
                # 有时 pandas 会把重复的同名列后缀成 .1/.2
                tb_dup = [c for c in df.columns if c.split(".")[0] == "traffic_bin"]
                if tb_dup:
                    df = df.rename(columns={tb_dup[0]: "traffic_bin"})
            # 丢弃多余的重复列
            extra_tb = [c for c in df.columns if c.startswith("traffic_bin.") and c != "traffic_bin"]
            if extra_tb:
                df = df.drop(columns=extra_tb, errors="ignore")

            # 检查必需列
            missing = [c for c in self.REQUIRED if c not in df.columns]
            if missing:
                raise ValueError(f"{p} missing required columns: {missing}")

            # 只保留并按固定顺序排列
            df = df[COLUMN_ORDER].copy()

            dfs.append(df)

        # 组装为 user->序列
        if self.keep_in_memory:
            all_df = pd.concat(dfs, ignore_index=True)
            # 全量排序：user_id -> time
            all_df = all_df.sort_values(["user_id", "time"]).reset_index(drop=True)
            self._df = all_df  # 便于调试

            user2rows: Dict[int, list] = defaultdict(list)
            for r in all_df.itertuples(index=False):
                user2rows[getattr(r, "user_id")].append(
                    (
                        getattr(r, "app_id"),
                        getattr(r, "time"),
                        getattr(r, "location"),
                        getattr(r, "app_cate_id"),
                        getattr(r, "traffic_bin"),
                        getattr(r, "traffic"),
                    )
                )
        else:
            user2rows: Dict[int, list] = defaultdict(list)
            for df in dfs:
                df = df.sort_values(["user_id", "time"])
                for r in df.itertuples(index=False):
                    user2rows[getattr(r, "user_id")].append(
                        (
                            getattr(r, "app_id"),
                            getattr(r, "time"),
                            getattr(r, "location"),
                            getattr(r, "app_cate_id"),
                            getattr(r, "traffic_bin"),
                            getattr(r, "traffic"),
                        )
                    )

        # 构建固定长度滑窗样本
        self.samples = []
        for _uid, seq in user2rows.items():
            n = len(seq)
            if n == 0:
                continue
            for start in range(0, max(1, n - self.seq_len + 1), self.stride):
                end = start + self.seq_len
                chunk = seq[start:end]
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]

        L = self.seq_len
        app_ids = np.full(L, self.pad_id, dtype=np.int64)
        loc_ids = np.full(L, self.pad_id, dtype=np.int64)
        times = np.full(L, self.pad_id, dtype=np.int64)
        app_cates = np.full(L, self.pad_id, dtype=np.int64)
        traffic_bins = np.full(L, self.pad_id, dtype=np.int64)
        traffic = np.zeros(L, dtype=np.float32)

        for i, (app, ts, loc, cate, tbin, tval) in enumerate(seq[:L]):
            app_ids[i] = app
            times[i] = ts
            loc_ids[i] = loc
            app_cates[i] = cate
            traffic_bins[i] = tbin
            traffic[i] = float(tval)

        return {
            "app_ids": torch.tensor(app_ids, dtype=torch.long),
            "time": torch.tensor(times, dtype=torch.long),
            "loc_ids": torch.tensor(loc_ids, dtype=torch.long),
            "app_cates": torch.tensor(app_cates, dtype=torch.long),
            "traffic_bins": torch.tensor(traffic_bins, dtype=torch.long),
            "traffic": torch.tensor(traffic/1024, dtype=torch.float32),
        }
