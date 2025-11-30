import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

class Selector(nn.Module):
    def __init__(self, num_apps, num_locs, num_app_cate, num_traffic_bin,
                 emb_dim=128, hidden_dim=256, key_pattern_len=20):
        super().__init__()
        self.app_emb = nn.Embedding(num_apps+1, emb_dim)
        self.loc_emb = nn.Embedding(num_locs+1, emb_dim)
        self.week_emb = nn.Embedding(7, emb_dim)
        self.hour_emb = nn.Embedding(24, emb_dim)
        self.cate_emb = nn.Embedding(num_app_cate+1, emb_dim)
        self.traffic_emb = nn.Embedding(num_traffic_bin+1, emb_dim)

        # self.scorer = nn.Linear(emb_dim * 5, 1)   # app + loc + time + cate + traffic
        feat_dim = emb_dim * 5
        self.scorer = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.key_pattern_len = key_pattern_len

    def forward(self, app_ids, time, loc_ids, app_cates, traffic_bins):
        # embeddings
        app_e  = self.app_emb(app_ids)
        loc_e  = self.loc_emb(loc_ids)
        cate_e = self.cate_emb(app_cates)
        traf_e = self.traffic_emb(traffic_bins)

        # 解析星期与小时（到小时即可）
        flat_ts = time.reshape(-1).detach().cpu().numpy()
        dows, hours = [], []
        for ts in flat_ts:
            ts_str = str(int(ts))
            dt = datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            dows.append(dt.weekday())
            hours.append(dt.hour)
        dows  = torch.tensor(dows,  device=time.device).view_as(time)
        hours = torch.tensor(hours, device=time.device).view_as(time)

        time_e = self.week_emb(dows) + self.hour_emb(hours)

        # 融合
        h = torch.cat([app_e, loc_e, cate_e, traf_e, time_e], dim=-1)  # [B,L,5*emb]
        scores = self.scorer(h).squeeze(-1)  # [B,L]
        

        # 选 top-K（加 k 保护）
        k = min(self.key_pattern_len, scores.size(1))
        topk = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.float)
        mask.scatter_(1, topk.indices, 1.0)

        return mask  # [B, L]，1表示选中

class SelectorTransformer(nn.Module):
    def __init__(self, num_apps, num_locs, num_app_cate, num_traffic_bin,
                 emb_dim=128, hidden_dim=256, n_layers=2, key_pattern_len=20):
        super().__init__()

        self.app_emb = nn.Embedding(num_apps+1, emb_dim)
        self.loc_emb = nn.Embedding(num_locs+1, emb_dim)
        self.week_emb = nn.Embedding(7, emb_dim)
        self.hour_emb = nn.Embedding(24, emb_dim)
        self.cate_emb = nn.Embedding(num_app_cate+1, emb_dim)
        self.traffic_emb = nn.Embedding(num_traffic_bin+1, emb_dim)

        feat_dim = emb_dim * 5
        self.input_proj = nn.Linear(feat_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 每个位置输出一个 score
        self.score_head = nn.Linear(hidden_dim, 1)
        self.key_pattern_len = key_pattern_len

    def forward(self, app_ids, time, loc_ids, app_cates, traffic_bins):
        # embedding 同你原来一样
        app_e  = self.app_emb(app_ids)
        loc_e  = self.loc_emb(loc_ids)
        cate_e = self.cate_emb(app_cates)
        traf_e = self.traffic_emb(traffic_bins)

        # 时间
        flat_ts = time.reshape(-1).detach().cpu().numpy()
        dows, hours = [], []
        for ts in flat_ts:
            dt = datetime.datetime.strptime(str(int(ts)), "%Y%m%d%H%M%S")
            dows.append(dt.weekday()); hours.append(dt.hour)
        dows  = torch.tensor(dows,  device=time.device).view_as(time)
        hours = torch.tensor(hours, device=time.device).view_as(time)
        time_e = self.week_emb(dows) + self.hour_emb(hours)

        x = torch.cat([app_e, loc_e, cate_e, traf_e, time_e], dim=-1)
        x = self.input_proj(x)

        h = self.encoder(x)                         # [B,L,H]
        scores = self.score_head(h).squeeze(-1)     # [B,L]

        k = min(self.key_pattern_len, scores.size(1))
        topk = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk.indices, 1.0)
        return mask


class Imputer(nn.Module):
    def __init__(self, num_apps, num_locs, num_app_cate, num_traffic_bin,
                 emb_dim=128, hidden_dim=256, n_layers=2):
        super().__init__()
        self.app_emb = nn.Embedding(num_apps+1, emb_dim)
        self.loc_emb = nn.Embedding(num_locs+1, emb_dim)
        self.week_emb = nn.Embedding(7, emb_dim)
        self.hour_emb = nn.Embedding(24, emb_dim)
        self.cate_emb = nn.Embedding(num_app_cate+1, emb_dim)
        self.traffic_emb = nn.Embedding(num_traffic_bin+1, emb_dim)

        # self.input_proj = nn.Linear(emb_dim * 5, hidden_dim)
        feat_dim = emb_dim * 5
        self.input_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pred_head = nn.Linear(hidden_dim, num_apps)

    def forward(self, app_ids, time, loc_ids, app_cates, traffic_bins,
                sel_mask, recon_mask=None):
        """
        sel_mask: [B,L]，1为选中的关键pattern（作为注意力的可用上下文）
        recon_mask: [B,L]，1为被遮蔽、需要重建的位置
        """
        # embeddings
        app_e  = self.app_emb(app_ids)
        loc_e  = self.loc_emb(loc_ids)
        cate_e = self.cate_emb(app_cates)
        traf_e = self.traffic_emb(traffic_bins)

        # 到小时
        flat_ts = time.view(-1).cpu().numpy()
        dows, hours = [], []
        for ts in flat_ts:
            ts_str = str(int(ts))
            dt = datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            dows.append(dt.weekday())
            hours.append(dt.hour)
        dows  = torch.tensor(dows,  device=time.device).view_as(time)
        hours = torch.tensor(hours, device=time.device).view_as(time)
        time_e = self.week_emb(dows) + self.hour_emb(hours)

        x = torch.cat([app_e, loc_e, cate_e, traf_e, time_e], dim=-1)  # [B,L,5*emb]

        # BERT-like：把需要重建的位置“遮蔽”，用0向量替代
        if recon_mask is not None:
            x = x * (1.0 - recon_mask.unsqueeze(-1).float())

        x = self.input_proj(x) 

        # 只允许 attend 到被 selector 选中的关键pattern（其余位置作为KV被屏蔽）
        # True 表示被 mask（忽略），因此取 sel_mask==0
        src_key_padding_mask = (sel_mask == 0)  # encoder不能看sel_mask=0（不是key pattern的位置）
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        logits = self.pred_head(h)  # [B,L,num_apps]
        return logits

