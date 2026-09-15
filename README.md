<div align="center">

# LLMApp
### Unleashing the Power of Large Language Model for Mobile App Usage Prediction

[![Venue](https://img.shields.io/badge/IEEE-TSC%202026-blue)](https://ieeexplore.ieee.org/)
[![Status](https://img.shields.io/badge/Status-Accepted-success)](https://tong89.github.io/tongli.github.io/)

**Bo Liu · Tong Li · Miao Xiao · Beihao Xia · Zhu Xiao · Zhuo Tang · Kenli Li**

*IEEE Transactions on Services Computing, 2026 — Accepted*

</div>

---

## Overview

**LLMApp** is a hierarchical mobile app usage prediction framework that leverages large language models as semantic reasoning engines. Instead of directly treating app histories as flat sequential signals, LLMApp compresses long-term behavioral logs, interprets heterogeneous contextual information, and performs coarse-to-fine reasoning over app usage behavior.

<p align="center">
  <img src="framework.png" width="92%" alt="LLMApp framework" />
</p>

## Abstract

Mobile applications serve as gateways to many essential services in everyday life. However, service delays, unstable resource usage, and inefficient traffic handling can substantially affect the overall user experience. Predicting which application a user is likely to access next enables systems to preload applications into memory, reduce service latency, improve resource allocation, and help network operators better respond to traffic variations. In this work, we introduce **LLMApp**, a hierarchical prediction framework that leverages large language models for mobile app usage prediction. First, LLMApp employs a **Prompt Compressor** to remove redundant information from long-term app usage histories and convert them into concise behavior summaries that an LLM can effectively reason over. Second, a **Context Interpreter** encodes heterogeneous contextual signals into unified representations and projects them into the semantic space of the LLM, enabling joint reasoning over spatiotemporal and behavioral information. Finally, a **Hierarchical Predictor** exploits the semantic reasoning capability of LLMs to perform coarse-to-fine prediction of future mobile app usage. Experiments on two real-world datasets show that LLMApp consistently outperforms competitive baselines under **Top-k Accuracy (ACC@K)**, **Mean Reciprocal Rank (MRR@K)**, and **Normalized Discounted Cumulative Gain (NDCG@K)**. Additional analyses demonstrate strong robustness and scalability, effective redundancy filtering, and the ability to capture spatiotemporal patterns underlying user app usage behavior.

## Highlights

- **Prompt Compressor** removes redundant long-term usage history while retaining behavior patterns useful for prediction.
- **Context Interpreter** unifies heterogeneous contextual signals and aligns them with the LLM semantic space.
- **Hierarchical Predictor** performs coarse-to-fine semantic reasoning for next-app prediction.
- Evaluated on **Shanghai** and **Nanchang** real-world app-usage datasets.
- Uses ranking-oriented metrics including **ACC@K, MRR@K, and NDCG@K**.

## Paper & Download

The paper was accepted by **IEEE Transactions on Services Computing (TSC)** on **September 6, 2026**.

- **Publisher:** IEEE Transactions on Services Computing
- **Status:** Accepted; publisher production/indexing is in progress.
- **IEEE Xplore:** https://ieeexplore.ieee.org/

> The final article page / PDF link will be updated once the publisher completes production and indexing.

## Method

LLMApp consists of three main stages:

1. **Long-term Behavior Compression** — compresses lengthy app usage histories into concise semantic summaries while preserving predictive behavioral patterns.
2. **Contextual Semantic Alignment** — encodes temporal, spatial, and behavioral context and aligns it with the representation space of the LLM.
3. **Hierarchical App Prediction** — performs coarse-to-fine semantic reasoning to rank likely future app usage.

## Dataset

The experiments use two real-world mobile app usage datasets:

- **Shanghai Mobile App Usage Dataset** (2016)
- **Nanchang Mobile App Usage Dataset** (2022)

Original data source: https://fi.ee.tsinghua.edu.cn/appusage/

Please follow the original dataset terms and privacy requirements.

## Run

```bash
chmod +x scripts/run_shanghai.sh
bash scripts/run_shanghai.sh
```

## Citation

```bibtex
@article{liu2026llmapp,
  title   = {LLMApp: Unleashing the Power of Large Language Model for Mobile App Usage Prediction},
  author  = {Liu, Bo and Li, Tong and Xiao, Miao and Xia, Beihao and Xiao, Zhu and Tang, Zhuo and Li, Kenli},
  journal = {IEEE Transactions on Services Computing},
  year    = {2026},
  note    = {Accepted}
}
```

## Contact

For questions about the paper or code, please open an issue or contact **Bo Liu** at `liubo317@hnu.edu.cn`.
