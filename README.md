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

**LLMApp** is a hierarchical mobile app usage prediction framework that uses large language models as semantic reasoning engines. The framework is designed to model long-term usage histories, heterogeneous contextual information, and coarse-to-fine app-selection behavior in a unified prediction pipeline.

<p align="center">
  <img src="framework.png" width="92%" alt="LLMApp framework" />
</p>

## Highlights

- **Prompt Compressor** removes redundant long-term usage history while preserving behavior patterns useful for prediction.
- **Context Interpreter** projects heterogeneous contextual signals into the LLM semantic space.
- **Hierarchical Predictor** performs coarse-to-fine semantic reasoning for mobile app usage prediction.
- Evaluated on **Shanghai** and **Nanchang** real-world app-usage datasets.
- Uses ranking-oriented metrics including **ACC@K, MRR@K, and NDCG@K**.

## Paper

The paper was accepted by **IEEE Transactions on Services Computing (TSC)** in September 2026.

> The final IEEE Xplore record may be updated during the publisher's production/indexing process. Please use the publication title and author list above when searching IEEE Xplore if the final article page is not yet available.

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
