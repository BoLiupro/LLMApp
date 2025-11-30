# LLMApp: Large Language Model for Mobile App Usage Prediction

## Abstract
LLMApp 是一个基于大语言模型的层次化移动应用预测框架，利用用户的时空轨迹、长期与短期使用记录以及多模态上下文信号来预测下一次使用的应用及其流量。模型包含一个用于压缩长序列噪声记录的自监督 prompt compressor、一个统一编码多源上下文信息的 context interpreter，以及一个 coarse-to-fine 的 hierarchical predictor，用于先预测类别级意图再预测具体 App 与流量。在上海与南昌两个真实数据集上的实验表明，LLMApp 在 ACC@K、MRR@K 和 NDCG@K 等指标上均显著优于现有方法。相关细节见原论文。

## Framework Overview

![Framework](framework.png)

## Dataset
模型使用两个真实世界移动应用数据集（含位置、时间、App、流量等字段）：

- Shanghai Mobile App Usage Dataset（2016）  
- Nanchang Mobile App Usage Dataset（2022）  

数据来源：  
https://fi.ee.tsinghua.edu.cn/appusage/

## Run
直接执行脚本即可运行完整流程：

```bash
chmod +x scripts/run_shanghai.sh
bash scripts/run_shanghai.sh
```

## Citation
```
@article{LLMApp2025,
  title={LLMApp: Unleashing the Power of Large Language Model for Mobile App Usage Prediction},
  author={Bo Liu and Tong Li and Miao Xiao and Beihao Xia and Zhu Xiao and Zhuo Tang and Kenli Li},
  year={2025}
}
```
