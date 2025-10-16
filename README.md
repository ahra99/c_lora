# C-LoRA  
This repository contains codes for ***Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models***  
**NeurIPS 2025**

[Paper (arXiv)](https://arxiv.org/pdf/2505.17773)  

---

## Overview  

C-LoRA is a **parameter-efficient**, **uncertainty-aware** fine-tuning method for large language models (LLMs) in few-shot or data-scarce settings. The key idea is to introduce **input-dependent (contextual) uncertainty modeling** in the LoRA adapters, thereby yielding better calibrated predictive uncertainties and reducing overconfidence.  

Unlike prior Bayesian LoRA or mean-field approaches, C-LoRA (1) uses a **lightweight factorization** to reduce complexity, and (2) integrates a small **contextual module** that conditions the posterior distribution of adapter parameters on each input sample.  

The method achieves strong performance on calibration metrics (like ECE, NLL) while maintaining competitive accuracy across reasoning benchmarks and showing robustness under distribution shift.  [oai_citation:1‡arXiv](https://arxiv.org/pdf/2505.17773)  

---

## Installation & Usage

Install the environment (env name need to be modified). 

```bash 
conda env create -f env.yaml
conda activate yourEnv
```

For experiments, use the script provided in scripts directory. 



---
## Citation
If you find this work useful, please cite our NeurIPS 2025 paper:
```bibtex
@inproceedings{
anonymous2025clora,
title={C-Lo{RA}: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models},
author={Anonymous},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=siPeAstQLq}
}
