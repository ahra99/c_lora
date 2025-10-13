# C-LoRA  
**Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models**  

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
```bibtex
@article{Rahmati2025CLORA,
  title   = {C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models},
  author  = {Amir Hossein Rahmati and Sanket Jantre and Weifeng Zhang and Yucheng Wang and Byung-Jun Yoon and Nathan M. Urban and Xiaoning Qian},
  journal = {arXiv preprint arXiv:2505.17773},
  year    = {2025}
}