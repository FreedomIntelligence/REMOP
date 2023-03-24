# Introduction
Code for the paper: **Modular Retrieval for Generalization and Interpretation** \[[link](https://arxiv.org/abs/2303.13419)\]. REMOP (Retrieval with modular prompt tuning) is a simple-yet-effective implementation of modular retrieval, and the code is mainly develped based on [DPTDR](https://github.com/FreedomIntelligence/DPTDR.git), [coCondenser](https://github.com/luyug/Dense.git) and [P-tuning v2](https://github.com/THUDM/P-tuning-v2).


## Installation

For environment, please run `sh install_env.sh` in a clean conda environment of `python>=3.7`.
Then just run `pip install -e`.

## Reproduction for BEIR
Please refer to `examples/condener_beir`.

## References

```bibtex
@misc{liang2023modular,
      title={Modular Retrieval for Generalization and Interpretation}, 
      author={Juhao Liang and Chen Zhang and Zhengyang Tang and Jie Fu and Dawei Song and Benyou Wang},
      year={2023},
      eprint={2303.13419},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```


