# ShareLoRA

This repo supports the paper "ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation", an effort to democratize access to LLM research.


**ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation** 
<br> *Yurun Song, Junchen Zhao, Ian G. Harris, Sangeetha Abdu Jyothi*<br>
[Paper](https://arxiv.org/abs/2305.14314): https://arxiv.org/pdf/2406.10785<br> 



## Overview
In this paper, we introduce Shared Low Rank Adaptation (ShareLoRA), a Large Language Model (LLM) fine-tuning technique that balances parameter efficiency, adaptability, and robustness without compromising performance. By strategically sharing the low-rank weight matrices across different layers, ShareLoRA achieves 44\% to 96\% reduction in trainable parameters compared to standard LoRA, alongside a substantial decrease in memory overhead. This efficiency gain scales with model size, making ShareLoRA particularly advantageous for resource-constrained environments. Importantly, ShareLoRA not only maintains model performance but also exhibits robustness in both classification and generation tasks across diverse models, including RoBERTa, GPT-2, and LLaMA series (1, 2, and 3).  It consistently outperforms LoRA in zero-shot, few-shot, and continual fine-tuning scenarios, achieving up to 1.2\% average accuracy improvement, and enhanced generalization across domains. In continual learning settings, ShareLoRA achieves 1.2\% higher accuracy on GSM8K, 0.6\% on HumanEval, and 0.5\% on both MMLU and MMLU-Pro. Our results demonstrate that ShareLoRA supports high-quality fine-tuning while offering strong generalization and continual adaptation across various model scales and diverse tasks.



## GLUE and E2E Benchmark

1. Clone the official [LoRA](https://github.com/microsoft/LoRA) repository and install the required dependencies.

2. Add `run_glue_share.py` to the `examples/NLU/examples/text-classification` directory, and update the script instructions in the `examples/NLU` folder from the LoRA repository.

3. Add `gpt2_share_ft.py` to the `examples/NLG/src` directory, and update the script instructions in the `examples/NLG` README file from the LoRA repository.


## ShareLoRA LLaMA experiments


1. Clone the official [QLoRA](https://github.com/artidoro/qlora)repository and install the required dependencies.

2. Run training scripts located in `sharelora/script/share` folder.

	```bash
	cd sharelora
	bash scripts/share/finetune_alpaca_llama3_8b_full.sh
	```

3. Run the evaluation or generation scripts located in `sharelora/script/share/eval` after obtaining the training checkpoints.

	```bash
	bash scripts/share/eval/MMLU_evaluate_llama3_8b_full.sh
	```


## Citation

```bibtex
@article{song2024sharelora,
  title={ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation},
  author={Song, Yurun and Zhao, Junchen and Harris, Ian G and Jyothi, Sangeetha Abdu},
  journal={arXiv preprint arXiv:2406.10785},
  year={2024}
}
```