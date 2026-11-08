---
library_name: peft
license: other
base_model: Qwen/Qwen3-1.7B-Base
tags:
- base_model:adapter:Qwen/Qwen3-1.7B-Base
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: train_2026-11-08-22-39-50
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2026-11-08-22-39-50

This model is a fine-tuned version of [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) on the support_tickets dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.8.0
- Pytorch 2.15.0
- Datasets 4.0.0
- Tokenizers 0.22.2