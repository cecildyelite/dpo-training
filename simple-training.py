# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
```

# LoRA:
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```
"""

"""
python simple-training.py     --dataset_name ./babyai_gotolocal_dpo.jsonl     --model_name_or_path ./my_model     --learning_rate 5.0e-6     --num_train_epochs 1     --per_device_train_batch_size 2     --max_steps 1000     --gradient_accumulation_steps 8     --gradient_checkpointing     --eval_strategy no  --output_dir output_model    --no_remove_unused_columns     --use_peft     --lora_r 32     --lora_alpha 16
"""

import argparse
import os

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import LoraConfig

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logger = logging.get_logger(__name__)

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

torch.cuda.empty_cache()

def main(script_args, training_args, model_args, dataset_args):

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        low_cpu_mem_usage=True,
        **model_kwargs
    )
    
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load the dataset
    # if dataset_args.datasets and script_args.dataset_name:
    #     logger.warning(
    #         "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
    #         "dataset and `dataset_name` will be ignored."
    #     )
    #     dataset = get_dataset(dataset_args)
    # elif dataset_args.datasets and not script_args.dataset_name:
    #     dataset = get_dataset(dataset_args)
    # elif not dataset_args.datasets and script_args.dataset_name:
    #     dataset = load_dataset(
    #         script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
    #     )
    # else:
    #     raise ValueError("Either `datasets` or `dataset_name` must be provided.")
    
    dataset = load_dataset("json", data_files="babyai_gotoobjmazeopen_dpo.jsonl", split="train")

    train_val_split = dataset.train_test_split(test_size=0.15, seed=42)
    val_test_split = train_val_split["test"].train_test_split(test_size=0.45, seed=42)

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_val_split['train'],
        eval_dataset=val_test_split['train'],
        peft_config=peft_config,
    )

    trainer.train()

    trainer.accelerator.print("✅ Training completed.")

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args) 