from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "training" / "configs" / "lora_config.example.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for the interview answer analysis model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load datasets/tokenizer/model and initialize trainer, but do not start training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them with: pip install -e .[training]"
        ) from exc

    print(f"Loading train dataset: {config['train_file']}")
    train_dataset = load_dataset("json", data_files=str(config["train_file"]), split="train")
    print(f"Loading eval dataset: {config['eval_file']}")
    eval_dataset = load_dataset("json", data_files=str(config["eval_file"]), split="train")

    print(f"Loading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {config['base_model']}")
    model_kwargs = _build_model_kwargs(config, torch, BitsAndBytesConfig)
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        **model_kwargs,
    )
    if config.get("fail_on_cpu_offload", False):
        _fail_if_cpu_offload(model)
    if config.get("gradient_checkpointing", False):
        model.config.use_cache = False
    if config.get("load_in_4bit", False) or config.get("load_in_8bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", False),
        )

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = _build_sft_config(SFTConfig, config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if args.dry_run:
        print("Dry run completed: trainer initialized successfully, training was not started.")
        print(f"Train records: {len(train_dataset)}")
        print(f"Eval records: {len(eval_dataset)}")
        print(f"Output dir: {config['output_dir']}")
        return

    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Training finished. Adapter saved to {config['output_dir']}")


def _build_model_kwargs(config: dict[str, Any], torch, bitsandbytes_config_cls) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if config.get("load_in_4bit", False):
        kwargs["quantization_config"] = bitsandbytes_config_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=_torch_dtype(torch, config.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        )
        kwargs["device_map"] = config.get("device_map", "auto")
        return kwargs

    if config.get("load_in_8bit", False):
        kwargs["quantization_config"] = bitsandbytes_config_cls(load_in_8bit=True)
        kwargs["device_map"] = config.get("device_map", "auto")
        return kwargs

    if "torch_dtype" in config:
        kwargs["torch_dtype"] = _torch_dtype(torch, config["torch_dtype"])
    if "device_map" in config:
        kwargs["device_map"] = config["device_map"]
    return kwargs


def _fail_if_cpu_offload(model) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return
    offloaded = {name: device for name, device in device_map.items() if str(device).lower() in {"cpu", "disk"}}
    if offloaded:
        sample = dict(list(offloaded.items())[:8])
        raise SystemExit(
            "Model was partially offloaded to CPU/disk. This would make training extremely slow. "
            f"Reduce max_seq_length/LoRA rank or run on a GPU with more VRAM. Offloaded modules: {sample}"
        )


def _torch_dtype(torch, value: str):
    normalized = str(value).lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def _build_sft_config(sft_config_cls, config: dict[str, Any]):
    signature = inspect.signature(sft_config_cls.__init__)
    available = set(signature.parameters)

    values: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "eval_strategy": config.get("eval_strategy", "epoch"),
        "save_strategy": config.get("save_strategy", "epoch"),
        "logging_steps": config["logging_steps"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "warmup_ratio": config["warmup_ratio"],
        "fp16": config.get("fp16", False),
        "bf16": config.get("bf16", False),
        "report_to": "none",
        "max_steps": config.get("max_steps", -1),
        "gradient_checkpointing": config.get("gradient_checkpointing", False),
        "dataset_kwargs": {"skip_prepare_dataset": False},
    }
    for optional_key in (
        "optim",
        "save_total_limit",
        "eval_steps",
        "save_steps",
        "lr_scheduler_type",
        "weight_decay",
        "max_grad_norm",
    ):
        if optional_key in config:
            values[optional_key] = config[optional_key]

    # TRL 0.18+ reads sequence length from SFTConfig, not from SFTTrainer.__init__.
    if "max_length" in available:
        values["max_length"] = config["max_seq_length"]
    elif "max_seq_length" in available:
        values["max_seq_length"] = config["max_seq_length"]

    filtered = {key: value for key, value in values.items() if key in available}
    return sft_config_cls(**filtered)


if __name__ == "__main__":
    main()

