#!/usr/bin/env python3
"""
OPSD + PrivilegedSelfDistillTrainer 训练脚本。

默认使用：
- 模型: Qwen/Qwen2.5-0.5B-Instruct
- 数据: data/opsd_combined_messages.jsonl
- max_length: 32768
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.gkd import PrivilegedGKDConfig, PrivilegedSelfDistillTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OPSD with PrivilegedSelfDistillTrainer")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="学生/教师共享模型路径。",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path("/Users/bytedance/trl/data/opsd_combined_messages.jsonl"),
        help="训练数据 jsonl（需包含 messages + privileged_messages）。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/Users/bytedance/trl/data/opsd_privileged_gkd_ckpt"),
        help="训练输出目录。",
    )
    parser.add_argument("--max_length", type=int, default=32768, help="训练上下文最大长度。")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="on-policy rollout 长度。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help=">0 时覆盖 num_train_epochs，便于快速 smoke。",
    )
    parser.add_argument("--lmbda", type=float, default=1.0, help="on-policy 概率。")
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help="仅取前 N 条样本（0 表示全量）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
        help="checkpoint 保存策略。默认 no，避免本地环境下 model card 依赖问题。",
    )
    parser.add_argument("--save_steps", type=int, default=100, help="save_strategy=steps 时生效。")
    parser.add_argument("--save_total_limit", type=int, default=3, help="最多保留多少个 checkpoint。")
    parser.add_argument("--bf16", action="store_true", help="启用 bf16 训练（推荐 GPU 集群开启）。")
    parser.add_argument("--fp16", action="store_true", help="启用 fp16 训练。")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="关闭 gradient checkpointing。")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="是否允许 transformers trust_remote_code。",
    )

    # 调试日志参数（可选）
    parser.add_argument("--rollout_log_steps", type=int, default=0)
    parser.add_argument("--rollout_log_samples", type=int, default=1)
    parser.add_argument("--rollout_log_max_new_tokens", type=int, default=128)
    parser.add_argument("--rollout_log_max_chars", type=int, default=1200)
    parser.add_argument("--debug_log_loss_steps", type=int, default=0)
    parser.add_argument("--debug_log_grad_norm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(f"train_file 不存在: {args.train_file}")

    dataset_dict = load_dataset("json", data_files={"train": str(args.train_file)})
    train_dataset = dataset_dict["train"]
    if args.sample_limit > 0:
        sample_limit = min(args.sample_limit, len(train_dataset))
        train_dataset = train_dataset.select(range(sample_limit))

    required_columns = {"messages", "privileged_messages"}
    missing = required_columns - set(train_dataset.column_names)
    if missing:
        raise ValueError(f"训练数据缺少字段: {missing}，当前字段: {train_dataset.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_config = PrivilegedGKDConfig(
        output_dir=str(args.output_dir),
        report_to="none",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        lmbda=args.lmbda,
        share_student_as_teacher=True,
        privileged_key="privileged_messages",
        seed=args.seed,
        eval_strategy="no",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        rollout_log_steps=args.rollout_log_steps,
        rollout_log_samples=args.rollout_log_samples,
        rollout_log_max_new_tokens=args.rollout_log_max_new_tokens,
        rollout_log_max_chars=args.rollout_log_max_chars,
        debug_log_loss_steps=args.debug_log_loss_steps,
        debug_log_grad_norm=args.debug_log_grad_norm,
    )

    trainer = PrivilegedSelfDistillTrainer(
        model=args.model_name_or_path,
        teacher_model=args.model_name_or_path,
        args=train_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("===== OPSD 训练配置 =====")
    print(f"train_file={args.train_file}")
    print(f"train_samples={len(train_dataset)}")
    print(f"model={args.model_name_or_path}")
    print(f"max_length={args.max_length}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"batch_size={args.per_device_train_batch_size}")
    print(f"grad_accum={args.gradient_accumulation_steps}")
    print(f"bf16={args.bf16} fp16={args.fp16}")
    print(f"save_strategy={args.save_strategy}")
    print("========================")

    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    print("TRAIN_DONE", result.metrics)


if __name__ == "__main__":
    main()
