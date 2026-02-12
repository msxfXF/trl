# OPSD 数据与训练说明

本目录包含 OPSD 数据构建与 `PrivilegedSelfDistillTrainer` 训练脚本。

## 1. 生成训练数据

输入文件：
- `0.system prompt_reduce.txt`
- `lite_doc_spec 评测_loop_agent_0210_ckpt2_导出视图 with fix.csv`

执行：

```bash
python3 /Users/bytedance/trl/data/build_opsd_dataset.py
```

输出文件：
- `opsd_student_messages.jsonl`
- `opsd_teacher_messages.jsonl`
- `opsd_combined_messages.jsonl`

其中 `opsd_combined_messages.jsonl` 的每条样本包含：
- `messages`：student 监督目标（system + user + assistant）
- `privileged_messages`：teacher 额外上下文（不重复 system/user）
- `teacher_messages`：完整 teacher 链路（用于分析对比）

## 2. 运行 OPSD 训练（默认 max_length=32k）

训练脚本：`train_opsd_privileged_gkd.py`

默认关键参数：
- `--max_length 32768`
- `--max_new_tokens 1024`
- `--model_name_or_path Qwen/Qwen2.5-0.5B-Instruct`
- `--train_file /Users/bytedance/trl/data/opsd_combined_messages.jsonl`
- `--save_strategy no`（默认关闭中途 checkpoint，避免本地包元数据缺失导致报错）

示例（先 smoke）：

```bash
HF_HUB_OFFLINE=1 python3 /Users/bytedance/trl/data/train_opsd_privileged_gkd.py \
  --sample_limit 16 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --output_dir /tmp/opsd-train-smoke-32k \
  --rollout_log_steps 1 \
  --debug_log_loss_steps 1 \
  --debug_log_grad_norm
```

示例（全量训练）：

```bash
HF_HUB_OFFLINE=1 python3 /Users/bytedance/trl/data/train_opsd_privileged_gkd.py \
  --output_dir /Users/bytedance/trl/data/opsd_privileged_gkd_ckpt \
  --save_strategy no
```

## 3. 正式训练推荐参数（GPU 集群）

下面是针对 GPU 集群（建议 A100/H100）的推荐配置，目标是先保证稳定收敛，再逐步提吞吐。

### 3.1 稳定版（建议先跑）

- `max_length=32768`
- `max_new_tokens=256`（相比 1024 更稳、显存压力更可控）
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=16`
- `learning_rate=1e-5`
- `num_train_epochs=2`
- `lmbda=1.0`
- `bf16=true`
- `save_strategy=epoch`

单机多卡示例（8 卡）：

```bash
HF_HUB_OFFLINE=1 accelerate launch --num_processes 8 \
  /Users/bytedance/trl/data/train_opsd_privileged_gkd.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_file /Users/bytedance/trl/data/opsd_combined_messages.jsonl \
  --output_dir /path/to/opsd_privileged_gkd_ckpt \
  --max_length 32768 \
  --max_new_tokens 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --lmbda 1.0 \
  --bf16 \
  --save_strategy epoch \
  --save_total_limit 5 \
  --logging_steps 10
```

### 3.2 吞吐版（稳定后再调）

在稳定版基础上可尝试：
- `gradient_accumulation_steps` 从 `16` 下调到 `8`
- `max_new_tokens` 从 `256` 上调到 `384/512`
- `learning_rate` 维持 `1e-5`，不要先加大学习率

## 4. 常见问题

1. 显存不足（OOM）：
- 降低 `--max_new_tokens`（优先）
- 降低 `--sample_limit` 做验证
- 降低 `--gradient_accumulation_steps` 以减少单次驻留激活

2. 训练速度慢：
- 先用 `--max_steps 1` 验证链路
- 关闭或减少 rollout/debug 日志（`--rollout_log_steps 0`、`--debug_log_loss_steps 0`）

3. 数据字段报错：
- 确保输入是 `opsd_combined_messages.jsonl`
- 样本必须包含 `messages` 与 `privileged_messages`

4. 需要中途 checkpoint（`save_strategy=steps/epoch`）但报 `PackageNotFoundError: trl`：
- 先执行 `pip install -e /Users/bytedance/trl`
- 或保持 `--save_strategy no`，训练结束后脚本仍会执行 `trainer.save_model(...)`
