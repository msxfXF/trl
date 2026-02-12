#!/usr/bin/env python3
"""
根据评测导出 CSV 生成 OPSD 训练数据（student / teacher / combined）。

默认输入：
- system prompt: data/0.system prompt_reduce.txt
- source csv: data/lite_doc_spec 评测_loop_agent_0210_ckpt2_导出视图 with fix.csv

默认输出：
- data/opsd_student_messages.jsonl
- data/opsd_teacher_messages.jsonl
- data/opsd_combined_messages.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "null"}:
        return None
    return text


def first_non_empty(row: pd.Series, keys: list[str]) -> str | None:
    for key in keys:
        value = normalize_text(row.get(key))
        if value:
            return value
    return None


def append_if_exist(messages: list[dict[str, str]], role: str, content: str | None) -> bool:
    if not content:
        return False
    messages.append({"role": role, "content": content})
    return True


def build_teacher_messages(
    system_prompt: str,
    user_input: str,
    row: pd.Series,
    include_final_assistant: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, bool]]:
    """
    返回：
    - teacher_messages_full：包含 system + user + 后续链路（按你的 teacher 格式）
    - privileged_messages：不重复 system/user 的链路（更适配 PrivilegedGKD）
    - flags：每个阶段是否被纳入
    """
    teacher_messages_full: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    privileged_messages: list[dict[str, str]] = []

    flags = {
        "has_generate": False,
        "has_validate": False,
        "has_fix_0": False,
        "has_judge_0": False,
        "has_fix_1": False,
        "has_judge_1": False,
        "has_fix_2": False,
        "has_judge_2": False,
        "has_final": False,
    }

    # assistant: generate_text
    generate_text = normalize_text(row.get("generate_text"))
    if append_if_exist(teacher_messages_full, "assistant", generate_text):
        append_if_exist(privileged_messages, "assistant", generate_text)
        flags["has_generate"] = True

    # user: validate_text
    validate_text = normalize_text(row.get("validate_text"))
    if not append_if_exist(teacher_messages_full, "user", validate_text):
        return teacher_messages_full, privileged_messages, flags
    append_if_exist(privileged_messages, "user", validate_text)
    flags["has_validate"] = True

    # assistant: fix_text
    fix_text = normalize_text(row.get("fix_text"))
    if not append_if_exist(teacher_messages_full, "assistant", fix_text):
        return teacher_messages_full, privileged_messages, flags
    append_if_exist(privileged_messages, "assistant", fix_text)
    flags["has_fix_0"] = True

    # user: judge_0_text -> assistant: fix_1_text
    judge_0_text = normalize_text(row.get("judge_0_text"))
    if not append_if_exist(teacher_messages_full, "user", judge_0_text):
        return teacher_messages_full, privileged_messages, flags
    append_if_exist(privileged_messages, "user", judge_0_text)
    flags["has_judge_0"] = True

    fix_1_text = normalize_text(row.get("fix_1_text"))
    if not append_if_exist(teacher_messages_full, "assistant", fix_1_text):
        return teacher_messages_full, privileged_messages, flags
    append_if_exist(privileged_messages, "assistant", fix_1_text)
    flags["has_fix_1"] = True

    # user: judge_1_text -> assistant: fix_2_text
    judge_1_text = normalize_text(row.get("judge_1_text"))
    if not append_if_exist(teacher_messages_full, "user", judge_1_text):
        return teacher_messages_full, privileged_messages, flags
    append_if_exist(privileged_messages, "user", judge_1_text)
    flags["has_judge_1"] = True

    fix_2_text = normalize_text(row.get("fix_2_text"))
    if not append_if_exist(teacher_messages_full, "assistant", fix_2_text):
        return teacher_messages_full, privileged_messages, flags
    append_if_exist(privileged_messages, "assistant", fix_2_text)
    flags["has_fix_2"] = True

    # user: judge_2_text
    judge_2_text = normalize_text(row.get("judge_2_text"))
    if not append_if_exist(teacher_messages_full, "user", judge_2_text):
        return teacher_messages_full, privileged_messages, flags
    append_if_exist(privileged_messages, "user", judge_2_text)
    flags["has_judge_2"] = True

    if include_final_assistant:
        final_text = normalize_text(row.get("final_text"))
        if append_if_exist(teacher_messages_full, "assistant", final_text):
            append_if_exist(privileged_messages, "assistant", final_text)
            flags["has_final"] = True

    return teacher_messages_full, privileged_messages, flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 OPSD 训练数据")
    parser.add_argument(
        "--system_prompt_path",
        type=Path,
        default=Path("/Users/bytedance/trl/data/0.system prompt_reduce.txt"),
        help="system prompt 文件路径",
    )
    parser.add_argument(
        "--source_csv",
        type=Path,
        default=Path("/Users/bytedance/trl/data/lite_doc_spec 评测_loop_agent_0210_ckpt2_导出视图 with fix.csv"),
        help="源 CSV 路径",
    )
    parser.add_argument(
        "--student_out",
        type=Path,
        default=Path("/Users/bytedance/trl/data/opsd_student_messages.jsonl"),
        help="student 数据输出 jsonl",
    )
    parser.add_argument(
        "--teacher_out",
        type=Path,
        default=Path("/Users/bytedance/trl/data/opsd_teacher_messages.jsonl"),
        help="teacher 数据输出 jsonl（完整 teacher 链路）",
    )
    parser.add_argument(
        "--combined_out",
        type=Path,
        default=Path("/Users/bytedance/trl/data/opsd_combined_messages.jsonl"),
        help="combined 输出 jsonl（messages + privileged_messages + teacher_messages）",
    )
    parser.add_argument(
        "--include_final_assistant",
        action="store_true",
        help="teacher 链路末尾是否附加 assistant: final_text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    system_prompt = args.system_prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError("system prompt 为空，请检查输入文件。")

    df = pd.read_csv(args.source_csv)

    # 学生端目标默认取“最终可用答案”：final > fix_2 > fix_1 > fix > generate
    assistant_candidate_keys = ["final_text", "fix_2_text", "fix_1_text", "fix_text", "generate_text"]

    total_rows = len(df)
    skipped_missing_input = 0
    skipped_missing_target = 0

    stage_stats = {
        "has_generate": 0,
        "has_validate": 0,
        "has_fix_0": 0,
        "has_judge_0": 0,
        "has_fix_1": 0,
        "has_judge_1": 0,
        "has_fix_2": 0,
        "has_judge_2": 0,
        "has_final": 0,
    }

    student_records: list[dict] = []
    teacher_records: list[dict] = []
    combined_records: list[dict] = []

    for _, row in df.iterrows():
        sample_id = normalize_text(row.get("自动编号"))
        user_input = normalize_text(row.get("input"))
        if not user_input:
            skipped_missing_input += 1
            continue

        student_target = first_non_empty(row, assistant_candidate_keys)
        if not student_target:
            skipped_missing_target += 1
            continue

        student_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": student_target},
        ]

        teacher_messages, privileged_messages, flags = build_teacher_messages(
            system_prompt=system_prompt,
            user_input=user_input,
            row=row,
            include_final_assistant=args.include_final_assistant,
        )
        for k, v in flags.items():
            stage_stats[k] += int(v)

        student_record = {
            "id": sample_id,
            "messages": student_messages,
        }
        teacher_record = {
            "id": sample_id,
            "messages": teacher_messages,
        }
        combined_record = {
            "id": sample_id,
            "messages": student_messages,
            "privileged_messages": privileged_messages,
            "teacher_messages": teacher_messages,
        }

        student_records.append(student_record)
        teacher_records.append(teacher_record)
        combined_records.append(combined_record)

    for path, records in [
        (args.student_out, student_records),
        (args.teacher_out, teacher_records),
        (args.combined_out, combined_records),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("OPSD 数据生成完成")
    print(f"- total_rows: {total_rows}")
    print(f"- kept_rows: {len(combined_records)}")
    print(f"- skipped_missing_input: {skipped_missing_input}")
    print(f"- skipped_missing_target: {skipped_missing_target}")
    print("- stage_stats:")
    for key, value in stage_stats.items():
        print(f"  - {key}: {value}")
    print("- outputs:")
    print(f"  - student: {args.student_out}")
    print(f"  - teacher: {args.teacher_out}")
    print(f"  - combined: {args.combined_out}")


if __name__ == "__main__":
    main()
