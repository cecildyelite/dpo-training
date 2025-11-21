#!/usr/bin/env python
"""
Convert conversational BabyAI dataset (like the JSON you showed) into a DPO dataset.

Input format (JSON file):
[
  {
    "conversations": [
      {"from": "human", "loss": null,  "value": "..."},
      {"from": "gpt",   "loss": false, "value": "..."},
      {"from": "human", "loss": null,  "value": "..."},
      {"from": "gpt",   "loss": true,  "value": "Thought: ...\n\nAction:\nmove forward"},
      ...
    ],
    "item_id": "babyai_0"
  },
  ...
]

Output format (JSONL file):
{"prompt": "<full chat history up to last human>", "chosen": "<gpt answer>", "rejected": "<gpt answer with wrong Action>"}
"""

import argparse
import json
import random
from typing import List, Dict, Any


def load_conversation_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect top-level list
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of items.")
    return data


def build_prompt_from_history(history: List[Dict[str, Any]]) -> str:
    """
    Convert the conversation history (all turns before the current gpt message)
    into a single text prompt.

    We label turns as 'Human:' and 'Assistant:' so it matches chat-like prompting.
    """
    parts = []
    for msg in history:
        role = msg.get("from")
        text = msg.get("value", "")
        if role == "human":
            parts.append(f"Human: {text}")
        elif role == "gpt":
            parts.append(f"Assistant: {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def make_rejected_from_chosen(chosen: str) -> str:
    """
    Create a 'rejected' response by modifying the Action line in the chosen response.

    Assumes chosen looks like:
      Thought:
      ...
      Action:
      <some action>

    We keep the Thought the same, but replace the action line with a different,
    likely-wrong action (e.g., 'turn left' or 'turn right').
    """
    lines = chosen.splitlines()

    # Find the line "Action:" (case-insensitive)
    action_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("action:"):
            action_idx = i
            break

    # If we don't find an explicit Action: line, just append a wrong action.
    if action_idx is None:
        wrong_action = "turn left"
        return chosen + f"\n\nAction:\n{wrong_action}"

    # The actual action is usually the next non-empty line
    action_line_idx = None
    for j in range(action_idx + 1, len(lines)):
        if lines[j].strip() != "":
            action_line_idx = j
            break

    # Candidate wrong actions
    candidate_wrong_actions = [
        "turn left",
        "turn right",
        "move forward",
        "go to grey ball 1",
        "pick up grey key 1",
    ]

    if action_line_idx is None:
        # No explicit action line found; just tack on a wrong one at the end
        wrong_action = random.choice(candidate_wrong_actions)
        return chosen + f"\n\nAction:\n{wrong_action}"

    original_action = lines[action_line_idx].strip()

    # Choose a different action than the original, if possible
    choices = [a for a in candidate_wrong_actions if a.lower() != original_action.lower()]
    if not choices:
        choices = candidate_wrong_actions
    wrong_action = random.choice(choices)

    # Replace the action line only
    lines[action_line_idx] = wrong_action
    return "\n".join(lines)


def convert_conversations_to_dpo(
    data: List[Dict[str, Any]],
    max_samples: int | None = None,
) -> List[Dict[str, str]]:
    """
    Convert conversation-style BabyAI data into a list of DPO examples:
      { "prompt": ..., "chosen": ..., "rejected": ... }
    """
    dpo_examples: List[Dict[str, str]] = []

    for item in data:
        conv = item.get("conversations", [])
        # Iterate over messages, find gpt messages with loss == true
        for i, msg in enumerate(conv):
            if msg.get("from") != "gpt":
                continue
            if msg.get("loss") is not True:
                continue

            # Build prompt from history up to this message
            history = conv[:i]
            prompt = build_prompt_from_history(history)

            chosen = msg.get("value", "")
            rejected = make_rejected_from_chosen(chosen)

            dpo_examples.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

            if max_samples is not None and len(dpo_examples) >= max_samples:
                return dpo_examples

    return dpo_examples


def save_dpo_jsonl(examples: List[Dict[str, str]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the conversational BabyAI JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="babyai_conversations_dpo.jsonl",
        help="Output JSONL file for DPO examples.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of DPO examples (for debugging / subsampling).",
    )
    args = parser.parse_args()

    data = load_conversation_data(args.input)
    dpo_examples = convert_conversations_to_dpo(data, max_samples=args.max_samples)
    print(f"Created {len(dpo_examples)} DPO examples.")
    save_dpo_jsonl(dpo_examples, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
