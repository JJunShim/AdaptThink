# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import math
from collections import defaultdict

import numpy as np
import torch
from sympy import primenu
from tqdm import tqdm

from verl import DataProto

from .adapt_think_rm import adapt_think_rm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AdaptThinkRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        is_training=True,
        ref_result_file=None,
        eps=1e-6,
        max_response_length=512,
        nothinking_bonus=0,
        length_bonus=0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or adapt_think_rm
        self.reward_fn_key = reward_fn_key
        self.is_training = is_training
        self.eps = eps
        self.max_response_length = max_response_length
        self.nothinking_bonus = nothinking_bonus
        self.length_bonus = length_bonus
        if ref_result_file:
            self.problem2ref_metrics = {
                js["problem"].strip(): js["metrics"]
                for js in tqdm(
                    json.load(open(ref_result_file, "r")), desc="LOADING REF METRICS"
                )
            }
        if self.is_training:
            print(f"\n\nNOTHINKING BONUS: {self.nothinking_bonus}\n\n")

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        id2scores = {"nothinking": defaultdict(list), "thinking": defaultdict(list)}
        id2mean_acc = defaultdict(dict)
        id2mean_len = defaultdict(dict)
        id2std_len = defaultdict(dict)
        all_scores = []
        uid2ref_metrics = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = (
                data_item.batch["attention_mask"][:prompt_length].sum().item()
            )
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = (
                data_item.batch["attention_mask"][prompt_length:].sum().item()
            )
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            uid = data_item.non_tensor_batch["uid"] if self.is_training else "validate"
            enforce_nothinking = data_item.batch["enforce_nothinking"].item()
            is_nothinking = response_str.strip().startswith("</think>")
            score.update(
                {
                    "response_length": valid_response_length,
                    "ground_truth": str(ground_truth),
                    "enforce_nothinking": enforce_nothinking,
                    "is_nothinking": is_nothinking,
                }
            )
            if self.is_training:
                if enforce_nothinking:
                    score.update(
                        {
                            "nothinking_response_length": valid_response_length,
                            "nothinking_acc": score["acc"],
                            "thinking_response_length": None,
                            "thinking_acc": None,
                        }
                    )
                else:
                    score.update(
                        {
                            "nothinking_response_length": None,
                            "nothinking_acc": None,
                            "thinking_response_length": valid_response_length,
                            "thinking_acc": score["acc"],
                        }
                    )
                problem = (
                    prompt_str.split("<｜User｜>")[1]
                    .split("<｜Assistant｜>")[0]
                    .strip()
                )
                assert problem in self.problem2ref_metrics, problem
                # if problem not in self.problem2ref_metrics:
                #     continue
                uid2ref_metrics[uid] = self.problem2ref_metrics[problem]
            else:
                if is_nothinking:
                    score.update(
                        {
                            "nothinking_response_length": valid_response_length,
                            "nothinking_acc": score["acc"],
                            "thinking_response_length": None,
                            "thinking_acc": None,
                        }
                    )
                else:
                    score.update(
                        {
                            "nothinking_response_length": None,
                            "nothinking_acc": None,
                            "thinking_response_length": valid_response_length,
                            "thinking_acc": score["acc"],
                        }
                    )

            all_scores.append(score)
            if self.is_training:
                if score["enforce_nothinking"]:
                    id2scores["nothinking"][uid].append(score)
                else:
                    id2scores["thinking"][uid].append(score)

            print_key = (
                f"source_{data_source}_{'nothinking' if is_nothinking else 'thinking'}"
            )
            if print_key not in already_print_data_sources:
                already_print_data_sources[print_key] = 0

            if already_print_data_sources[print_key] < self.num_examine:
                already_print_data_sources[print_key] += 1
                print(f"\n\n[data_source]{print_key}")
                print("[prompt]", prompt_str)
                # print("[response]", response_str)
                # print("[ground_truth]", ground_truth)
                for key, value in score.items():
                    print(f"[{key}]", value)

        if self.is_training:
            for key in ["nothinking", "thinking"]:
                for uid, scores in id2scores[key].items():
                    if scores:
                        id2mean_acc[key][uid] = np.mean(
                            [score["acc"] for score in scores]
                        )
                        lens = [
                            score["response_length"]
                            for score in scores
                            if score["acc"] == 1
                        ]
                        if len(lens) == 0:
                            id2mean_len[key][uid] = 0
                            id2std_len[key][uid] = 1
                        else:
                            id2mean_len[key][uid] = np.mean(lens)
                            id2std_len[key][uid] = np.std(lens) + 1e-7
            # print(f"id2mean_acc_{key}: {id2mean_acc}")
            # print(f"id2mean_len_{key}: {id2mean_len}")

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            score = all_scores[i]
            if self.is_training:
                uid = data_item.non_tensor_batch["uid"]
                enforce_nothinking, acc, response_len = (
                    score["enforce_nothinking"],
                    score["acc"],
                    score["response_length"],
                )
                mean_acc_thinking, mean_acc_nothinking = (
                    id2mean_acc["thinking"][uid],
                    id2mean_acc["nothinking"][uid],
                )
                mean_len_thinking, mean_len_nothinking = (
                    id2mean_len["thinking"][uid],
                    id2mean_len["nothinking"][uid],
                )
                std_len_thinking, std_len_nothinking = (
                    id2std_len["thinking"][uid],
                    id2std_len["nothinking"][uid],
                )

                ref_metrics = uid2ref_metrics[uid]
                ref_mean_acc = ref_metrics["avg_acc_thinking"]
                ref_mean_len = ref_metrics["avg_len_thinking"]

                base = acc - ref_mean_acc
                reward = base
                if acc:
                    if enforce_nothinking:
                        reward += self.nothinking_bonus
                    else:
                        easiness = ref_mean_acc
                        efficiency = mean_len_thinking / (ref_mean_len + self.eps)
                        reward += math.exp(-efficiency * easiness) * self.length_bonus

                score.update({"score": base, "reward": reward})
                if enforce_nothinking:
                    score.update(
                        {
                            "nothinking_reward": reward,
                            "thinking_reward": None,
                        }
                    )
                else:
                    score.update(
                        {
                            "nothinking_reward": None,
                            "thinking_reward": reward,
                        }
                    )
            else:
                reward = score["score"]
            # Store the information including original reward
            for key, value in score.items():
                reward_extra_info[key].append(value)

            reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
