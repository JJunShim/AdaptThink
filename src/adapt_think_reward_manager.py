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
import re
from collections import defaultdict

import torch
from tqdm import tqdm

from .adapt_think_rm import adapt_think_rm


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
        self.problem2ref_metrics = {}

        self._problem_re = re.compile(r"<｜User｜>(.*?)<｜Assistant｜>", re.DOTALL)

        if self.is_training:
            if ref_result_file is None:
                raise ValueError("ref_result_file is required when is_training=True")
            with open(ref_result_file, "r") as f:
                ref_data = json.load(f)

            self.problem2ref_metrics = {
                js["problem"].strip(): js["metrics"]
                for js in tqdm(ref_data, desc="LOADING REF METRICS")
            }

            print(f"\n\nTRAINING MODE:")
            print(f"  - Loaded {len(self.problem2ref_metrics)} reference problems")
            print(f"  - NOTHINKING BONUS: {self.nothinking_bonus}")
            print(f"  - LENGTH BONUS: {self.length_bonus}\n\n")

    def __call__(self, data, return_dict=False):
        if "rm_scores" in data.batch.keys():
            scores = data.batch["rm_scores"]
            return {"reward_tensor": scores} if return_dict else scores

        device = data.batch["responses"].device
        N = len(data)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # ── 1. 배치 추출 ─────────────────────────────────────────────────
        meta = self._extract_batch_metadata(data, N, device)

        # ── 2. 일괄 디코딩 ───────────────────────────────────────────────
        prompt_strs = self.tokenizer.batch_decode(
            meta["prompt_ids_list"], skip_special_tokens=True
        )
        response_strs = self.tokenizer.batch_decode(
            meta["response_ids_list"], skip_special_tokens=True
        )
        is_nothinking_list = [r.strip().startswith("</think>") for r in response_strs]

        # ── 3. 점수 계산 ─────────────────────────────────────────────────
        all_scores, id2scores, uid2ref_metrics = self._compute_scores(
            N, meta, prompt_strs, response_strs, is_nothinking_list
        )

        # ── 4. UID 집계 통계 ─────────────────────────────────────────────
        id2mean_len = self._compute_uid_stats(id2scores)

        # ── 5. 보상 계산 ─────────────────────────────────────────────────
        reward, enforce_mask = self._compute_reward(
            N, all_scores, meta, uid2ref_metrics, id2mean_len, device
        )

        # ── 6. Scatter ───────────────────────────────────────────────────
        idx = torch.arange(N, device=device)
        reward_tensor[idx, meta["valid_response_lengths"] - 1] = reward

        # ── 7. Extra info ────────────────────────────────────────────────
        reward_extra_info = self._build_extra_info(all_scores, reward, enforce_mask)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor

    @staticmethod
    def _resolve_mode(enforce_nothinking: bool, is_nothinking: bool) -> str:
        return "nothinking" if (enforce_nothinking or is_nothinking) else "thinking"

    @staticmethod
    def _build_score_fields(
        score, resp_len, ground_truth, enforce_nothinking, is_nothinking, mode
    ):
        is_nt = mode == "nothinking"
        return {
            "response_length": resp_len,
            "ground_truth": str(ground_truth),
            "enforce_nothinking": enforce_nothinking,
            "is_nothinking": is_nothinking,
            "nothinking_response_length": resp_len if is_nt else None,
            "nothinking_acc": score["acc"] if is_nt else None,
            "thinking_response_length": None if is_nt else resp_len,
            "thinking_acc": None if is_nt else score["acc"],
        }

    def _print_debug_info(self, print_key: str, prompt: str, score: dict) -> None:
        """Print structured debug information with sorted keys."""
        header = [
            "",
            "",
            f"[data_source] {print_key}",
            f"[prompt] {prompt}",
        ]
        body = [f"[{k}] {score[k]}" for k in sorted(score)]

        print("\n".join(header + body))

    def _parse_problem(self, prompt_str: str) -> str:
        m = self._problem_re.search(prompt_str)
        return m.group(1).strip() if m else prompt_str

    def _extract_batch_metadata(self, data, N, device):
        valid_prompt_lengths = torch.zeros(N, dtype=torch.long, device=device)
        valid_response_lengths = torch.zeros(N, dtype=torch.long, device=device)
        enforce_mask = torch.zeros(N, dtype=torch.bool, device=device)

        prompt_ids_list = []
        response_ids_list = []
        ground_truths = []
        data_sources = []
        extra_infos = []
        uids = []

        for i, item in enumerate(data):
            batch = item.batch
            non_tensor = item.non_tensor_batch

            prompt_ids = batch["prompts"]
            response_ids = batch["responses"]
            attention_mask = batch["attention_mask"]
            prompt_len = prompt_ids.shape[-1]

            valid_prompt_len = attention_mask[:prompt_len].sum()
            valid_resp_len = attention_mask[prompt_len:].sum()

            valid_prompt_lengths[i] = valid_prompt_len
            valid_response_lengths[i] = valid_resp_len
            enforce_mask[i] = batch["enforce_nothinking"]

            prompt_ids_list.append(prompt_ids[-valid_prompt_len:])
            response_ids_list.append(response_ids[:valid_resp_len])

            ground_truths.append(non_tensor["reward_model"]["ground_truth"])
            data_sources.append(non_tensor[self.reward_fn_key])
            extra_infos.append(non_tensor.get("extra_info"))

            uids.append(non_tensor["uid"] if self.is_training else "validate")

        return {
            "valid_prompt_lengths": valid_prompt_lengths,
            "valid_response_lengths": valid_response_lengths,
            "enforce_mask": enforce_mask,
            "prompt_ids_list": prompt_ids_list,
            "response_ids_list": response_ids_list,
            "ground_truths": ground_truths,
            "data_sources": data_sources,
            "extra_infos": extra_infos,
            "uids": uids,
        }

    def _compute_scores(self, N, meta, prompt_strs, response_strs, is_nothinking_list):
        resp_len_cpu = meta["valid_response_lengths"].cpu().tolist()
        enforce_cpu = meta["enforce_mask"].cpu().tolist()

        data_sources = meta["data_sources"]
        ground_truths = meta["ground_truths"]
        extra_infos = meta["extra_infos"]
        uids = meta["uids"]

        all_scores = []
        id2scores = {"nothinking": defaultdict(list), "thinking": defaultdict(list)}
        uid2ref_metrics = {}
        already_print_data_sources = {}

        for i, (
            data_source,
            solution_str,
            ground_truth,
            extra_info,
            prompt_str,
            is_nothinking,
            resp_len,
            enforce_nothinking,
        ) in enumerate(
            zip(
                data_sources,
                response_strs,
                ground_truths,
                extra_infos,
                prompt_strs,
                is_nothinking_list,
                resp_len_cpu,
                enforce_cpu,
            )
        ):

            score = self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            mode = self._resolve_mode(enforce_nothinking, is_nothinking)

            score.update(
                self._build_score_fields(
                    score=score,
                    resp_len=resp_len,
                    ground_truth=ground_truth,
                    enforce_nothinking=enforce_nothinking,
                    is_nothinking=is_nothinking,
                    mode=mode,
                )
            )

            if self.is_training:
                uid = uids[i]
                id2scores[mode][uid].append(score)

                problem = self._parse_problem(prompt_str)
                if problem not in self.problem2ref_metrics:
                    raise KeyError(f"Problem not found in reference metrics: {problem}")

                uid2ref_metrics[uid] = self.problem2ref_metrics[problem]

            all_scores.append(score)

            # debug printing (rate-limited)
            print_key = f"source_{data_source}_{mode}"
            if already_print_data_sources.get(print_key, 0) < self.num_examine:
                already_print_data_sources[print_key] = (
                    already_print_data_sources.get(print_key, 0) + 1
                )
                self._print_debug_info(print_key, prompt_str, score)

        return all_scores, id2scores, uid2ref_metrics

    def _compute_uid_stats(self, id2scores):
        id2mean_len = defaultdict(dict)

        if not self.is_training:
            return id2mean_len

        for mode in ["nothinking", "thinking"]:
            for uid, scores in id2scores[mode].items():
                if not scores:
                    continue
                # CPU 리스트 연산 → GPU 동기화 0회
                correct_lengths = [
                    s["response_length"] for s in scores if s["acc"] == 1.0
                ]
                id2mean_len[mode][uid] = (
                    sum(correct_lengths) / len(correct_lengths)
                    if correct_lengths
                    else 0.0
                )

        return id2mean_len

    def _compute_reward(
        self, N, all_scores, meta, uid2ref_metrics, id2mean_len, device
    ):
        scale = 0.8

        uids = meta["uids"]
        enforce_mask = meta["enforce_mask"]

        # ---- base reward ----
        reward = torch.as_tensor(
            [s["acc"] for s in all_scores],
            dtype=torch.float32,
            device=device,
        )

        if not self.is_training:
            return reward, enforce_mask

        acc_mask = reward.bool()

        # ---- masks ----
        nothinking_mask = acc_mask & enforce_mask
        thinking_mask = acc_mask & (~enforce_mask)

        # ---- reference tensors ----
        ref_mean_acc = torch.as_tensor(
            [uid2ref_metrics[uid]["avg_acc_thinking"] for uid in uids],
            dtype=torch.float32,
            device=device,
        )

        ref_mean_len = torch.as_tensor(
            [uid2ref_metrics[uid]["avg_len_thinking"] for uid in uids],
            dtype=torch.float32,
            device=device,
        )

        mean_len_thinking = torch.as_tensor(
            [id2mean_len["thinking"].get(uid, 0.0) for uid in uids],
            dtype=torch.float32,
            device=device,
        )

        # ---- compute easiness ----
        easiness = ref_mean_acc.mul(scale).add_(self.eps)

        # ---- compute efficiency ----
        efficiency = mean_len_thinking / (ref_mean_len + self.eps)

        # ---- thinking bonus ----
        thinking_bonus = torch.exp(-(efficiency * easiness))
        thinking_bonus.mul_(self.length_bonus)

        # ---- reward update ----
        reward.sub_(easiness)

        reward[nothinking_mask] += self.nothinking_bonus
        reward[thinking_mask] += thinking_bonus[thinking_mask]
        # reward *= acc_mask

        return reward, enforce_mask

    def _build_extra_info(self, all_scores, reward, enforce_mask):
        reward_cpu = reward.cpu().tolist()
        enforce_cpu = enforce_mask.cpu().tolist()
        reward_extra_info = defaultdict(list)

        for i, s in enumerate(all_scores):
            r = reward_cpu[i]
            is_nt = enforce_cpu[i]

            s["nothinking_reward"] = r if is_nt else None
            s["thinking_reward"] = None if is_nt else r

            for k, v in s.items():
                reward_extra_info[k].append(v)

        return reward_extra_info
