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
from collections import defaultdict

import torch
from tqdm import tqdm

from verl import DataProto

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

    def __call__(self, data: DataProto, return_dict=False):
        """Fully vectorized reward computation"""

        if "rm_scores" in data.batch.keys():
            return (
                {"reward_tensor": data.batch["rm_scores"]}
                if return_dict
                else data.batch["rm_scores"]
            )

        device = data.batch["responses"].device
        N = len(data)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # =====================
        # BATCH EXTRACTION (vectorized where possible)
        # =====================
        valid_prompt_lengths = torch.zeros(N, dtype=torch.long, device=device)
        valid_response_lengths = torch.zeros(N, dtype=torch.long, device=device)
        enforce_mask = torch.zeros(N, dtype=torch.bool, device=device)

        prompt_ids_list = []
        response_ids_list = []
        ground_truths = []
        data_sources = []
        extra_infos = []
        uids = []

        for i in range(N):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attention_mask = data_item.batch["attention_mask"]
            prompt_length = prompt_ids.shape[-1]

            # Vectorized length computation
            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_response_length = attention_mask[prompt_length:].sum()

            valid_prompt_lengths[i] = valid_prompt_length
            valid_response_lengths[i] = valid_response_length
            enforce_mask[i] = data_item.batch["enforce_nothinking"]

            prompt_ids_list.append(prompt_ids[-valid_prompt_length:])
            response_ids_list.append(response_ids[:valid_response_length])

            # Collect metadata
            ground_truths.append(
                data_item.non_tensor_batch["reward_model"]["ground_truth"]
            )
            data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])
            extra_infos.append(data_item.non_tensor_batch.get("extra_info", None))
            uids.append(
                data_item.non_tensor_batch["uid"] if self.is_training else "validate"
            )

        # Batch decode all at once
        prompt_strs = self.tokenizer.batch_decode(
            prompt_ids_list, skip_special_tokens=True
        )
        response_strs = self.tokenizer.batch_decode(
            response_ids_list, skip_special_tokens=True
        )

        # Check if responses start with </think> (vectorized string op)
        is_nothinking_list = [r.strip().startswith("</think>") for r in response_strs]

        # =====================
        # COMPUTE SCORES (parallelizable if compute_score supports it)
        # =====================
        all_scores = []
        id2scores = {"nothinking": defaultdict(list), "thinking": defaultdict(list)}
        uid2ref_metrics = {}
        already_print_data_sources = {}

        # Batch score computation (if possible)
        for i in range(N):
            score = self.compute_score(
                data_source=data_sources[i],
                solution_str=response_strs[i],
                ground_truth=ground_truths[i],
                extra_info=extra_infos[i],
            )

            is_nothinking = is_nothinking_list[i]
            enforce_nothinking = enforce_mask[i].item()

            # Training-specific bookkeeping
            if self.is_training:
                uid = uids[i]

                if enforce_nothinking:
                    score.update(
                        {
                            "response_length": valid_response_lengths[i].item(),
                            "ground_truth": str(ground_truths[i]),
                            "enforce_nothinking": enforce_nothinking,
                            "is_nothinking": is_nothinking,
                            "nothinking_response_length": valid_response_lengths[
                                i
                            ].item(),
                            "nothinking_acc": score["acc"],
                            "thinking_response_length": None,
                            "thinking_acc": None,
                        }
                    )
                    id2scores["nothinking"][uid].append(score)
                else:
                    score.update(
                        {
                            "response_length": valid_response_lengths[i].item(),
                            "ground_truth": str(ground_truths[i]),
                            "enforce_nothinking": enforce_nothinking,
                            "is_nothinking": is_nothinking,
                            "nothinking_response_length": None,
                            "nothinking_acc": None,
                            "thinking_response_length": valid_response_lengths[
                                i
                            ].item(),
                            "thinking_acc": score["acc"],
                        }
                    )
                    id2scores["thinking"][uid].append(score)

                # Extract problem (can be cached if repeated)
                problem = (
                    prompt_strs[i]
                    .split("<｜User｜>")[1]
                    .split("<｜Assistant｜>")[0]
                    .strip()
                )
                if problem not in self.problem2ref_metrics:
                    raise KeyError(f"Problem not found in reference metrics: {problem}")
                uid2ref_metrics[uid] = self.problem2ref_metrics[problem]
            else:
                # Validation-specific bookkeeping
                if is_nothinking:
                    score.update(
                        {
                            "response_length": valid_response_lengths[i].item(),
                            "ground_truth": str(ground_truths[i]),
                            "enforce_nothinking": enforce_nothinking,
                            "is_nothinking": is_nothinking,
                            "nothinking_response_length": valid_response_lengths[
                                i
                            ].item(),
                            "nothinking_acc": score["acc"],
                            "thinking_response_length": None,
                            "thinking_acc": None,
                        }
                    )
                else:
                    score.update(
                        {
                            "response_length": valid_response_lengths[i].item(),
                            "ground_truth": str(ground_truths[i]),
                            "enforce_nothinking": enforce_nothinking,
                            "is_nothinking": is_nothinking,
                            "nothinking_response_length": None,
                            "nothinking_acc": None,
                            "thinking_response_length": valid_response_lengths[
                                i
                            ].item(),
                            "thinking_acc": score["acc"],
                        }
                    )

            all_scores.append(score)

            # Debug printing (rate-limited)
            print_key = f"source_{data_sources[i]}_{'nothinking' if (enforce_nothinking if self.is_training else is_nothinking) else 'thinking'}"
            if already_print_data_sources.get(print_key, 0) < self.num_examine:
                already_print_data_sources[print_key] = (
                    already_print_data_sources.get(print_key, 0) + 1
                )
                self._print_debug_info(print_key, prompt_strs[i], score)

        # =====================
        # UID-AGGREGATE STATS (fully vectorized)
        # =====================
        id2mean_acc = defaultdict(dict)
        id2mean_len = defaultdict(dict)
        id2std_len = defaultdict(dict)

        if self.is_training:
            for mode in ["nothinking", "thinking"]:
                for uid, scores in id2scores[mode].items():
                    if not scores:
                        continue

                    # Vectorized stats computation
                    accs = torch.tensor(
                        [s["acc"] for s in scores], dtype=torch.float32, device=device
                    )
                    lengths = torch.tensor(
                        [s["response_length"] for s in scores],
                        dtype=torch.float32,
                        device=device,
                    )

                    id2mean_acc[mode][uid] = accs.mean().item()

                    # Filter for correct answers only
                    correct_mask = accs == 1
                    correct_lengths = lengths[correct_mask]

                    if correct_lengths.numel() == 0:
                        id2mean_len[mode][uid] = 0.0
                        id2std_len[mode][uid] = 1.0
                    else:
                        id2mean_len[mode][uid] = correct_lengths.mean().item()
                        id2std_len[mode][uid] = (correct_lengths.std() + 1e-7).item()

        # =====================
        # VECTORIZED REWARD COMPUTATION
        # =====================
        acc = torch.tensor(
            [s["acc"] for s in all_scores], dtype=torch.float32, device=device
        )

        if self.is_training:
            # Build per-sample tensors
            mean_len_thinking = torch.tensor(
                [id2mean_len["thinking"][uid] for uid in uids],
                dtype=torch.float32,
                device=device,
            )
            ref_mean_acc = torch.tensor(
                [uid2ref_metrics[uid]["avg_acc_thinking"] for uid in uids],
                dtype=torch.float32,
                device=device,
            )
            ref_mean_len = torch.tensor(
                [uid2ref_metrics[uid]["avg_len_thinking"] for uid in uids],
                dtype=torch.float32,
                device=device,
            )

            # Compute easiness and budget
            easiness = ref_mean_acc * 0.9 + self.eps
            budget = ref_mean_len + self.eps

            # Base reward
            reward = acc - easiness

            # Add bonuses (vectorized)
            acc_mask = acc.bool()

            # Nothinking bonus: apply when correct AND enforce_nothinking is True
            nothinking_bonus_mask = acc_mask & enforce_mask
            reward = reward + nothinking_bonus_mask.float() * self.nothinking_bonus

            # Thinking bonus: apply when correct AND enforce_nothinking is False
            thinking_bonus_mask = acc_mask & ~enforce_mask
            efficiency = mean_len_thinking / budget
            thinking_bonus = self.length_bonus * torch.exp(-efficiency * easiness)
            reward = reward + thinking_bonus_mask.float() * thinking_bonus

            # Create separate reward views for logging
            nothinking_reward = torch.where(
                enforce_mask, reward, torch.tensor(float("nan"), device=device)
            )
            thinking_reward = torch.where(
                ~enforce_mask, reward, torch.tensor(float("nan"), device=device)
            )
        else:
            # Validation mode: reward is just the score
            # reward = torch.tensor(
            #     [s["score"] for s in all_scores], dtype=torch.float32, device=device
            # )
            reward = acc
            nothinking_reward = torch.full((N,), float("nan"), device=device)
            thinking_reward = torch.full((N,), float("nan"), device=device)

        # =====================
        # WRITE BACK (fully vectorized)
        # =====================
        idx = torch.arange(N, device=device)
        reward_tensor[idx, valid_response_lengths - 1] = reward

        # Update scores and collect extra info
        for i in range(N):
            s = all_scores[i]
            s["nothinking_reward"] = (
                None
                if torch.isnan(nothinking_reward[i])
                else nothinking_reward[i].item()
            )
            s["thinking_reward"] = (
                None if torch.isnan(thinking_reward[i]) else thinking_reward[i].item()
            )

            for k, v in s.items():
                reward_extra_info[k].append(v)

        return (
            {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
            if return_dict
            else reward_tensor
        )

    def _print_debug_info(self, print_key: str, prompt: str, score: dict) -> None:
        """Helper method for debug printing"""
        print(f"\n\n[data_source]{print_key}")
        print("[prompt]", prompt)
        for key, value in score.items():
            print(f"[{key}]", value)


import torch
import json
import re
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class OptimizedAdaptThinkRewardManager:
    """Optimized reward manager with vectorized operations and parallel processing."""

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
        num_workers=8,  # NEW: 병렬 처리 워커 수
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or adapt_think_rm
        self.reward_fn_key = reward_fn_key
        self.is_training = is_training
        self.eps = eps
        self.max_response_length = max_response_length
        self.nothinking_bonus = nothinking_bonus
        self.length_bonus = length_bonus
        self.num_workers = num_workers
        self.problem2ref_metrics = {}

        # NEW: 캐싱 및 로깅
        self._problem_cache = {}
        self.logger = logging.getLogger(__name__)
        self.debug_enabled = num_examine > 0

        # NEW: 정규표현식 컴파일
        self.problem_pattern = re.compile(r"<｜User｜>(.*?)<｜Assistant｜>", re.DOTALL)

        if self.is_training:
            if ref_result_file is None:
                raise ValueError("ref_result_file is required when is_training=True")
            with open(ref_result_file, "r") as f:
                ref_data = json.load(f)

            self.problem2ref_metrics = {
                js["problem"].strip(): js["metrics"]
                for js in tqdm(ref_data, desc="LOADING REF METRICS")
            }

            self.logger.info(f"\n\nTRAINING MODE:")
            self.logger.info(
                f"  - Loaded {len(self.problem2ref_metrics)} reference problems"
            )
            self.logger.info(f"  - NOTHINKING BONUS: {self.nothinking_bonus}")
            self.logger.info(f"  - LENGTH BONUS: {self.length_bonus}")
            self.logger.info(f"  - NUM WORKERS: {self.num_workers}\n\n")

    def _extract_problem(self, prompt_str: str) -> str:
        """Extract problem with caching."""
        if prompt_str not in self._problem_cache:
            match = self.problem_pattern.search(prompt_str)
            self._problem_cache[prompt_str] = match.group(1).strip() if match else ""
        return self._problem_cache[prompt_str]

    def _batch_extract_sequences(self, data, N: int, device) -> Tuple:
        """Vectorized batch extraction of sequences and metadata."""
        # Pre-allocate tensors
        valid_prompt_lengths = torch.zeros(N, dtype=torch.long, device=device)
        valid_response_lengths = torch.zeros(N, dtype=torch.long, device=device)
        enforce_mask = torch.zeros(N, dtype=torch.bool, device=device)

        # Lists for variable-length data
        prompt_ids_list = []
        response_ids_list = []
        ground_truths = []
        data_sources = []
        extra_infos = []
        uids = []

        # Single pass extraction
        for i in range(N):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attention_mask = data_item.batch["attention_mask"]
            prompt_length = prompt_ids.shape[-1]

            # Vectorized length computation
            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_response_length = attention_mask[prompt_length:].sum()

            valid_prompt_lengths[i] = valid_prompt_length
            valid_response_lengths[i] = valid_response_length
            enforce_mask[i] = data_item.batch["enforce_nothinking"]

            prompt_ids_list.append(prompt_ids[-valid_prompt_length:])
            response_ids_list.append(response_ids[:valid_response_length])

            # Metadata
            ground_truths.append(
                data_item.non_tensor_batch["reward_model"]["ground_truth"]
            )
            data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])
            extra_infos.append(data_item.non_tensor_batch.get("extra_info", None))
            uids.append(
                data_item.non_tensor_batch["uid"] if self.is_training else "validate"
            )

        return (
            valid_prompt_lengths,
            valid_response_lengths,
            enforce_mask,
            prompt_ids_list,
            response_ids_list,
            ground_truths,
            data_sources,
            extra_infos,
            uids,
        )

    def _compute_scores_parallel(
        self,
        data_sources: List,
        response_strs: List[str],
        ground_truths: List,
        extra_infos: List,
    ) -> List[Dict]:
        """Parallel score computation using thread pool."""
        if self.num_workers <= 1:
            # Sequential fallback
            return [
                self.compute_score(
                    data_source=ds, solution_str=rs, ground_truth=gt, extra_info=ei
                )
                for ds, rs, gt, ei in zip(
                    data_sources, response_strs, ground_truths, extra_infos
                )
            ]

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self.compute_score,
                    data_source=ds,
                    solution_str=rs,
                    ground_truth=gt,
                    extra_info=ei,
                )
                for ds, rs, gt, ei in zip(
                    data_sources, response_strs, ground_truths, extra_infos
                )
            ]
            return [f.result() for f in futures]

    def _vectorized_uid_stats(self, id2scores: Dict, device) -> Tuple[Dict, Dict, Dict]:
        """Fully vectorized UID-level statistics computation."""
        id2mean_acc = defaultdict(dict)
        id2mean_len = defaultdict(dict)
        id2std_len = defaultdict(dict)

        for mode in ["nothinking", "thinking"]:
            for uid, scores in id2scores[mode].items():
                if not scores:
                    continue

                # Vectorized stats
                accs = torch.tensor(
                    [s["acc"] for s in scores], dtype=torch.float32, device=device
                )
                lengths = torch.tensor(
                    [s["response_length"] for s in scores],
                    dtype=torch.float32,
                    device=device,
                )

                id2mean_acc[mode][uid] = accs.mean().item()

                # Filter for correct answers only
                correct_mask = accs == 1
                correct_lengths = lengths[correct_mask]

                if correct_lengths.numel() == 0:
                    id2mean_len[mode][uid] = 0.0
                    id2std_len[mode][uid] = 1.0
                else:
                    id2mean_len[mode][uid] = correct_lengths.mean().item()
                    id2std_len[mode][uid] = (correct_lengths.std() + 1e-7).item()

        return id2mean_acc, id2mean_len, id2std_len

    def _compute_training_rewards(
        self,
        acc: torch.Tensor,
        enforce_mask: torch.Tensor,
        valid_response_lengths: torch.Tensor,
        uids: List[str],
        id2mean_len: Dict,
        uid2ref_metrics: Dict,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized training reward computation."""
        N = len(uids)

        # Build per-sample tensors
        mean_len_thinking = torch.tensor(
            [id2mean_len["thinking"].get(uid, 0.0) for uid in uids],
            dtype=torch.float32,
            device=device,
        )
        ref_mean_acc = torch.tensor(
            [uid2ref_metrics[uid]["avg_acc_thinking"] for uid in uids],
            dtype=torch.float32,
            device=device,
        )
        ref_mean_len = torch.tensor(
            [uid2ref_metrics[uid]["avg_len_thinking"] for uid in uids],
            dtype=torch.float32,
            device=device,
        )

        # Compute easiness and budget
        easiness = ref_mean_acc * 0.9 + self.eps
        budget = ref_mean_len + self.eps

        # Base reward
        reward = acc - easiness

        # Vectorized bonus computation
        acc_mask = acc.bool()

        # Nothinking bonus
        nothinking_bonus_mask = acc_mask & enforce_mask
        if nothinking_bonus_mask.any():
            reward = reward + nothinking_bonus_mask.float() * self.nothinking_bonus

        # Thinking bonus (conditional computation)
        thinking_bonus_mask = acc_mask & ~enforce_mask
        if thinking_bonus_mask.any():
            efficiency = mean_len_thinking / budget
            thinking_bonus = torch.zeros_like(reward)

            # Only compute for applicable samples
            indices = thinking_bonus_mask.nonzero(as_tuple=True)[0]
            thinking_bonus[indices] = self.length_bonus * torch.exp(
                -efficiency[indices] * easiness[indices]
            )
            reward = reward + thinking_bonus

        # Separate views for logging
        nothinking_reward = torch.where(
            enforce_mask, reward, torch.tensor(float("nan"), device=device)
        )
        thinking_reward = torch.where(
            ~enforce_mask, reward, torch.tensor(float("nan"), device=device)
        )

        return reward, nothinking_reward, thinking_reward

    def __call__(self, data, return_dict=False):
        """Optimized fully vectorized reward computation."""
        try:
            # Early return for pre-computed scores
            if "rm_scores" in data.batch.keys():
                return (
                    {"reward_tensor": data.batch["rm_scores"]}
                    if return_dict
                    else data.batch["rm_scores"]
                )

            device = data.batch["responses"].device
            N = len(data)

            reward_tensor = torch.zeros_like(
                data.batch["responses"], dtype=torch.float32
            )
            reward_extra_info = defaultdict(list)

            # =====================
            # BATCH EXTRACTION (optimized)
            # =====================
            (
                valid_prompt_lengths,
                valid_response_lengths,
                enforce_mask,
                prompt_ids_list,
                response_ids_list,
                ground_truths,
                data_sources,
                extra_infos,
                uids,
            ) = self._batch_extract_sequences(data, N, device)

            # Batch decode
            prompt_strs = self.tokenizer.batch_decode(
                prompt_ids_list, skip_special_tokens=True
            )
            response_strs = self.tokenizer.batch_decode(
                response_ids_list, skip_special_tokens=True
            )

            # Vectorized nothinking check
            is_nothinking_list = [
                r.strip().startswith("</think>") for r in response_strs
            ]

            # =====================
            # PARALLEL SCORE COMPUTATION
            # =====================
            all_scores = self._compute_scores_parallel(
                data_sources, response_strs, ground_truths, extra_infos
            )

            # =====================
            # SCORE ENRICHMENT & BOOKKEEPING
            # =====================
            id2scores = {"nothinking": defaultdict(list), "thinking": defaultdict(list)}
            uid2ref_metrics = {}
            already_print_data_sources = {}

            for i in range(N):
                score = all_scores[i]
                is_nothinking = is_nothinking_list[i]
                enforce_nothinking = enforce_mask[i].item()

                # Common score updates
                base_update = {
                    "response_length": valid_response_lengths[i].item(),
                    "ground_truth": str(ground_truths[i]),
                    "enforce_nothinking": enforce_nothinking,
                    "is_nothinking": is_nothinking,
                }

                if self.is_training:
                    uid = uids[i]

                    if enforce_nothinking:
                        score.update(
                            {
                                **base_update,
                                "nothinking_response_length": valid_response_lengths[
                                    i
                                ].item(),
                                "nothinking_acc": score["acc"],
                                "thinking_response_length": None,
                                "thinking_acc": None,
                            }
                        )
                        id2scores["nothinking"][uid].append(score)
                    else:
                        score.update(
                            {
                                **base_update,
                                "nothinking_response_length": None,
                                "nothinking_acc": None,
                                "thinking_response_length": valid_response_lengths[
                                    i
                                ].item(),
                                "thinking_acc": score["acc"],
                            }
                        )
                        id2scores["thinking"][uid].append(score)

                    # Cached problem extraction
                    problem = self._extract_problem(prompt_strs[i])
                    if problem not in self.problem2ref_metrics:
                        raise KeyError(
                            f"Problem not found in reference metrics: {problem}"
                        )
                    uid2ref_metrics[uid] = self.problem2ref_metrics[problem]
                else:
                    # Validation mode
                    if is_nothinking:
                        score.update(
                            {
                                **base_update,
                                "nothinking_response_length": valid_response_lengths[
                                    i
                                ].item(),
                                "nothinking_acc": score["acc"],
                                "thinking_response_length": None,
                                "thinking_acc": None,
                            }
                        )
                    else:
                        score.update(
                            {
                                **base_update,
                                "nothinking_response_length": None,
                                "nothinking_acc": None,
                                "thinking_response_length": valid_response_lengths[
                                    i
                                ].item(),
                                "thinking_acc": score["acc"],
                            }
                        )

                # Optimized debug printing
                if self.debug_enabled:
                    print_key = f"source_{data_sources[i]}_{'nothinking' if (enforce_nothinking if self.is_training else is_nothinking) else 'thinking'}"
                    if already_print_data_sources.get(print_key, 0) < self.num_examine:
                        already_print_data_sources[print_key] = (
                            already_print_data_sources.get(print_key, 0) + 1
                        )
                        self._print_debug_info(print_key, prompt_strs[i], score)

            # =====================
            # VECTORIZED UID STATS
            # =====================
            id2mean_acc, id2mean_len, id2std_len = {}, {}, {}
            if self.is_training:
                id2mean_acc, id2mean_len, id2std_len = self._vectorized_uid_stats(
                    id2scores, device
                )

            # =====================
            # VECTORIZED REWARD COMPUTATION
            # =====================
            acc = torch.tensor(
                [s["acc"] for s in all_scores], dtype=torch.float32, device=device
            )

            if self.is_training:
                reward, nothinking_reward, thinking_reward = (
                    self._compute_training_rewards(
                        acc,
                        enforce_mask,
                        valid_response_lengths,
                        uids,
                        id2mean_len,
                        uid2ref_metrics,
                        device,
                    )
                )
            else:
                reward = acc
                nothinking_reward = torch.full((N,), float("nan"), device=device)
                thinking_reward = torch.full((N,), float("nan"), device=device)

            # =====================
            # WRITE BACK
            # =====================
            idx = torch.arange(N, device=device)
            reward_tensor[idx, valid_response_lengths - 1] = reward

            # Update scores
            for i in range(N):
                s = all_scores[i]
                s["nothinking_reward"] = (
                    None
                    if torch.isnan(nothinking_reward[i])
                    else nothinking_reward[i].item()
                )
                s["thinking_reward"] = (
                    None
                    if torch.isnan(thinking_reward[i])
                    else thinking_reward[i].item()
                )

                for k, v in s.items():
                    reward_extra_info[k].append(v)

            return (
                {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
                if return_dict
                else reward_tensor
            )

        finally:
            # Explicit memory cleanup for large tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def _print_debug_info(self, print_key: str, prompt: str, score: dict) -> None:
        """Helper method for debug printing."""
        self.logger.debug(f"\n\n[data_source]{print_key}")
        self.logger.debug(f"[prompt]{prompt}")
        for key, value in score.items():
            self.logger.debug(f"[{key}]{value}")
