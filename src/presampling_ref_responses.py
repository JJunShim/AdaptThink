import json
import logging
import math
import os
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
from omegaconf import OmegaConf
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port

# 사용자 설정 (가정)
from src.config.presampling_ref import Config

# --- Type Hints ---
RankData = List[Dict[str, Any]]
PromptData = Tuple[List[str], List[Dict[str, Any]]]


def setup_logger(name: str, rank: Optional[int] = None) -> logging.Logger:
    """로거 설정 (공통 유틸)"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    rank_prefix = f"[Rank {rank}] " if rank is not None else "[Master] "
    formatter = logging.Formatter(
        f"%(asctime)s - {rank_prefix}%(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class PathManager:
    """경로 및 파일명 관리"""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output.output_dir)
        self.dataset_name = Path(config.data.dataset_path).stem
        self.model_short = config.model.name.split("/")[-1]

    @property
    def final_output_path(self) -> Path:
        suffix = "_nothinking" if self.config.prompt.nothinking else ""
        filename = (
            f"{self.model_short}_{self.dataset_name}_"
            f"n{self.config.data.num_samples}_"
            f"K{self.config.data.K}_"
            f"len{self.config.model.max_tokens}{suffix}.jsonl"
        )
        return self.output_dir / filename

    def get_rank_output_path(self, rank: int) -> Path:
        final = self.final_output_path
        return final.parent / f"{final.stem}_rank{rank}{final.suffix}"

    def ensure_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


class DataManager:
    """
    데이터 로딩, 전처리, 프롬프트 변환을 담당하는 클래스
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("DataManager")

    def load_and_shard(self) -> List[Tuple[RankData, int]]:
        """
        데이터를 로드하고 DP 사이즈에 맞춰 분할(Shard)하여 반환
        Return: [(chunk_data, start_index), ...]
        """
        self.logger.info(f"Loading dataset: {self.config.data.dataset_path}")

        with open(self.config.data.dataset_path, "r") as f:
            data = json.load(f)

        if self.config.data.num_samples > 0:
            data = data[: self.config.data.num_samples]

        total_items = len(data)
        dp_size = self.config.parallel.dp_size
        chunk_size = math.ceil(total_items / dp_size)

        shards = []
        for i in range(dp_size):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_items)

            if start_idx >= total_items:
                chunk = []
            else:
                chunk = data[start_idx:end_idx]

            shards.append((chunk, start_idx))

        self.logger.info(f"Loaded {total_items} items, split into {dp_size} shards.")
        return shards

    @staticmethod
    def chunk_to_prompts(chunk: RankData, start_idx: int, config: Config) -> PromptData:
        """
        Raw 데이터 청크를 vLLM 입력용 프롬프트와 메타데이터로 변환
        (Worker 프로세스에서 호출됨 -> staticmethod 권장)
        """
        prompts = []
        metadata = []

        # 템플릿 선택
        template = (
            config.prompt.nothink if config.prompt.nothinking else config.prompt.think
        )
        K = config.data.K

        for i, item in enumerate(chunk):
            real_idx = start_idx + i
            prompt_text = template.format(question=item["problem"])

            # K번 샘플링을 위해 복제
            for j in range(K):
                prompts.append(prompt_text)
                metadata.append(
                    {
                        "_id": f"{real_idx}_{j}",
                        "problem_idx": real_idx,
                        "sample_idx": j,
                        "original": item,
                    }
                )

        return prompts, metadata


def worker_logic(
    rank: int,
    local_rank: int,
    shard_data: Tuple[RankData, int],  # (data, start_idx)
    config: Config,
    master_addr: str,
    master_port: int,
):
    """
    개별 GPU 워커 프로세스 로직
    """
    logger = setup_logger(f"Worker-{rank}", rank=rank)
    paths = PathManager(config)

    chunk, start_idx = shard_data

    # 1. 환경변수 설정
    os.environ.update(
        {
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(local_rank),
            "VLLM_DP_SIZE": str(config.parallel.dp_size),
            "VLLM_DP_MASTER_IP": master_addr,
            "VLLM_DP_MASTER_PORT": str(master_port),
        }
    )

    try:
        if not chunk:
            logger.warning("Empty chunk received. Exiting.")
            paths.get_rank_output_path(rank).touch()  # 빈 파일 생성
            return

        # 2. 프롬프트 생성 (DataManager의 정적 메서드 사용)
        prompts, metadata = DataManager.chunk_to_prompts(chunk, start_idx, config)
        logger.info(f"Prepared {len(prompts)} prompts.")

        # 3. LLM 로드
        llm = LLM(
            model=config.model.name,
            tensor_parallel_size=config.parallel.tp_size,
            gpu_memory_utilization=config.model.gpu_memory_utilization,
            max_model_len=config.model.max_tokens,
            trust_remote_code=config.model.trust_remote_code,
            seed=config.sampling.seed_offset + rank,
        )

        sampling_params = SamplingParams(
            temperature=config.sampling.temperature,
            top_p=config.sampling.top_p,
            max_tokens=config.model.max_tokens,
        )

        # 4. 추론 실행
        logger.info("Running inference...")
        outputs = llm.generate(prompts, sampling_params)

        # 5. 결과 정리
        results = []
        for meta, output in zip(metadata, outputs):
            o = output.outputs[0]
            res = meta["original"].copy()
            res.update(
                {
                    "_id": meta["_id"],
                    "problem_idx": meta["problem_idx"],
                    "sample_idx": meta["sample_idx"],
                    "dp_rank": rank,
                    "response": o.text,
                    "tokens": len(o.token_ids)
                }
            )
            results.append(res)

        # 6. 저장
        out_path = paths.get_rank_output_path(rank)
        with jsonlines.open(out_path, "w") as writer:
            writer.write_all(results)

        logger.info(f"Saved {len(results)} items to {out_path}")

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise e


class InferenceRunner:
    """
    전체 프로세스 관리 및 실행 (Orchestrator)
    """

    def __init__(self, config: Config, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.paths = PathManager(config)
        self.logger = setup_logger("Runner")

        self.paths.ensure_dir()

    def run(self):
        self._print_config()

        # 1. 데이터 준비
        shards = self.data_manager.load_and_shard()

        # 2. 통신 설정
        master_ip = self.config.parallel.master_addr
        if self.config.parallel.node_size == 1:
            master_ip = "127.0.0.1"
            master_port = get_open_port()
        else:
            master_port = self.config.parallel.master_port

        self.logger.info(f"Master Address: {master_ip}:{master_port}")

        # 3. 워커 프로세스 실행
        processes = []
        dp_per_node = self.config.parallel.dp_size // self.config.parallel.node_size
        node_rank = self.config.parallel.node_rank
        start_global_rank = node_rank * dp_per_node

        for local_rank in range(dp_per_node):
            global_rank = start_global_rank + local_rank

            p = Process(
                target=worker_logic,
                kwargs={
                    "rank": global_rank,
                    "local_rank": local_rank,
                    "shard_data": shards[global_rank],
                    "config": self.config,
                    "master_addr": master_ip,
                    "master_port": master_port,
                },
            )
            p.start()
            processes.append(p)
            self.logger.info(f"Started worker {global_rank} (PID: {p.pid})")

        # 4. 대기 및 에러 체크
        failed = False
        for p in processes:
            p.join()
            if p.exitcode != 0:
                self.logger.error(f"Worker {p.pid} failed with exit code {p.exitcode}")
                failed = True

        if failed:
            self.logger.error("Aborting due to worker failures.")
            exit(1)

        # 5. 결과 병합
        self._merge_results()

    def _merge_results(self):
        self.logger.info("Merging results from all ranks...")
        merged_data = []

        for rank in range(self.config.parallel.dp_size):
            p = self.paths.get_rank_output_path(rank)
            if p.exists():
                with jsonlines.open(p, "r") as reader:
                    merged_data.extend(list(reader))

                if not self.config.output.save_intermediate:
                    p.unlink()

        # 정렬
        merged_data.sort(key=lambda x: (x["problem_idx"], x["sample_idx"]))

        final_path = self.paths.final_output_path
        with jsonlines.open(final_path, "w") as writer:
            writer.write_all(merged_data)

        self.logger.info(
            f"Done. Final output: {final_path} ({len(merged_data)} samples)"
        )

    def _print_config(self):
        self.logger.info("=" * 40)
        self.logger.info(f"Model: {self.config.model.name}")
        self.logger.info(f"DP Size: {self.config.parallel.dp_size}")
        self.logger.info(
            f"Samples: {self.config.data.num_samples} * {self.config.data.K}"
        )
        self.logger.info("=" * 40)


def main():
    # 설정 로드
    cli_conf = OmegaConf.from_cli()
    base_conf = OmegaConf.structured(Config)

    if cli_conf.get("config"):
        file_conf = OmegaConf.load(cli_conf.config)
        config = OmegaConf.merge(base_conf, file_conf, cli_conf)
    else:
        config = OmegaConf.merge(base_conf, cli_conf)

    # 객체 주입 (Dependency Injection)
    data_manager = DataManager(config)
    runner = InferenceRunner(config, data_manager)

    runner.run()


if __name__ == "__main__":
    main()
