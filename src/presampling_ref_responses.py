import json
import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Process
from pathlib import Path
from time import sleep
from typing import Dict, List

import jsonlines
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


# 로거 설정
def setup_logger(name: str, rank: int = None) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 핸들러가 이미 있으면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()

    # 포맷터 설정
    rank_str = f"[Rank {rank}] " if rank is not None else ""
    formatter = logging.Formatter(
        f'%(asctime)s - {rank_str}%(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    max_tokens: int = 16384
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.9


@dataclass
class ParallelConfig:
    """병렬 처리 설정"""
    dp_size: int = 2  # Data parallel size
    tp_size: int = 1  # Tensor parallel size
    node_size: int = 1  # Total number of nodes
    node_rank: int = 0  # Rank of current node
    master_addr: str = ""  # Master node IP
    master_port: int = 0  # Master node port
    timeout: int = 3600  # Timeout in seconds


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    dataset_path: str = "./data/train/deepscaler.json"
    num_samples: int = 10  # 0 for all data
    K: int = 16  # Samples per problem


@dataclass
class SamplingConfig:
    """샘플링 파라미터 설정"""
    temperature: float = 0.6
    top_p: float = 0.95
    seed_offset: int = 42  # Base seed, will add rank


@dataclass
class PromptConfig:
    """프롬프트 관련 설정"""
    nothinking: bool = False

    think: str = '<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n'
    nothink: str = '<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n</think>'


@dataclass
class OutputConfig:
    """출력 관련 설정"""
    output_dir: str = "./data/train/ref_presampling"
    save_intermediate: bool = True  # Save intermediate rank results


@dataclass
class Config:
    """전체 설정"""
    config: str = field()
    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class ConfigManager:
    """설정 관리 클래스"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = setup_logger("ConfigManager")

    def get_output_path(self) -> Path:
        """최종 출력 파일 경로"""
        dataset = Path(self.config.data.dataset_path).stem
        model_short = self.config.model.name.split('/')[-1]
        suffix = '_nothinking' if self.config.prompt.nothinking else ''

        filename = (
            f"{model_short}_{dataset}_"
            f"n{self.config.data.num_samples}_"
            f"K{self.config.data.K}_"
            f"len{self.config.model.max_tokens}{suffix}.jsonl"
        )

        output_dir = Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    def get_rank_output_path(self, rank: int) -> Path:
        """각 rank별 임시 출력 경로"""
        output_path = self.get_output_path()
        rank_filename = output_path.stem + f"_rank{rank}" + output_path.suffix
        return output_path.parent / rank_filename

    def get_prompt_template(self) -> str:
        """프롬프트 템플릿 반환"""

        if self.config.prompt.nothinking:
            return self.config.prompt.nothink
        else:
            return self.config.prompt.think


    def log_config(self):
        """설정 정보 로깅"""
        self.logger.info("="*60)
        self.logger.info("Configuration:")
        self.logger.info("="*60)
        self.logger.info(f"Model: {self.config.model.name}")
        self.logger.info(f"Dataset: {self.config.data.dataset_path}")
        self.logger.info(f"Num samples: {self.config.data.num_samples}")
        self.logger.info(f"K (samples per problem): {self.config.data.K}")
        self.logger.info(f"Data Parallel size: {self.config.parallel.dp_size}")
        self.logger.info(
            f"Tensor Parallel size: {self.config.parallel.tp_size}")
        self.logger.info(f"Max tokens: {self.config.model.max_tokens}")
        self.logger.info(
            f"GPU memory utilization: {self.config.model.gpu_memory_utilization}")
        self.logger.info(f"Temperature: {self.config.sampling.temperature}")
        self.logger.info(f"Top-p: {self.config.sampling.top_p}")
        self.logger.info(f"No thinking: {self.config.prompt.nothinking}")
        self.logger.info(f"Output: {self.get_output_path()}")
        self.logger.info("="*60)


def format_prompt(question: str, template: str) -> str:
    """프롬프트 포맷팅"""
    return template.format(question=question)


def prepare_prompts_and_metadata(
    data: List[Dict],
    config_manager: ConfigManager
) -> tuple:
    """프롬프트와 메타데이터 준비"""
    logger = logging.getLogger("PrepareData")

    prompts = []
    metadata = []
    template = config_manager.get_prompt_template()

    for i, js in enumerate(data):
        prompt = format_prompt(js['problem'], template)

        # 각 문제에 대해 K개의 샘플 생성
        for j in range(config_manager.config.data.K):
            prompts.append(prompt)
            metadata.append({
                '_id': f'{i}_{j}',
                'problem_idx': i,
                'sample_idx': j,
                'original': js
            })

    logger.info(f"Prepared {len(prompts)} prompts from {len(data)} problems")
    return prompts, metadata


def distribute_work(
    prompts: List,
    metadata: List,
    dp_size: int,
    global_dp_rank: int
) -> tuple:
    """각 DP rank에 작업 분배"""
    logger = logging.getLogger("DistributeWork")

    total_tasks = len(prompts)
    floor = total_tasks // dp_size
    remainder = total_tasks % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    start_idx = start(global_dp_rank)
    end_idx = start(global_dp_rank + 1)

    rank_prompts = prompts[start_idx:end_idx]
    rank_metadata = metadata[start_idx:end_idx]

    # 빈 경우 placeholder 추가
    if len(rank_prompts) == 0:
        rank_prompts = ["Placeholder"]
        rank_metadata = [{'_id': 'placeholder', 'is_placeholder': True}]
        logger.warning(f"Rank {global_dp_rank} has no work, using placeholder")
    else:
        logger.info(
            f"Rank {global_dp_rank}: assigned {len(rank_prompts)} tasks "
            f"(indices {start_idx}-{end_idx-1})"
        )

    return rank_prompts, rank_metadata


def save_results(results: List[Dict], output_path: Path, logger: logging.Logger):
    """결과 저장"""
    with jsonlines.open(output_path, 'w') as writer:
        count = 0
        for result in results:
            if not result.get('is_placeholder', False):
                writer.write(result)
                count += 1
    logger.info(f"Saved {count} results to {output_path}")


def worker_process(
    config_dict: dict,
    data: List[Dict],
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    tp_size: int,
):
    """각 DP rank에서 실행되는 워커 프로세스"""

    # 로거 설정
    logger = setup_logger("Worker", rank=global_dp_rank)

    # Config 재구성
    config = OmegaConf.create(config_dict)
    config_manager = ConfigManager(config)

    # 환경 변수 설정
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    logger.info(f"Starting worker process (Local rank: {local_dp_rank})")

    # 프롬프트와 메타데이터 준비
    all_prompts, all_metadata = prepare_prompts_and_metadata(
        data, config_manager)

    # 이 rank의 작업 분배
    prompts, metadata = distribute_work(
        all_prompts, all_metadata, dp_size, global_dp_rank
    )

    logger.info(f"Processing {len(prompts)} prompts")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        max_tokens=config.model.max_tokens,
        seed=config.sampling.seed_offset + global_dp_rank,
    )

    logger.info(
        f"Sampling params: temp={config.sampling.temperature}, "
        f"top_p={config.sampling.top_p}, "
        f"seed={config.sampling.seed_offset + global_dp_rank}"
    )

    # LLM 초기화
    logger.info("Initializing LLM engine...")
    llm = LLM(
        model=config.model.name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        max_model_len=config.model.max_tokens,
        trust_remote_code=config.model.trust_remote_code,
    )
    logger.info("LLM engine initialized successfully")

    # 추론 실행
    logger.info("Starting inference...")
    outputs = llm.generate(prompts, sampling_params)
    logger.info("Inference completed")

    # 결과 구성
    results = []
    for meta, output in zip(metadata, outputs):
        if meta.get('is_placeholder', False):
            continue

        result = meta['original'].copy()
        result['_id'] = meta['_id']
        result['problem_idx'] = meta['problem_idx']
        result['sample_idx'] = meta['sample_idx']
        result['dp_rank'] = global_dp_rank
        result['response'] = {
            'text': output.outputs[0].text,
            'tokens': len(output.outputs[0].token_ids),
            'finish_reason': output.outputs[0].finish_reason,
        }
        results.append(result)

    # 결과 저장
    rank_output_path = config_manager.get_rank_output_path(global_dp_rank)
    save_results(results, rank_output_path, logger)

    # 샘플 로깅
    logger.info(f"Sample outputs (showing first 3):")
    for i, result in enumerate(results[:3]):
        logger.info(
            f"  [{i+1}] Problem: {result.get('problem', 'N/A')[:80]}...")
        logger.info(f"      Response: {result['response']['text'][:80]}...")

    logger.info(f"Worker process completed successfully")
    sleep(1)


def merge_results(
    config_manager: ConfigManager,
    dp_size: int,
    logger: logging.Logger
):
    """모든 rank의 결과를 하나로 병합"""
    logger.info("Merging results from all ranks...")

    all_results = []
    for rank in range(dp_size):
        rank_output_path = config_manager.get_rank_output_path(rank)
        if rank_output_path.exists():
            with jsonlines.open(rank_output_path, 'r') as reader:
                rank_results = list(reader)
                all_results.extend(rank_results)
                logger.info(
                    f"Loaded {len(rank_results)} results from rank {rank}")

            # 임시 파일 삭제
            if config_manager.config.output.save_intermediate:
                logger.info(f"Keeping intermediate file: {rank_output_path}")
            else:
                rank_output_path.unlink()
                logger.info(f"Deleted intermediate file: {rank_output_path}")
        else:
            logger.warning(f"No output file found for rank {rank}")

    # 결과를 _id 순으로 정렬
    all_results.sort(key=lambda x: (x['problem_idx'], x['sample_idx']))

    # 최종 파일에 저장
    output_path = config_manager.get_output_path()
    with jsonlines.open(output_path, 'w') as writer:
        for result in all_results:
            writer.write(result)

    logger.info(f"Merged {len(all_results)} total results to {output_path}")


def load_data(config: DictConfig, logger: logging.Logger) -> List[Dict]:
    """데이터 로드 및 샘플링"""
    logger.info(f"Loading data from {config.data.dataset_path}")

    with open(config.data.dataset_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} problems")

    if config.data.num_samples > 0:
        data = data[:config.data.num_samples]
        logger.info(
            f"Using {len(data)} problems (num_samples={config.data.num_samples})")
    else:
        logger.info(f"Using all {len(data)} problems")

    total_inferences = len(data) * config.data.K
    logger.info(f"Total inferences to perform: {total_inferences}")

    return data


def main():
    # 메인 로거 설정
    logger = setup_logger("Main")

    # CLI 인자 파싱
    cli_conf = OmegaConf.from_cli()

    # YAML 설정 로드 (존재하는 경우)
    if cli_conf.get('config'):
        yaml_conf = OmegaConf.load(cli_conf.config)
        logger.info(f"Loaded config from {cli_conf.config}")
    else:
        yaml_conf = OmegaConf.create()

    # 기본 설정
    default_conf = OmegaConf.structured(Config)

    # 병합: default < yaml < cli
    config = OmegaConf.merge(default_conf, yaml_conf, cli_conf)

    # ConfigManager 생성
    config_manager = ConfigManager(config)
    config_manager.log_config()

    # 데이터 로드
    data = load_data(config, logger)

    # DP master 설정
    if config.parallel.node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = config.parallel.master_addr
        dp_master_port = config.parallel.master_port

    logger.info(f"DP Master: {dp_master_ip}:{dp_master_port}")

    # Config를 dict로 변환 (multiprocessing을 위해)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # 각 DP rank를 위한 프로세스 시작
    logger.info(f"Starting {config.parallel.dp_size} worker processes...")
    procs = []

    dp_per_node = config.parallel.dp_size // config.parallel.node_size
    start_rank = config.parallel.node_rank * dp_per_node
    end_rank = (config.parallel.node_rank + 1) * dp_per_node

    for local_dp_rank, global_dp_rank in enumerate(range(start_rank, end_rank)):
        proc = Process(
            target=worker_process,
            args=(
                config_dict,
                data,
                config.parallel.dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                config.parallel.tp_size,
            ),
        )
        proc.start()
        procs.append(proc)
        logger.info(
            f"Started process for rank {global_dp_rank} (PID: {proc.pid})")

    # 모든 프로세스 완료 대기
    logger.info("Waiting for all processes to complete...")
    exit_code = 0
    for i, proc in enumerate(procs):
        # proc.join(timeout=config.parallel.timeout)
        proc.join()
        if proc.exitcode is None:
            logger.error(
                f"Process {proc.pid} (rank {start_rank + i}) "
                f"didn't stop within {config.parallel.timeout}s timeout. Killing..."
            )
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            logger.error(
                f"Process {proc.pid} (rank {start_rank + i}) exited with code {proc.exitcode}")
            exit_code = proc.exitcode
        else:
            logger.info(
                f"Process {proc.pid} (rank {start_rank + i}) completed successfully")

    if exit_code == 0:
        # 결과 병합
        merge_results(config_manager, config.parallel.dp_size, logger)

        # 통계 출력
        output_path = config_manager.get_output_path()
        with jsonlines.open(output_path, 'r') as reader:
            results = list(reader)

        problems_processed = len(set(r['problem_idx'] for r in results))
        avg_samples = len(results) / \
            problems_processed if problems_processed > 0 else 0

        logger.info("="*60)
        logger.info("Final Statistics:")
        logger.info("="*60)
        logger.info(f"Problems processed: {problems_processed}")
        logger.info(f"Total samples: {len(results)}")
        logger.info(f"Average samples per problem: {avg_samples:.2f}")
        logger.info("="*60)
        logger.info("Inference completed successfully!")
    else:
        logger.error(f"Inference failed with exit code {exit_code}")

    exit(exit_code)


if __name__ == "__main__":
    main()
