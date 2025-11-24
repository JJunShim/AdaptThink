from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """모델 설정"""

    name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    max_tokens: int = 16384
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.9


@dataclass
class ParallelConfig:
    """병렬 처리 설정"""

    dp_size: int = 2
    tp_size: int = 1
    node_size: int = 1
    node_rank: int = 0
    master_addr: str = ""
    master_port: int = 0


@dataclass
class DataConfig:
    """데이터 설정"""

    dataset_path: str = "./data/train/deepscaler.json"
    num_samples: int = 10
    K: int = 16


@dataclass
class SamplingConfig:
    """샘플링 설정"""

    temperature: float = 0.6
    top_p: float = 0.95
    seed_offset: int = 42


@dataclass
class PromptConfig:
    """프롬프트 설정"""

    nothinking: bool = False
    think: str = (
        "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    nothink: str = (
        "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    )


@dataclass
class OutputConfig:
    """출력 설정"""

    output_dir: str = "./data/train/ref_presampling"
    save_intermediate: bool = True


@dataclass
class Config:
    """전체 설정"""

    config: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
