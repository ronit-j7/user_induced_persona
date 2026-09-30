from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "main"
    num_layers: int = 28
    num_heads: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128
    hidden_size: int = 3584
    smh_layer: int = 19
    smh_heads: tuple = (2, 4, 27)
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    attention_implementation: str = "sdpa"
    max_length: int = 2048
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "smh_heads", tuple(self.smh_heads))
        if min(self.num_layers, self.num_heads, self.num_kv_heads, self.head_dim, self.max_length) <= 0:
            raise ValueError("Model dimensions and max_length must be positive")
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("This adapter requires hidden_size = query_heads * head_dim")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if not 0 <= self.smh_layer < self.num_layers:
            raise ValueError("SMH layer out of range")
        if len(set(self.smh_heads)) != len(self.smh_heads) or not self.smh_heads:
            raise ValueError("SMH heads must be unique and nonempty")
        if any(not 0 <= h < self.num_heads for h in self.smh_heads):
            raise ValueError("SMH head out of range")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dtype")

    def as_dict(self):
        return asdict(self)


def load_config(path):
    return ModelConfig(**json.loads(Path(path).read_text()))
