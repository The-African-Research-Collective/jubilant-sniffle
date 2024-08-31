from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import argparse
import yaml
from typing import List, Optional, Type, TypeVar

T = TypeVar('T', bound='BaseConfig')

@dataclass_json
@dataclass
class BaseConfig:
    @classmethod
    def from_yaml(cls: Type[T], yaml_file: str) -> T:
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = '') -> None:
        for field_name, field_type in cls.__annotations__.items():
            parser.add_argument(
                f"--{prefix}{field_name.replace('_', '-')}", 
                type=field_type,
                help=f"{field_name}"
            )

@dataclass_json
@dataclass
class TaskConfig(BaseConfig):
    task_name: str = field(default="default_task")
    languages: List[str] = field(default_factory=list)
    task_template: str = field(default="")
    wandb_project: str = field(default="default_project")
    wandb_job_type: str = field(default="eval")

@dataclass_json
@dataclass
class ModelConfig(BaseConfig):
    model_name: str = field(default="default_model")
    languages_supported: List[str] = field(default_factory=list)
    model_type: str = field(default="hf")
    trust_remote_code: bool = field(default=False)
    parallelize: bool = field(default=False)
    add_bos_token: bool = field(default=False)

@dataclass_json
@dataclass
class EvaluationConfig(BaseConfig):
    batch_size: int = field(default=8)
    max_batch_size: int = field(default=8)
    num_fewshot: int = field(default=0)
    random_seed: int = field(default=42)
    numpy_random_seed: int = field(default=42)
    torch_random_seed: int = field(default=42)
    fewshot_random_seed: int = field(default=42)
    use_cache: bool = field(default=True)
    log_samples: bool = field(default=True)
    write_out: bool = field(default=False)
    limit: int = field(default=-1)


@dataclass_json
@dataclass
class ScriptConfig(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    model_config_yaml: Optional[str] = field(default=None)
    task_config_yaml: Optional[str] = field(default=None)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        for field_name, field_type in cls.__annotations__.items():
            if field_name in ['model_config_yaml', 'task_config_yaml']:
                parser.add_argument(
                    f"--{field_name.replace('_', '-')}",
                    type=str,
                    default=None,
                    help=f"Path to YAML file for {field_name.split('_')[0]} configuration"
                )
            elif isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                field_type.add_args(parser, prefix=f"{field_name}.")
            else:
                parser.add_argument(
                    f"--{field_name.replace('_', '-')}",
                    type=field_type,
                    help=f"{field_name} (default: )"
                )

    @classmethod
    def from_args(cls) -> 'ScriptConfig':
        parser = argparse.ArgumentParser(description="Evaluation Configuration")
        cls.add_args(parser)
        args = parser.parse_args()
        
        config = cls()
        
        if args.model_config_yaml:
            config.model = ModelConfig.from_yaml(args.model_config_yaml)
            config.model_config_yaml = args.model_config_yaml
        
        if args.task_config_yaml:
            config.task = TaskConfig.from_yaml(args.task_config_yaml)
            config.task_config_yaml = args.task_config_yaml
        
        # Override with command-line arguments
        for key, value in vars(args).items():
            if key not in ['model_config_yaml', 'task_config_yaml'] and value is not None:
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        
        return config

def load_config() -> ScriptConfig:
    return ScriptConfig.from_args()