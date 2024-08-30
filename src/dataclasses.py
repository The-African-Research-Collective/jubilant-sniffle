from typing import List
from pydantic import BaseModel, validator

class TaskConfig(BaseModel):
    task_name: str
    languages: List[str]
    task_template: str
    wandb_project: str
    wandb_job_type: str = "eval"

    @validator('languages', each_item=True)
    def validate_languages(cls, v):
        if not v:
            raise ValueError("Languages list cannot be empty. At least one language must be specified.")
        return v


class ModelConfig(BaseModel):
    model_name: str
    model_type: str = "hf"
    trust_remote_code: bool = False
    parallelize: bool = False
    add_bos_token: bool = False

    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ["hf"]  # Add more as needed
        if v not in allowed_types:
            raise ValueError(f"Invalid model type. Allowed types are: {', '.join(allowed_types)}")
        return v
