import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum

from agent.config import AgentConfig
from ctrnn.config import CTRNNConfig
from environment.config import EnvConfig
from ga.config import GAConfig
from plasticity.config import HPConfig


class Condition(Enum):
    HP_OFF = "HP_OFF"
    HP_DEV_ONLY = "HP_DEV_ONLY"
    HP_BEHAVIOUR_ONLY = "HP_BEHAVIOUR_ONLY"
    HP_BOTH = "HP_BOTH"


@dataclass
class RunConfig:
    ctrnn: CTRNNConfig = field(default_factory=CTRNNConfig)
    hp: HPConfig = field(default_factory=HPConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    condition: Condition = Condition.HP_OFF
    seed: int = 0
    output_dir: str = ""
    dev_phase_steps: int = 6000
    git_commit: str = ""


def run_config_to_json(config: RunConfig, path: str) -> None:
    d = dataclasses.asdict(config)
    d["condition"] = config.condition.name
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def run_config_from_json(path: str) -> RunConfig:
    with open(path) as f:
        d = json.load(f)
    d["condition"] = Condition[d["condition"]]
    d["ctrnn"] = CTRNNConfig(**d["ctrnn"])
    d["hp"] = HPConfig(**d["hp"])
    d["env"] = EnvConfig(**d["env"])
    d["agent"] = AgentConfig(**d["agent"])
    d["ga"] = GAConfig(**d["ga"])
    return RunConfig(**d)
