import math
import tempfile
import os

from agent.config import AgentConfig
from ctrnn.config import CTRNNConfig
from environment.config import EnvConfig
from experiments.config import Condition, RunConfig, run_config_from_json, run_config_to_json
from ga.config import GAConfig
from plasticity.config import HPConfig


def test_run_config_round_trip():
    original = RunConfig(
        ctrnn=CTRNNConfig(n_nodes=5, n_sensors=3, n_motors=2, dt=0.1, genotype_length=35),
        hp=HPConfig(h_low=0.1, h_high=0.9, tau_w=50.0, tau_b=25.0),
        env=EnvConfig(
            shape_radius=8.0,
            agent_radius=4.0,
            n_shapes=15,
            n_trials=8,
            spawn_height=90.0,
            spawn_x_range=20.0,
            vx_low=-0.2,
            vx_high=0.2,
            vy_low=-0.4,
            vy_high=-0.1,
        ),
        agent=AgentConfig(n_rays=5, ray_spread=math.pi / 4, s_max=8.0, d_max=150.0, tau_x=0.1),
        ga=GAConfig(pop_size=20, n_gens=200, n_runs=3, n_elite=3, p_m=0.05, sigma_m=0.4, tournament_k=5),
        condition=Condition.HP_BOTH,
        seed=42,
        output_dir="/tmp/test_results",
        dev_phase_steps=3000,
        git_commit="abc123",
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        run_config_to_json(original, path)
        recovered = run_config_from_json(path)
    finally:
        os.unlink(path)

    assert recovered == original

    assert isinstance(recovered.condition, Condition)
    assert recovered.condition is Condition.HP_BOTH

    assert isinstance(recovered.ctrnn, CTRNNConfig)
    assert isinstance(recovered.hp, HPConfig)
    assert isinstance(recovered.env, EnvConfig)
    assert isinstance(recovered.agent, AgentConfig)
    assert isinstance(recovered.ga, GAConfig)
