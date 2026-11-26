import jax
import equinox as eqx

from env.robotaxi import RobotaxiEnv
from algorithms.train import ActorCritic
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper


key = jax.random.PRNGKey(0)
env_params, init_state = RobotaxiEnv.init_params(
    key=key,
    map_id=2,
    max_steps=1000,
    discretization_scale=1,
    path_length=100,
    fps=60,
    perception_radius=5.0,
    num_ray_sensors=32,
)

env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)

# размерность наблюдений, скелет модели
sample_obs = RobotaxiEnv.get_observation(init_state, env_params).as_vector()
obs_dim = sample_obs.shape[-1]
action_dim = 2

skeleton_model = ActorCritic(obs_dim, action_dim, key)

# загрузка обученных весов из чекпоинта
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
checkpoint_path = os.path.join(base_dir, "checkpoints", "ppo_robotaxi.eqx")
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(os.path.dirname(__file__), "algorithms", "checkpoints", "ppo_robotaxi.eqx")

trained_model = eqx.tree_deserialise_leaves(checkpoint_path, skeleton_model)

#функция инференса агента (через JIT для 60 fps)
@eqx.filter_jit
def agent_policy(state):
    obs_vec = RobotaxiEnv.get_observation(state, env_params).as_vector()
    mean, _, _ = trained_model(obs_vec)
    return mean


# запуск рендерера в режиме оценки (eval_mode=True)
frontend = PygameFrontend(
    env,
    env_params,
    init_state,
    eval_mode=True,
    agent_fn=agent_policy,
)
frontend.run()