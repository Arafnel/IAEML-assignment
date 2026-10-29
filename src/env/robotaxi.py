import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from .base import BaseEnv, BaseEnvState, BaseEnvParams, BaseEnvObservation
from typing import NamedTuple

class EnvState(NamedTuple):
    x: float
    y: float
    theta: float # угол ориентации, куда смотрит машина
    v: float # crjhjcnm
    # если препятствия движутся, то дописать сюда

class EnvParams(NamedTuple):
    dt: float = 0.1 # шаг времени дискретизации
    max_accel: float = 0.3 # максимальное ускорение
    max_steering: float = 0.5 # максимальный угол поворота руля
    friction: float = 0.1 # коэф трения (замедление)

class RobotaxiEnv(BaseEnv):

    def init_params(self) -> Tubple[EnvParams, EnvState]:
        params = EnvParams()
        init_state = EnvState(x=0.0, y=0.0, theta=0.0, v=0.0) # иниц в начале координат
        return params, init_state

    def step(self,
             key: jax.random.PRNGKey,
             state: EnvState,
             action: jnp.ndarray,
             params: EnvParams) -> Tuple[EnvParams, EnvState, float, bool, Dict[str, Any]]:
        accel, steering = action[0], action[1]

        accel = jnp.clip(accel, -params.max_accel, params.max_accel) # ограничение экшенов (клип), чтобы физика не взрывалась
        steering = jnp.clip(steering, -params.max_steering, params.max_steering)

        L = 2.0 # Динамика - длина между осями у автомобился пусть будет 2.0

        new_x = state.x + state.v * jnp.cos(state.theta) * params.dt
        new_y = state.y + state.v * jnp.sin(state.theta) * params.dt
        new_theta = state.theta + (state.v / L) * params.dt + jnp.tan(steering)

        new_v = state.v + (accel - params.friction * state.v) * params.dt
        new_v = jnp.maximum(new_v, 0.0)

        next_state = EnvState(x=new_x, y=new_y, theta=new_theta, v=new_v)

        # награды
        reward = 0.0
        done = False #при прохождении будет True

        return next_state, jnp.array(reward), done, {}

    def get_observation(
            self,
            env_state: EnvState,
            env_params: EnvParams) -> jnp.ndarray:

        pass