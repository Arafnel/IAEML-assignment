import jax
import jax.numpy as jnp
import equinox
import chex
from functools import partial
from typing import Tuple, Dict, Any
from .base import (BaseEnv,BaseEnvState,BaseEnvParams,BaseEnvObservation,NUM_RAY_SENSORS,FOV,FPS,)


class RobotaxiEnvParams(BaseEnvParams):
    max_accel: float = 5.0  # максимальное ускорение
    max_steering: float = 0.6  # максимальный угол поворота руля
    friction: float = 0.5  # коэф трения (замедление)
    wheelbase: float = 1.0

class RobotaxiEnvState(BaseEnvState):
    theta: chex.Scalar  # угол ориентации, куда смотрит машина
    v: chex.Scalar  # скорость

class RobotaxiEnvObservation(BaseEnvObservation):
    velocity: chex.Array

    def as_vector(self) -> chex.Array:
        return jnp.concatenate(
            [
                jnp.atleast_1d(self.distance_to_path),
                self.direction_of_path,
                self.collision_rays.flatten(),
                jnp.atleast_1d(self.velocity),
            ],
            axis=-1,
        )

class RobotaxiEnv(BaseEnv):
    def init_params(
            key: chex.PRNGKey,
            map_id: int = 0,
            max_steps: int = 1000,
            path_length: int = 50,
            discretization_scale: int = 1,
            perception_radius: float = 5.0,
            num_ray_sensors: int = NUM_RAY_SENSORS,
            fov: float = FOV,
            fps: float = FPS,
    ) -> Tuple[RobotaxiEnvParams, RobotaxiEnvState]:
        base_params, base_state = BaseEnv.init_params(
            key, map_id, max_steps, path_length, discretization_scale,
            perception_radius, num_ray_sensors, fov, fps
        )

        params = RobotaxiEnvParams(
            max_steps_in_episode=base_params.max_steps_in_episode,
            fps=base_params.fps,
            step_size=base_params.step_size,
            map_height_width=base_params.map_height_width,
            nav_grid_shape_dtype=base_params.nav_grid_shape_dtype,
            path_array_shape_dtype=base_params.path_array_shape_dtype,
            discretization_scale=base_params.discretization_scale,
            num_ray_sensors=base_params.num_ray_sensors,
            agent_FOV=base_params.agent_FOV,
            perception_radius=base_params.perception_radius,
            max_accel=5.0,
            max_steering=0.6,
            friction=0.5,
            wheelbase=1.0,
        )

        init_theta = jnp.arctan2(
            base_state.agent_forward_dir[1], base_state.agent_forward_dir[0]
        )

        # иниц в начале координат
        state = RobotaxiEnvState(
            time=base_state.time,
            goal_pos=base_state.goal_pos,
            agent_pos=base_state.agent_pos,
            agent_forward_dir=base_state.agent_forward_dir,
            static_obstacles=base_state.static_obstacles,
            kinematic_obstacles=base_state.kinematic_obstacles,
            kinematic_obst_velocities=base_state.kinematic_obst_velocities,
            path_array=base_state.path_array,
            v=jnp.asarray(0.0, dtype=jnp.float32),
            theta=init_theta.astype(jnp.float32),
        )

        return params, state

    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
            key: chex.PRNGKey,
            env_params: RobotaxiEnvParams,
            init_state: RobotaxiEnvState,
    ) -> Tuple[RobotaxiEnvObservation, RobotaxiEnvState]:
        obs = RobotaxiEnv.get_observation(init_state, env_params)
        return obs, init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(
            key: chex.PRNGKey,
            env_state: RobotaxiEnvState,
            action: chex.Array,
            env_params: RobotaxiEnvParams,
    ) -> Tuple[RobotaxiEnvObservation, RobotaxiEnvState, chex.Scalar, chex.Array, Dict[Any, Any]]:
        dt = env_params.step_size
        L = env_params.wheelbase

        # W/S - action (газ/тормоз); A/D - action (поворот руля)
        accel = action[1] * env_params.max_accel
        steering = -action[0] * env_params.max_steering

        accel = jnp.clip(accel, -env_params.max_accel,
                         env_params.max_accel)  # ограничение экшенов (клип), чтобы физика не взрывалась
        steering = jnp.clip(steering, -env_params.max_steering, env_params.max_steering)

        x, y = env_state.agent_pos[0], env_state.agent_pos[1]
        v = env_state.v
        theta = env_state.theta

        new_x = x + v * jnp.cos(theta) * dt
        new_y = y + v * jnp.sin(theta) * dt
        new_theta = theta + (v / L) * jnp.tan(steering) * dt
        new_theta = jnp.arctan2(jnp.sin(new_theta), jnp.cos(new_theta))

        new_v = v + (accel - env_params.friction * v) * dt
        new_v = jnp.maximum(new_v, 0.0)

        new_agent_pos = jnp.array([new_x, new_y], dtype=jnp.float32)
        new_forward_dir = jnp.array([jnp.cos(new_theta), jnp.sin(new_theta)], dtype=jnp.float32)

        kinematic_obstacles = BaseEnv._move_kinematic_obstacles(env_state, env_params)
        all_obstacles = jnp.concatenate(
            [env_state.static_obstacles, kinematic_obstacles], axis=0
        )

        goal_done = BaseEnv._check_goal(new_agent_pos, env_state.goal_pos)
        collision_done = BaseEnv._check_collisions(new_agent_pos, all_obstacles)
        time_done = env_state.time >= env_params.max_steps_in_episode
        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))  # при прохождении будет True

        new_time = env_state.time + 1
        pred = jnp.mod(new_time, env_params.fps) == 0
        path_array = jax.lax.cond(
            pred,
            lambda _: BaseEnv._find_path(
                new_agent_pos, env_state.goal_pos, all_obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        new_state = RobotaxiEnvState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_pos=new_agent_pos,
            agent_forward_dir=new_forward_dir,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
            v=new_v,
            theta=new_theta,
        )

        obs = RobotaxiEnv.get_observation(new_state, env_params)

        # награды
        forward_progress = jnp.dot(new_forward_dir, obs.direction_of_path) * new_v

        path_penalty = -0.1 * obs.distance_to_path
        steering_penalty = -0.05 * (steering ** 2)
        time_penalty = -0.01

        reward = 0.5 * forward_progress + path_penalty + steering_penalty + time_penalty

        reward = jnp.where(goal_done, reward + 100.0, reward)
        reward = jnp.where(collision_done, reward - 100.0, reward)

        info = {
            "time": new_time,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "velocity": new_v,
            "goal_done": goal_done,
            "collision_done": collision_done,
        }

        return obs, new_state, reward, done, info

    def get_observation(
            env_state: RobotaxiEnvState, env_params: RobotaxiEnvParams
    ) -> RobotaxiEnvObservation:
        base_obs = BaseEnv.get_observation(env_state, env_params)

        return RobotaxiEnvObservation(
            distance_to_path=base_obs.distance_to_path,
            direction_of_path=base_obs.direction_of_path,
            collision_rays=base_obs.collision_rays,
            velocity=jnp.asarray(env_state.v, dtype=jnp.float32),
        )