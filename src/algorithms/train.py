import os
import sys
import time
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.robotaxi import RobotaxiEnv, RobotaxiEnvParams, RobotaxiEnvState
from utils.autoreset import AutoResetWrapper


# 1. архитектура Actor-Critic на equinox
class ActorCritic(eqx.Module):
    actor_mlp: eqx.nn.MLP
    critic_mlp: eqx.nn.MLP
    log_std: jax.Array

    def __init__(self, obs_dim: int, action_dim: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.actor_mlp = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=action_dim,
            width_size=64,
            depth=2,
            activation=jax.nn.tanh,
            key=k1,)
        self.critic_mlp = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=1,
            width_size=64,
            depth=2,
            activation=jax.nn.tanh,
            key=k2,)
        # exploration
        self.log_std = jnp.zeros((action_dim,), dtype=jnp.float32)

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # среднее значение действий в [-1, 1] через tanh
        mean = jnp.tanh(self.actor_mlp(obs))
        std = jnp.exp(self.log_std)
        value = jnp.squeeze(self.critic_mlp(obs), axis=-1)
        return mean, std, value


# transition
class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array


# для работы с норм распределением
def gaussian_log_prob(mean: jax.Array, std: jax.Array, action: jax.Array) -> jax.Array:
    var = std ** 2
    log_scale = jnp.log(std)
    return jnp.sum(
        -0.5 * (((action - mean) ** 2) / var + 2.0 * log_scale + jnp.log(2.0 * jnp.pi)),
        axis=-1,)

def gaussian_entropy(std: jax.Array) -> jax.Array:
    return jnp.sum(0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(std), axis=-1)

# 2. функция Generalized Advantage Estimation (GAE) через lax.scan
def compute_gae(
        rewards: jax.Array,
        values: jax.Array,
        dones: jax.Array,
        last_value: jax.Array,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,) -> Tuple[jax.Array, jax.Array]:
    def _gae_step(carry, transition_data):
        gae, next_value = carry
        reward, value, done = transition_data
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, value), gae

    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = jax.lax.scan(
        _gae_step,
        init_carry,
        (rewards, values, dones),
        reverse=True,)
    returns = advantages + values
    return advantages, returns

def train():
    # гиперпараметры обучения
    NUM_ENVS = 64  # количество параллельных сред (vmap)
    NUM_STEPS = 128  # длина роллаута на одну среду перед обновлением
    TOTAL_UPDATES = 400  # количество циклов обновления PPO
    PPO_EPOCHS = 4  # количество проходов по собранным данным
    MINIBATCH_SIZE = 256  # размер мини-батча
    LR = 3e-4  # learning rate
    CLIP_EPS = 0.2  # параметр клиппинга PPO
    ENT_COEF = 0.01  # коэффициент энтропии
    VF_COEF = 0.5  # вес ошибки критика
    MAX_GRAD_NORM = 0.5  # клиппинг градиентов

    key = jax.random.PRNGKey(42)
    key, env_key, model_key = jax.random.split(key, 3)

    # инициализация среды
    raw_states = []
    for m_id in [0, 1, 2]:
        p, st = RobotaxiEnv.init_params(
            key=env_key,
            map_id=m_id,
            max_steps=600,
            discretization_scale=1,
            path_length=100,
            fps=60,
            perception_radius=5.0,
            num_ray_sensors=32,
        )
        raw_states.append(st)
    env_params = p

    max_static = max(s.static_obstacles.shape[0] for s in raw_states)
    max_kin = max(s.kinematic_obstacles.shape[0] for s in raw_states)

    padded_states = []
    for st in raw_states:
        pad_s = max_static - st.static_obstacles.shape[0]
        if pad_s > 0:
            st = eqx.tree_at(
                lambda t: t.static_obstacles,
                st,
                jnp.pad(st.static_obstacles, ((0, pad_s), (0, 0)), constant_values=-100.0),
            )
        pad_k = max_kin - st.kinematic_obstacles.shape[0]
        if pad_k > 0:
            st = eqx.tree_at(
                lambda t: t.kinematic_obstacles,
                st,
                jnp.pad(st.kinematic_obstacles, ((0, pad_k), (0, 0)), constant_values=-100.0),
            )
            st = eqx.tree_at(
                lambda t: t.kinematic_obst_velocities,
                st,
                jnp.pad(st.kinematic_obst_velocities, ((0, pad_k), (0, 0)), constant_values=0.0),
            )
        padded_states.append(st)

    all_init_states = jax.tree.map(lambda *xs: jnp.stack(xs), *padded_states)

    map_assignments = jnp.arange(NUM_ENVS) % 3
    batch_init_states = jax.tree.map(lambda x: x[map_assignments], all_init_states)

    def single_reset(k, init_st):
        return RobotaxiEnv.reset(k, env_params, init_st)

    def single_step(k, st, act, init_st):
        obs, next_st, reward, done, info = RobotaxiEnv.step(k, st, act, env_params)
        obs, next_st = jax.lax.cond(
            done,
            lambda _: single_reset(k, init_st),
            lambda _: (obs, next_st),
            None,
        )
        return obs, next_st, reward, done, info

    batch_reset = jax.vmap(single_reset, in_axes=(0, 0))
    batch_step = jax.vmap(single_step, in_axes=(0, 0, 0, 0))

    env_keys = jax.random.split(env_key, NUM_ENVS)
    obs_batch, env_states = batch_reset(env_keys, batch_init_states)
    obs_vec_batch = jax.vmap(lambda o: o.as_vector())(obs_batch)
    obs_dim = obs_vec_batch.shape[-1]
    action_dim = 2

    print(f"Obs dim: {obs_dim} | Parallel envs: {NUM_ENVS} | Device: {jax.devices()[0]}")

    # инициализация модели и оптимизатора optax
    model = ActorCritic(obs_dim, action_dim, model_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(learning_rate=LR, eps=1e-5),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # разделение модели на обучаемые веса, функции активации
    init_params, static_model = eqx.partition(model, eqx.is_array)

    # loss function PPO для одного мини батча
    def ppo_loss_fn(params, batch: Tuple[jax.Array, ...]):
        curr_model = eqx.combine(params, static_model)
        b_obs, b_actions, b_log_probs, b_advantages, b_returns = batch
        means, stds, values = jax.vmap(curr_model)(b_obs)

        new_log_probs = gaussian_log_prob(means, stds, b_actions)
        entropy = jnp.mean(gaussian_entropy(stds))

        ratio = jnp.exp(new_log_probs - b_log_probs)
        adv_norm = (b_advantages - jnp.mean(b_advantages)) / (jnp.std(b_advantages) + 1e-8)

        loss_actor1 = ratio * adv_norm
        loss_actor2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_norm
        actor_loss = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))

        critic_loss = 0.5 * jnp.mean((values - b_returns) ** 2)

        total_loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy
        return total_loss, (actor_loss, critic_loss, entropy)

    # полный шаг обучения, через JIT
    @jax.jit
    def update_step(carry, _):
        params, opt_state, env_states, obs_vec, rng = carry

        #Rollout
        def _env_step(step_carry, _):
            states, curr_obs, step_rng = step_carry
            step_rng, act_rng, env_rng = jax.random.split(step_rng, 3)

            curr_model = eqx.combine(params, static_model)
            means, stds, values = jax.vmap(curr_model)(curr_obs)
            noise = jax.random.normal(act_rng, shape=means.shape)
            actions = jnp.clip(means + stds * noise, -1.0, 1.0)
            log_probs = gaussian_log_prob(means, stds, actions)

            step_keys = jax.random.split(env_rng, NUM_ENVS)
            next_obs_struct, next_states, rewards, dones, _ = batch_step(
                step_keys, states, actions, batch_init_states
            )
            next_obs_vec = jax.vmap(lambda o: o.as_vector())(next_obs_struct)

            transition = Transition(
                obs=curr_obs,
                action=actions,
                log_prob=log_probs,
                reward=rewards,
                done=dones,
                value=values,
            )
            return (next_states, next_obs_vec, step_rng), transition

        (env_states, obs_vec, rng), traj = jax.lax.scan(
            _env_step, (env_states, obs_vec, rng), None, length=NUM_STEPS
        )

        # расчет GAE
        curr_model = eqx.combine(params, static_model)
        _, _, last_values = jax.vmap(curr_model)(obs_vec)
        advantages, returns = compute_gae(
            traj.reward, traj.value, traj.done, last_values
        )

        batch_size = NUM_STEPS * NUM_ENVS
        flat_obs = traj.obs.reshape((batch_size, obs_dim))
        flat_actions = traj.action.reshape((batch_size, action_dim))
        flat_log_probs = traj.log_prob.reshape((batch_size,))
        flat_advantages = advantages.reshape((batch_size,))
        flat_returns = returns.reshape((batch_size,))

        # обновление весов PPO по эпохам и мини батчам
        def _ppo_epoch(epoch_carry, _):
            curr_params, curr_opt_state, epoch_rng = epoch_carry
            epoch_rng, perm_rng = jax.random.split(epoch_rng)
            perm = jax.random.permutation(perm_rng, batch_size)

            num_minibatches = batch_size // MINIBATCH_SIZE
            mb_indices = perm[: num_minibatches * MINIBATCH_SIZE].reshape(
                (num_minibatches, MINIBATCH_SIZE)
            )

            def _minibatch_update(mb_carry, idx):
                p, opt_s = mb_carry
                mb_data = (
                    flat_obs[idx],
                    flat_actions[idx],
                    flat_log_probs[idx],
                    flat_advantages[idx],
                    flat_returns[idx],)
                (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(p, mb_data)
                updates, opt_s = optimizer.update(grads, opt_s, p)
                p = optax.apply_updates(p, updates)
                return (p, opt_s), loss

            (curr_params, curr_opt_state), losses = jax.lax.scan(_minibatch_update, (curr_params, curr_opt_state), mb_indices)
            return (curr_params, curr_opt_state, epoch_rng), jnp.mean(losses)

        (params, opt_state, rng), epoch_losses = jax.lax.scan(
            _ppo_epoch, (params, opt_state, rng), None, length=PPO_EPOCHS)

        metrics = {
            "mean_reward": jnp.mean(traj.reward),
            "loss": jnp.mean(epoch_losses),
        }
        return (params, opt_state, env_states, obs_vec, rng), metrics

    # запуск
    print("compiling and learning ppo")
    start_time = time.time()
    carry = (init_params, opt_state, env_states, obs_vec_batch, key)

    for update in range(1, TOTAL_UPDATES + 1):
        carry, metrics = update_step(carry, None)
        if update % 25 == 0 or update == 1:
            elapsed = time.time() - start_time
            print(f"Update {update:3d}/{TOTAL_UPDATES} | "
                f"Mean Step Reward: {float(metrics['mean_reward']):.4f} | "
                f"Loss: {float(metrics['loss']):.4f} | "
                f"Time: {elapsed:.1f}s")

    # финальная модель из обученных весов и статики
    trained_model = eqx.combine(carry[0], static_model)

    # сохранение чекпоинта модели
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/ppo_robotaxi.eqx"
    eqx.tree_serialise_leaves(checkpoint_path, trained_model)
    print(f"веса в {checkpoint_path}")

if __name__ == "__main__":
    train()