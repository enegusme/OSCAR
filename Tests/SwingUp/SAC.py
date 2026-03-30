import numpy as np
import torch
import random
import time
from SoftActorCritic.SAC import SAC
from Buffer.ReplayBuffer import ReplayBuffer
from dm_control import suite

from joblib import Parallel, delayed

def execute(episode_number, seed, test_number, path):
    env = suite.load(domain_name="cartpole", task_name="swingup", task_kwargs={'random': seed})

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    actor = SAC(obs_dim=5,
              act_dim=1,
              act_min=np.array([-1]),
              act_max=np.array([1]),
              std_min=-5,
              std_max=2,
              policy_hidden=256,
              q_hidden=256,
              actor_lr=3e-4,
              critic_lr=1e-3,
              discount_factor=0.99,
              update_rate=0.005,
              temperature=0.2,
              autotune=False)

    env_buffer = ReplayBuffer(int(1e6))

    result_reward = []

    for ep in range(episode_number):
        current_state = env.reset()
        current_state = np.concatenate([obs.flatten() for obs in current_state.observation.values()])

        done = False

        ep_reward = 0
        ep_length = 0

        while not done:

            ep_length += 1

            action = actor.sample_action(current_state)

            time_step = env.step(action)
            reward = time_step.reward
            done = time_step.last()
            next_state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])

            ep_reward += reward

            env_buffer.insert(current_state, action, reward, next_state, done)

            if len(env_buffer) > 256:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = env_buffer.sample(256)
                actor.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

            current_state = next_state

        result_reward.append(ep_reward)
        print(f"{ep}/{ep_length} Reward: {ep_reward:05.2f}")

        np.save(path + f"/rewards{test_number}", np.array(result_reward))

if __name__ == "__main__":
    starting_time = time.time()
    # How many episodes
    episode_numbers = 300
    # Number of seeds
    seeds = np.random.choice(range(0, 1000), 10, replace=False).tolist()
    # Path to save results, one file per seed
    path = ""

    Parallel(n_jobs=-1)(delayed(execute)(episode_numbers, s, i, path) for s, i in zip(seeds, range(len(seeds))))

    print(f"ENDED IN {time.time() - starting_time}")
