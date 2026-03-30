import numpy as np
import pysindy as ps
import gymnasium as gym
import torch
import random
import time

from Utils.OnlinePredictor import OnlinePredictor
from Buffer.ReplayBuffer import ReplayBuffer
from SoftActorCritic.SAC import SAC

from joblib import Parallel, delayed

def execute(episode_number, seed, N_models, test_number, path):
    env = gym.make("MountainCarContinuous-v0")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    sindy_library = ps.PolynomialLibrary(degree=2) + ps.FourierLibrary(n_frequencies=3)

    sindy_optim = ps.STLSQ(threshold=1e-3, alpha=1e-3)
    sindy_ensemble_optim = ps.optimizers.EnsembleOptimizer(opt=sindy_optim, bagging=True, library_ensemble=False)

    sindy_model = ps.SINDy(feature_library=sindy_library, optimizer=sindy_ensemble_optim, discrete_time=True)

    sindy_traj_states = []
    sindy_traj_actions = []

    actor = SAC(obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.shape[0],
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
                autotune=True)

    env_buffer = ReplayBuffer(int(1e6))
    model_buffer = ReplayBuffer(int(1e6))

    batch_size = 256

    online_predictor = OnlinePredictor(regularization=1e-3, basis_functions=[], state_dim=2, act_dim=1, matrix_size=28,
                                   library=ps.PolynomialLibrary(degree=2) + ps.FourierLibrary(n_frequencies=3))

    result_reward = []

    sindy_fitted = False

    q_fitted = False

    for ep in range(episode_number):
        current_state, _ = env.reset()
        done = False

        ep_reward = 0
        ep_length = 0

        episode_states = []
        episode_actions = []

        while not done:
            ep_length += 1

            if online_predictor.w is not None:
                action = actor.sample_action(current_state)
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            env_buffer.insert(current_state, action, reward, next_state, done)

            episode_states.append(current_state)
            episode_actions.append(action)

            ep_reward += reward

            if sindy_fitted:
                predicted = sindy_model.simulate(x0=current_state, t=2, u=np.array([action])).squeeze()[-1]
                online_predictor.partial_fit(current_state, action, predicted)
                if online_predictor.w is not None:
                    if not q_fitted or done:

                        q_fitted = True

                        model_current_state, _, _, _, _ = env_buffer.sample(1)
                        model_current_state = np.array(model_current_state)

                        for _ in range(N_models):
                            sampled_matrix = online_predictor.w

                            model_action = actor.sample_action(model_current_state).reshape(-1, 1)

                            features_batch = sindy_library.fit_transform(np.concatenate((model_current_state, model_action), axis=1))

                            model_next_states = features_batch @ sampled_matrix.T

                            model_rewards = -0.1 * (model_action ** 2).squeeze() + np.where(((model_next_states[:, 0]) > 0.45), 100, 0)

                            model_dones = np.where((model_next_states[:, 0] > -1.2) &
                                                   (model_next_states[:, 0] < 0.45) &
                                                   (abs(model_next_states[:, 1]) < 0.07), False, True)

                            model_buffer.insert(model_current_state[0], model_action[0], model_rewards[0], model_next_states[0], model_dones[0])

                            if len(model_buffer) > batch_size:
                                state_batch, action_batch, reward_batch, next_state_batch, done_batch = model_buffer.sample(batch_size)
                                actor.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

                            model_current_state = model_next_states

                            if model_dones[0] is True:
                                model_current_state, _, _, _, _ = env_buffer.sample(1)
                                model_current_state = np.array(model_current_state)

            current_state = next_state

        sindy_traj_states.append(np.array(episode_states))
        sindy_traj_actions.append(np.array(episode_actions))

        print(f"{ep:03}/{ep_length:03} Reward: {ep_reward:03}")

        if ep >= 1:
            sindy_model.fit(x=sindy_traj_states, t=1, u=sindy_traj_actions, multiple_trajectories=True)

            if not sindy_fitted:
                for ts, ta in zip(sindy_traj_states, sindy_traj_actions):
                    for state, action in zip(ts, ta):
                        next_state = sindy_model.simulate(x0=state, t=2, u=np.array([action])).squeeze()[-1]
                        online_predictor.partial_fit(state, action, next_state)
                sindy_fitted = True

        result_reward.append(ep_reward)

        np.save(path + f"/rewards{test_number}", np.array(result_reward))

if __name__ == "__main__":
    starting_time = time.time()
    # How many episodes
    episode_numbers = 100
    # Number of model interactions
    models_interactions = 5 * 1000
    # Number of seeds
    seeds = np.random.choice(range(0, 1000), 10, replace=False).tolist()
    # Path to save results, one file per seed
    path = ""

    Parallel(n_jobs=-1)(delayed(execute)(episode_numbers, s, models_interactions, i, path) for s, i in zip(seeds, range(len(seeds))))

    print(f"ENDED IN {time.time() - starting_time}")
