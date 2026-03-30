import numpy as np
import pysindy as ps
import gymnasium as gym
import torch
import random
import time

from Utils.OnlinePredictor import OnlinePredictor
from Buffer.ReplayBuffer import ReplayBuffer
from SoftActorCritic.SACNorm import SAC

from joblib import Parallel, delayed

def execute(episode_number, seed, distance, N_models, test_number, path):
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

    actor = SAC(obs_dim=env.observation_space.shape[0] + 28 * 2,
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

    batch_size = 512

    online_predictor = OnlinePredictor(regularization=1e-3, basis_functions=[], state_dim=2, act_dim=1, matrix_size=28, library=ps.PolynomialLibrary(degree=2) + ps.FourierLibrary(n_frequencies=3))

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
                extended_current_state = np.concatenate([current_state, online_predictor.w.flatten()])
                action = actor.sample_action(extended_current_state)
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

                        model_current_state, _, _, _, _ = env_buffer.sample(batch_size)
                        model_current_state = np.array(model_current_state)

                        for _ in range(N_models):
                            n, d = online_predictor.w.shape
                            k = len(model_current_state)
                            M = distance * online_predictor.get_inverse()
                            L0 = np.linalg.cholesky(M)
                            scale_factors = np.random.rand(k)
                            sqrt_scales = np.sqrt(scale_factors)
                            L = sqrt_scales[:, np.newaxis, np.newaxis] * L0
                            z = np.random.randn(k, n, d)
                            sampled_matrix = online_predictor.w + np.matmul(z, L.swapaxes(-1, -2))

                            augmented_states = np.hstack((model_current_state, sampled_matrix.reshape(len(model_current_state), -1)))

                            augmented_states_action = np.hstack((model_current_state, sampled_matrix[np.random.permutation(sampled_matrix.shape[0])].reshape(len(model_current_state), -1)))

                            model_action = actor.sample_action(augmented_states_action).reshape(-1, 1)

                            features_batch = sindy_library.fit_transform(np.concatenate((model_current_state, model_action), axis=1))

                            model_next_states = np.einsum('nd,nmd->nm', features_batch, sampled_matrix)

                            model_rewards = -0.1 * (model_action ** 2).squeeze() + np.where(((model_next_states[:, 0]) > 0.45), 100, 0)

                            model_dones = np.where((model_next_states[:, 0] > -1.2) &
                                                   (model_next_states[:, 0] < 0.45) &
                                                   (abs(model_next_states[:, 1]) < 0.07), False, True)

                            augmented_next_states = np.hstack((model_next_states, sampled_matrix.reshape(len(model_current_state), -1)))

                            actor.update(augmented_states, model_action, model_rewards, augmented_next_states, model_dones)

                            model_current_state = model_next_states[~model_dones]

                            if len(model_current_state) < batch_size:
                                new_states, _, _, _, _ = env_buffer.sample(batch_size - len(model_current_state))
                                new_states = np.array(new_states)
                                model_current_state = np.vstack((model_current_state, new_states))

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
    # Lambda
    distance_regulator = 0.1
    # Number of model interactions
    models_interactions = 5 * 1000
    # Number of seeds
    seeds = np.random.choice(range(0, 1000), 10, replace=False).tolist()
    # Path to save results, one file per seed
    path = ""

    Parallel(n_jobs=-1)(delayed(execute)(episode_numbers, s, distance_regulator, models_interactions, i, path) for s, i in zip(seeds, range(len(seeds))))

    print(f"ENDED IN {time.time() - starting_time}")
