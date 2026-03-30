# Provably Efficient Reinforcement Learning for Sparse Dynamical Systems with Non-Gaussian Noise


## OPTIMISTIC SIMULATION WITH CONFIDENCE-BALL REGRESSION (OSCAR)
OSCAR is a model-based RL algorithm which exploits the uncertainty
of the learned model to guide the exploration.
It builds a confidence ball around the coefficients of a learned SINDy
model and explore these dynamics learning a policy parametrized
by the coefficients of the linear model other than the current state 
of the agent.

OSCAR was tested over three different environments Cartpole SwingUp,
Acrobot and MountainCar Continuous overperforming the baselines of Soft
Actor-Critic (SAC), model-based SAC with SINDy and Dreamer.

<p align="center">
  <img src="/img/SwingUpDreamer.png" width="44%">
  <img src="/img/AcrobotDreamer.png" width="45%">
</p>

<p align="center">
  <img src="/img/MountainCarDreamer.png" width="45%">
</p>

## Instructions
The code allows to replicate the results obtained in the paper running
the relative file to create the numpy array with the returns obtained.

To run each file:
```powershell
python .\Test\<env>\<algorithm>.py
```

The available environments are:
- Acrobot
- MountainCar
- SwingUp

The available algorithms are:
- OSCAR
- SAC
- SINDy