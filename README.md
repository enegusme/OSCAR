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

## Instructions
The code allows to replicate the results obtained in the paper running
the relative file to create the numpy array with the returns obtained.

To create and activate an `conda` environment with python 3.14:
```bash
conda create --name <env_name> python=3.14
activate <env_name>
```

To install the required packages:
```bash
pip install -r requirements.txt
```

To set the path where the results will be saved, 
modify the `PATH` variable inside the test you want to run.

To run the test:
```bash
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