### Fast Evolution Strategy for Walking Marvin

This is a design doc for the implementation that I've come up with.

## Install Guide

1. Create a virtual environment `python3 -m venv marvin_env`
2. Activate it `source marvin_env/bin/activate`
3. Install Swig library `brew install swig`.
4. `pip install numpy==1.17.2 gym==0.14.0 Box2D==2.3.2 box2d-py==2.3.8`
5. Copy `gym` directory provided in this repo to `marvin_env/lib/python3.7/site-packages` (with replacement, like `cp -r gym marvin_env/lib/python3.7/site_packages`)
6. `import gym`
   `env = gym.make("Marvin-v0")` to create an environment
7. Other environments should work fine too `env = gym.make("BipedalWalker-v2)"`

If you encounter an error, contact me. It's likely that this will break in the future due to dependencies.

## Server

The purpose of Server is to synchronize progress across multiple Clients as well as distribute work to each of the Client.
In order to save make transmission lighter, Server work package consists of:
1. Random seed to be used by a client in order to generate perturbations. This seed is remembered on server-side in order to replicate
   perturbations during update phase.
2. Population size
3. Current model architecture (weights) to be recreated locally
4. Time limit

Client returns:
1. List of rewards for the given population size

* [Evolution Strategies as a Scalable Alternative to RL](https://openai.com/blog/evolution-strategies/)
* [Mirrored Sampling](https://hal.inria.fr/inria-00530202v2/document)