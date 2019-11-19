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

In order to run distributed version you need [Ray](https://github.com/ray-project/ray): `pip install ray psutil`

If you encounter an error, contact me. It's likely that this will break in the future due to dependencies.

![gif](ezgif-1-f3294c3febd5.gif)
## Server

The purpose of Server is to synchronize progress across multiple Clients as well as distribute work to each of the Client. It does so by creating a list of Client actors, initializing them with model architecture, random seed used for model initialization, seed for perturbation generation, and environment identifier.


## Client
Client is initialized with it's personal random seed that is known for Server. When `evaluate` method
is called, it samples weights perturbation according to it's seed and evaluates model with it, sending
only the reward back to Server.

Client can run `evaluate` multiple times with perturbation added to the same set of weights.


Once Server is done distributing evaluation across Clients, it collects the rewards and reproduces
perturbations on the client nodes. It then proceeds with performing weights update according with the
Evolution Strategy. It then broadcasts new weights across all clients by calling `update` method.

## Bibliography
* [Evolution Strategies as a Scalable Alternative to RL](https://openai.com/blog/evolution-strategies/)
* [Mirrored Sampling](https://hal.inria.fr/inria-00530202v2/document)
