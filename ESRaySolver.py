import numpy as np
import gym
import ray

from nn import NN
from common import *



def evaluate_model(model, env):
    
    observation = env.reset()
    done = False
    i = 0
    r_sum = 0
    while not done and i < 1500:
        action = model.predict(observation)
        observation, reward, done, _ = env.step(action)
        i += 1
        r_sum += reward
    return r_sum


@ray.remote(num_cpus=0.12)
class ESWorker:
    def __init__(self, layer_sizes, init_seed, env_name, seed, sigma=.1):
        self.model = NN(layer_sizes, init_seed)
        self.env = gym.make(env_name)
        self.rs = np.random.RandomState(seed=seed)
        self.sigma = sigma
    
    def evaluate(self):
        
        candidate = sample_like(self.model.weights, rs=self.rs)
        self.model.set_weights(combine_weights(self.model.get_weights(), candidate, self.sigma))
        reward = evaluate_model(self.model, self.env)
        return reward
    
    def update(self, weights):
        self.model.set_weights(weights, copy=True)


class ESRaySolver:
    def __init__(self, model, env_name, population_size=30, max_episode_len=1500,
                 lr=0.03, lr_decay=0.999, sigma=0.1, verbose=False):
        self.model = model
        self.env_name = env_name
        self.population_size = population_size
        self.max_episode_len = max_episode_len
        self.lr = lr
        self.lr_decay = lr_decay
        self.sigma = sigma
        self.verbose = verbose
        self.w_seeds = [seed for seed in range(population_size)]
        self.w_rss = [np.random.RandomState(seed=seed) for seed in self.w_seeds]
        ray.init(memory=4 * 1024 * 1024 * 1024, object_store_memory=4 * 1024 * 1024 * 1024)

        self.workers = [ESWorker.remote(model.layer_sizes, model.seed, env_name, seed) for seed in self.w_seeds]  # actor handles
    
    def solve(self, weights=None, fitness_fn=None, n_generations=100, seed=None):
        """
        If weights is none, simple MLP is assumed, otherwise this should be the list of weights matrices from some model
        """
        if weights is None:
            weights = self.model.get_weights(copy=True)
        if fitness_fn is None:
            fitness_fn = evaluate_model

        if seed is not None:
            np.random.seed(seed)

        lr = self.lr
        for generation in range(n_generations):
    
            rewards = [ray.get(w.evaluate.remote()) for w in self.workers]
            population = [sample_like(weights, rs=rs) for rs in self.w_rss]

            rewards = np.array(rewards)
            r_mean, r_std = rewards.mean(), rewards.std()
            rewards = (rewards - r_mean) / r_std
            
            update_params(weights, population, rewards, lr=lr, sigma=self.sigma)
            [ray.get(w.update.remote(weights)) for w in self.workers]
        
            lr = lr * self.lr_decay
            if self.verbose and (generation % int(self.verbose) == 0):
                print(f'[{generation}]: E[R]={r_mean:.4f}, std(R)={r_std:.4f} | lr={lr:.4f}')
        return weights
