import numpy as np
from EScommon import *
import gym

class ESSolver:
    def __init__(self, model, env_name, population_size=30, max_episode_len=1500,
                 lr=0.05, lr_decay=0.999, sigma=0.1, verbose=False):
        self.model = model
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.population_size = population_size
        self.max_episode_len = max_episode_len
        self.lr = lr
        self.lr_decay = lr_decay
        self.sigma = sigma
        self.verbose = verbose
    
    def solve(self, weights=None, fitness_fn=None, n_generations=100, seed=None, detailed_log=False):
        """
        If weights is none, simple MLP is assumed, otherwise this should be the list of weights matrices from some model
        """
        if weights is None:
            weights = self.model.get_weights(copy=True)
        if fitness_fn is None:
            fitness_fn = self.evaluate_model

        if seed is not None:
            np.random.seed(seed)

        lr = self.lr
        for generation in range(n_generations):
    
            population = []
            rewards = []
            
            for i in range(self.population_size):
                candidate = sample_like(weights)
                
                weights_combined = combine_weights(weights, candidate, sigma=self.sigma)
                reward = fitness_fn(weights_combined)
                
                population.append(candidate)
                rewards.append(reward)
                if detailed_log:
                    print(f'[{generation}: {i}/{self.population_size}]: E[R]={reward:.4f}')

            
            rewards = np.array(rewards)
            r_mean, r_std = rewards.mean(), rewards.std()
            rewards = (rewards - r_mean) / r_std
            
            update_params(weights, population, rewards, lr=lr, sigma=self.sigma)
        
        
            lr = lr * self.lr_decay
            if self.verbose and (generation % int(self.verbose) == 0):
                print(f'[{generation}]: E[R]={r_mean:.4f}, std(R)={r_std:.4f} | lr={lr:.4f}')
        return weights
    
    
    def evaluate_model(self, weights):
        self.model.set_weights(weights)
        
        observation = self.env.reset()
        done = False
        i = 0
        r_sum = 0
        while not done and i < self.max_episode_len:
            action = self.model.predict(observation)
            observation, reward, done, _ = self.env.step(action)
            i += 1
            r_sum += reward
        return r_sum
    
