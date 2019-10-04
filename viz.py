
def render_env(model, env, max_iter=None, verbose=True):
    """
    Render model performance on the environment
    """
    observation = env.reset()
    done = False
    i = 0
    r_sum = 0
    while not done:
        if (max_iter is not None and i >= max_iter):
            break
        env.render()
        action = model.predict(observation)
        observation, reward, done, _ = env.step(action)
        i += 1
        r_sum += reward
    if verbose:
        print(f"Episode end after {i} iterations with reward = {r_sum} and done status {done}")