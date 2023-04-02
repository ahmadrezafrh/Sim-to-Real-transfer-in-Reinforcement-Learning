# -*- coding: utf-8 -*-

import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy



def main():
    
    eval_ep = 50
    model_dir = "./models_dom/ppo/370000"
    env = gym.make("CustomHopper-target-v0")
    model = PPO.load(model_dir, env=env)
    mean_reward, std_reward = evaluate_policy(model,
                                              model.get_env(),
                                              n_eval_episodes=eval_ep)

    print(mean_reward)
    episodes = 500
    for i in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            
    env.close()

if __name__ == '__main__':
    main()

