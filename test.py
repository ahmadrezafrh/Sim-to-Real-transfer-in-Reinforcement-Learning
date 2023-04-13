# -*- coding: utf-8 -*-

import gym
import pandas as pd
import os
import csv

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy



def main():
    
    one_dir = True
    domain_randomization = True
    save_data = True
    eval_ep = 50
    policy = 'mlp'
    network = 'emprical'
    rewards_dir = './rewards'
    models_dir = './models/ppo/mlp/brute_force/domain_randomized/emprical'
    model_ts = '500000'
    one_model = 'MlpPolicy_0.0004215440315139262_0.9917611787616847_0.12587511822701497_0_3904_0.959132778841134_[[1.5, 2.4]]'
    
    rewards = {}
    rewards['learning_rate'] = []
    rewards['gamma'] = []
    rewards['clip_range'] = []
    rewards['ent_coef'] = []
    rewards['n_steps'] = []
    rewards['gae_lambda'] = []
    rewards['randomization distribution'] = []
    rewards['policy'] = []
    rewards['network'] = []
    rewards['eval episodes'] = []
    rewards['source reward'] = []
    rewards['target reward'] = []
    
    if one_dir:
        models_params = [one_model]
    else:
        models_params = []
        for rootdir, dirs, files in os.walk(models_dir):
            for subdir in dirs:
                models_params.append(subdir)
    
    
    for c, model_dir in enumerate(models_params):
        mod = os.path.join(models_dir, model_dir, model_ts)
        env_source = gym.make("CustomHopper-source-v0")
        env_target = gym.make("CustomHopper-target-v0")
        
        envs = {
            'source': env_source,
            'target': env_target
        }
        
        splitted_dir = model_dir.split('_')
        if domain_randomization:
            randomization_distribution = splitted_dir[7][1:-1]
        else:
            randomization_distribution = "not randomized"
            
        source_target_reward = {}
        for env in envs.items():
            
            print(f'evaluating model ({c}/{len(models_params)} on {env[0]})')
            model = PPO.load(mod, env=env[1])
            mean_reward, std_reward = evaluate_policy(model,
                                                      model.get_env(),
                                                      n_eval_episodes=eval_ep)
            
            source_target_reward[env[0]] = mean_reward
            print(f'randomization distribution: {randomization_distribution}')
            print(f'mean reward: {mean_reward}')
            env[1].close()
            
        
        
        learning_rate=splitted_dir[1]
        gamma=splitted_dir[2]
        clip_range=splitted_dir[3]
        ent_coef=splitted_dir[4]
        n_steps=splitted_dir[5]
        gae_lambda=splitted_dir[6]

            
            

        rewards['learning_rate'].append(learning_rate)
        rewards['gamma'].append(gamma)
        rewards['clip_range'].append(clip_range)
        rewards['ent_coef'].append(ent_coef)
        rewards['n_steps'].append(n_steps)
        rewards['gae_lambda'].append(gae_lambda)
        rewards['randomization distribution'].append(randomization_distribution)
        rewards['eval episodes'].append(eval_ep)
        rewards['policy'].append(policy)
        rewards['network'].append(network)
        rewards['source reward'].append(source_target_reward['source'])
        rewards['target reward'].append(source_target_reward['target'])
        

            
    if not os.path.exists(rewards_dir):
        os.makedirs(rewards_dir)
    
    df = pd.DataFrame.from_dict(rewards)
    if save_data: 
        df.to_csv(os.path.join(rewards_dir, 'rewards.csv'), encoding='utf-8', mode='a')
        print(f"rewards saved in {os.path.join(rewards_dir, 'domain_randomization_scores.csv')}")
                    
                

        

         
if __name__ == '__main__':
    main()

