#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:20:14 2023

@author: ahmadrezafrh
"""

from env.custom_hopper import *

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
# from table_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO



import optuna
import gym
import os
import pickle

import warnings



def main():
    
    
    '''
    
    Another approach for hyperparameter tuning is using optuna.
    this approach have been supported by the the stable_baselines3.
    
    In this method we can choose n number of trials, and the model
    will be trained for n set of hyper parameters (all hyperparameters
    chosen randomly with a uniform distribution). 
    
    In brute force approach optimization (train.py), due to the hardware
    limitation, it is not possible to train with all possible set of hyperparameters.
    Therefore, we use a set of randomized hyperparameters chosen with a ubiform
    distribution.
    
    The reason we do this approach is because of the high sensitivity of RL models
    to hyperparameters.
    
    We use the best hyperparameters extracted from MLP and use it in CNN. However,
    it is better to optimize CNN seperately but due to the limtation of the software we 
    just optimize the paramaeters of the CNN's network.
    
    Finally, for domain randomization we just define multiple distribution and we use them
    to optimize the network with hypereparameters extracted above. (we do not consider
    domain randomization optimization with this approach)
        
    '''


    
    def optimize_ppo(trial):
        return {
            'n_steps':trial.suggest_int('n_steps', 1024, 4096, step=64),
            'gamma':trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'learning_rate':trial.suggest_loguniform('learning_rate', 1e-4, 1e-3),
            'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.3),
            'gae_lambda':trial.suggest_uniform('gae_lambda', 0.9, 0.99),
        }
     
    
    
    def general_paramaeters():
        return {
            'normalize':False,
            'time_steps':5e5,
            'time_steps_per_iter':30000,
            'env':"CustomHopper-source-v0"
        }
    
    def optimize_agent(trial):
        
        model_params = optimize_ppo(trial)
        other_params = general_paramaeters()
        
        
        logs_dir = os.path.join('./logs')
        models_dir = os.path.join('./models','ppo', 'mlp', f'optuna_trial_{trial.number}', 'not_domain_randomized', 'emprical')
        
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        env = gym.make(other_params['env'])
        if  other_params['normalize']:
            env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_obs=10.)  
            
        model = PPO(env=env,
                    policy='MlpPolicy',
                    verbose=0,
                    device="cuda",
                    tensorboard_log=logs_dir,
                    **model_params)
        
        ts = other_params['time_steps']
        ts_per_iter = other_params['time_steps_per_iter']
        models_dir_steps = os.path.join(models_dir, f"mlp_{model_params['learning_rate']}_{model_params['gamma']}_{model_params['clip_range']}_{model_params['n_steps']}_{model_params['gae_lambda']}")
        
        if not os.path.exists(models_dir_steps):
            os.makedirs(models_dir_steps)
            
        for i in range(int(ts/ts_per_iter)):
            model.learn(total_timesteps=ts_per_iter, reset_num_timesteps=False, tb_log_name=f"{models_dir_steps}")
            model.save(os.path.join(models_dir_steps, f"{ts_per_iter*i + ts_per_iter}"))
        
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50)
        del model
        env.close()
        
        return mean_reward
    
    
    
    


    study_num = 1
    n_trials = 150
    study_dir = os.path.join('./params', 'studies')
    params_dir = os.path.join('./params', 'best_params')
    ignore_warninngs = True
    
    
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
        
        
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    
    
    
    '''
    Ignoring all warnings for better visualization
    '''
    if ignore_warninngs:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
    
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=n_trials, n_jobs=1)
    print(study.best_params)
    
    while True:
        if os.path.exists(os.path.join(study_dir, f'best_params_{study_num}.pkl')):
            study_num += 1
            continue
        with open(os.path.join(study_dir, f'best_params_{study_num}.pkl'), 'wb') as hp:
            pickle.dump(study.best_params, hp)
            print(f'best hyperparameters saved in best_params_{study_num}.pkl')
        
        
        
        if os.path.exists(os.path.join(params_dir, f'study_{study_num}.pkl')):
            study_num += 1
            continue
        with open(os.path.join(params_dir, f'study_{study_num}.pkl'), 'wb') as st:
            pickle.dump(study, st)
            print(f'study saved in study_{study_num}.pkl')
        
        break
            
if __name__ == '__main__':
    main()
