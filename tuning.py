#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:20:14 2023

@author: ahmadrezafrh
"""

from env.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy

from utils import create_model, create_model_path
from utils import create_meta, create_env
from utils import check_path, save_meta, load_configue
from utils import ignore_warnings, print_model
from utils import custom_extractor

from models import CNNBaseExtractor
from models import CNNSimple
from models import CNNMobileNet
from models import CNNLstm
from models import CNNDecreasedFilters
from models import CNNInreasedFilters


import optuna
import os



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
    just optimize the paramaeters of the MLP's network.
    
    Finally, for domain randomization we just define multiple distribution and we use them
    to optimize the network with hypereparameters extracted above. (we do not consider
    domain randomization optimization with this approach)
        
    '''


     
    
    def optimize_ppo(trial):
        
        tot_conf = general_paramaeters()
        configue = {
            'n_steps':trial.suggest_int('n_steps', tot_conf['n_steps'][0], tot_conf['n_steps'][1], step=64),
            'gamma':trial.suggest_float('gamma', tot_conf['gamma'][0], tot_conf['gamma'][1]),
            'learning_rate':trial.suggest_float('learning_rate', tot_conf['learning_rate'][0], tot_conf['learning_rate'][1], log=True),
            'clip_range':trial.suggest_float('clip_range', tot_conf['clip_range'][0], tot_conf['clip_range'][1]),
            'gae_lambda':trial.suggest_float('gae_lambda', tot_conf['gae_lambda'][0], tot_conf['gae_lambda'][1]),
        }
        
            
        
        if tot_conf['obs']=='cnn':
            configue['smooth'] = trial.suggest_categorical('smooth', tot_conf['smooth'])
            configue['resize_shape'] = trial.suggest_categorical('resize_shape', tot_conf['resize_shape'])
            configue['preprocess'] = trial.suggest_categorical('preprocess', tot_conf['preprocess'])
            configue['n_frame_stacks'] = trial.suggest_categorical('n_frame_stacks', tot_conf['n_frame_stacks'])
        

            
        if tot_conf['custom_arch']:
            configue['policy_kwargs'] = trial.suggest_categorical('policy_kwargs', tot_conf['policy_kwargs'])
            
        return configue
    
    def general_paramaeters():
        
        configues_dir = './configues/optuna'
        conf_name = 'optuna.json'
        configue = load_configue(os.path.join(configues_dir, conf_name))
        return configue
    
    def generate_architectures():
        cnn_architectures = {
            'base': custom_extractor(CNNBaseExtractor, 512),
            'simple': custom_extractor(CNNSimple, 128),
            'mobile_net': custom_extractor(CNNMobileNet, 128),
            'cnn_lstm': custom_extractor(CNNLstm, 256),
            'large' : custom_extractor(CNNInreasedFilters, 512),
            'small' : custom_extractor(CNNDecreasedFilters, 128)
        }
        
        return cnn_architectures
    
    
    def optimize_agent(trial):
        

        model_params = optimize_ppo(trial)
        other_params = general_paramaeters()
        check_path(other_params['logs_dir'])
        check_path(other_params['models_dir'])
        meta = create_meta(other_params, model_params, method='optuna')
        cnn_architectures = generate_architectures()
        ignore_warnings(meta['ignore_warnings'])
        
        
        
        env = create_env(meta)
        model = create_model(env, meta, logs_dir=other_params['logs_dir'], policy_kwargs=cnn_architectures)
        print_model(meta)

        
        model_path = create_model_path(other_params['models_dir'])
        check_path(model_path)
        save_meta(meta, model_path)


        model.learn(total_timesteps=meta['time_steps'], tb_log_name=f"{model_path}")
        model.save(os.path.join(model_path, f"{meta['time_steps']}"))
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50)

        del model
        env.close()
        
        return mean_reward
    

    cfg = general_paramaeters()
    n_trials = cfg['n_trials']
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=n_trials, n_jobs=1)
    print(study.best_params)
            
if __name__ == '__main__':
    main()
