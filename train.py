#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:20:14 2023

@author: ahmadrezafrh
"""

import os
from models import CNNBaseExtractor
from models import CNNSimple
from models import CNNMobileNet
from models import CNNLstm
from models import CNNDecreasedFilters
from models import CNNInreasedFilters

from utils import create_model, create_model_path
from utils import create_meta, create_env, create_params
from utils import check_path, save_meta, load_configue
from utils import ignore_warnings, print_model
from utils import custom_extractor


    
def main():
    
    
    '''
    Setting parameters for training in configue dictionary,
    
        1- alg : it is the algorithm we use, for this project for sake of simplicity and reducing
                 training time, we decided to choose PPO.
                   
        2- obs : it determines wether we use "mlp"(for raw states) or "cnn"(for pixels state).

        3- dist1/dist2/dist3 : they determine three distributions for three links
                               of the hopper.
        
        4- logs_dir/models_dir : these folders are used for saving all trained models.
                                 mainly used for visualization with tensorboard.
                                            
        5- ignore_warnings : ignores the tensor warnings (for better visualiziation of the progress bar).
        
        6- domain_randomization : if it's True we use domain randomization on.
        
        7- stacked_environment : dir for saving best trained models.
        
        8- gray_scale : changing RGP to gray scale.
        
        9- vec_normalize : wether we normalize input vectors or not.
        
        10- custom_arch : determines wether we use custom architecture or not.
        
        11- policies : list of policies we wanna use in tuning.
        
        12- n_frame_stacks : total frames stacked together for processing in cnn.
        
        13- time_steps : total time_steps.
        
        14- step_per_iter : total steps taken(frames) per iteration.
        
        15- policy_kwargs : are the parameters for custom architecture of actor/critic networks
                            (it can be different from mlp to cnn)
        
    '''

    
    configues_dir = './configues/gird'
    conf_name = 'configue.json'
    cnn_architectures = {
        'base': custom_extractor(CNNBaseExtractor, 256),
        'simple': custom_extractor(CNNSimple, 128),
        'mobile_net': custom_extractor(CNNMobileNet, 128),
        'cnn_lstm': custom_extractor(CNNLstm, 256),
        'large' : custom_extractor(CNNInreasedFilters, 512),
        'small' : custom_extractor(CNNDecreasedFilters, 128)
    }

    configue = load_configue(os.path.join(configues_dir, conf_name))
    ignore_warnings(configue['ignore_warnings'])
    logs_dir = configue['logs_dir']
    models_dir = configue['models_dir']
    check_path(logs_dir)
    check_path(models_dir)
    hyperparams = create_params(configue)

    
    for c, hp in enumerate(hyperparams):
        
        print(f'training {c+1}/{len(hyperparams)}:\n')
        
        meta = create_meta(configue, hp, method='grid')
        env = create_env(meta)
        model = create_model(env ,meta, logs_dir=logs_dir, policy_kwargs=cnn_architectures)
        print_model(meta)

        
        model_path = create_model_path(models_dir)
        check_path(model_path)
        save_meta(meta, model_path)


        model.learn(total_timesteps=meta['time_steps'], tb_log_name=f"{model_path}", progress_bar=True)
        model.save(os.path.join(model_path, f"{meta['time_steps']}"))
        if configue['domain_randomization']:
            print('Dynamics parameters:', env.get_parameters())
            
        del model
        print('__________________________________________________________________________________\n')




            
if __name__ == '__main__':
    main()
