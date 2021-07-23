# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:54:18 2020

@author: xxx
"""

from meta.experiment.Experiment import Experiment
from meta.train_ops.ClassificationMAMLTrainOP import ClassificationMAMLTrainOP

#dataloader related modules
from meta.data.StaticDataset import SinusoidStaticDataset, FullBatchSampler
from torch.utils.data.dataloader import DataLoader
from meta.data.CorrectImagenetDataset import ImagenetStaticDataset, ImagenetBatchSampler

from torch.nn import CrossEntropyLoss
from meta.meta_learners.ClassificationMetaModel import ConvMetaModelImagenet
from meta.utils.ClassificationExperimentLogger import ClassificationExperimentLogger
import meta.CONSTANTS as see #contains the global logger: see.logs

import matplotlib.pyplot as plt

import torch

import numpy as np
import argparse
import json
import copy
import os
import pickle



class ImagenetExperiment(Experiment):
    def __init__(self, args, config, train_op):
        self.args = args #argument parser object
        self.config = config #dict: contains configuration info
        self.train_op = train_op #train operation object
    
    def read_config(self, config):
        return config

    @staticmethod
    def update_args_from_config(args, config):
        args.no_of_classes = config['no_of_classes']
        args.no_of_tasks = config['no_of_tasks']
        args.datapoints_per_task_per_taskclass = config['datapoints_per_task_per_taskclass']
        return args

        
    def setup_logs(self, init=True):
        #re-write the global logger
        see.logs = ClassificationExperimentLogger(self.config)
        
        if init:
            see.logs.make_logdir()
    
            # save the training and configuration information
            see.logs.write(self.config, 'config.pickle')
            see.logs.write(self.args, 'args.pickle')
            
            see.logs.cache = {}
            see.logs.cache['train_tasks_hist_data'] = {}
            see.logs.cache['test_tasks_hist_data'] = {}
            see.logs.cache['train_avg_loss'] = []
            see.logs.cache['val_avg_loss'] = []
            see.logs.cache['lr_rates'] = []
            see.logs.cache['expected_meta_update_time'] = []
            see.logs.cache['total_meta_updates'] = []
            
            see.logs.cache['test_loss'] = []
            see.logs.cache['test_accuracy'] = []
            see.logs.cache['train_accuracy'] = []

        
        else:
            with open(os.path.join(see.logs.log_folder, 'log.pickle'), 'rb') as handle:
                see.logs.cache = pickle.load(handle)
        
    def run(self):
        #train for the current configuration
        ImagenetTrainingDataset = ImagenetStaticDataset(self.args.root_dir, 'train', no_of_tasks=self.args.no_of_tasks, classes_per_task=self.args.no_of_classes, no_of_data_points_per_task=self.args.datapoints_per_task_per_taskclass)
        ImagenetTrainingSampler = ImagenetBatchSampler(data_source = ImagenetTrainingDataset, no_of_tasks = 25, no_of_data_points_per_task = 50)

        TrainDataloader = DataLoader(ImagenetTrainingDataset, batch_sampler=ImagenetTrainingSampler, num_workers = self.args.num_workers)
        
        ImagenetValidationDataset = ImagenetStaticDataset(self.args.root_dir, 'val', no_of_tasks=500, classes_per_task=self.args.no_of_classes, no_of_data_points_per_task=10)
        ImagenetValidationSampler = ImagenetBatchSampler(data_source = ImagenetValidationDataset, no_of_tasks = None, no_of_data_points_per_task = None)
        ValDataloader = DataLoader(ImagenetValidationDataset, batch_sampler=ImagenetValidationSampler, num_workers = self.args.num_workers)

        self.train_op.train(TrainDataloader, ValDataloader) ## Do MetaTraining; validation dataset only used for early stopping
        see.logs.cache['train_accuracy'] = self.train_op.get_accuracy(TrainDataloader)

    def evaluate(self,):
        """this needs to be changed"""
        #--------------------------------------
        ImagenetTestDataset = ImagenetStaticDataset(self.args.root_dir, 'test', no_of_tasks = 500, classes_per_task=self.args.no_of_classes, no_of_data_points_per_task = 10)
        ImagenetTestSampler = ImagenetBatchSampler(data_source = ImagenetTestDataset, no_of_tasks = None, no_of_data_points_per_task = None)
        TestDataloader = DataLoader(ImagenetTestDataset, batch_sampler=ImagenetTestSampler, num_workers = self.args.num_workers)
        
        # Update the model in the train_op
        self.train_op.model = see.logs.load_model(checkpoint_index='best')
        self.train_op.model.eval()
        
        see.logs.cache['test_loss'] = self.train_op.mean_outer_loss(TestDataloader)
        see.logs.cache['test_accuracy'] = self.train_op.get_accuracy(TestDataloader)
        #update the log file
        see.logs.write(see.logs.cache, name='log.pickle')
        #--------------------------------------
        
if __name__ == '__main__':
    # 1.) set args
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    
    parser.add_argument('--dataset', type=str, default='Imagenet')
    
    parser.add_argument('--exp_config_dir', type=str, default='meta/experiment/config_experiment1.json',
            help='Directory of the configuration of the experiment.')

    parser.add_argument('--root-dir', type=str, default='meta/dataset//miniimagenet',
                         help='root directory folder')
    
    parser.add_argument('--train_test_split_inner', type=int, default=0.5,
            help='Train test split for the inner loop. Default: 0.3')
    
    parser.add_argument('--first-order', action='store_true',
            help='Use the first-order approximation of MAML.')
    
    parser.add_argument('--inner-step-size', type=float, default=0.01,
            help='Step-size for the inner loop. Default: 0.01')
    
    parser.add_argument('--train-adapt-steps', type=int, default=1, #5
            help='Number of inner gradient updates during training. Default: 1')
    
    parser.add_argument('--eval-adapt-steps', type=int, default=1, #10
            help='Number of inner gradient updates during evaluation. Default: 1')
    
    parser.add_argument('--meta_lr', type=float, default=0.001,
            help='The learning rate of the meta optimiser (outer loop lr). Default: 0.001')
    
    parser.add_argument('--hidden-size', type=int, default=32,
            help='Number of channels for each convolutional layer (default: 64).')
    
    parser.add_argument('--max-num-epochs', type=int, default=60000,
            help='Max Number of epochs')
    
    parser.add_argument('--num-workers', type=int, default=0,
            help='Number of workers for data loading (default: 0).')

    parser.add_argument('--num_classes', type=int, default=5,
            help='Number of classes')

    
    parser.add_argument('--use-cuda', type=int, default=1, help='For cuda set to 1. For cpu set to 0. Default: 1.')

    parser.add_argument('--budget', type=int, default=60000, help='budget')
    args = parser.parse_args()
    
    
    #args.DataloaderProcessing = DataloaderProcessing #data processing function for preparing the data for the meta_outer_loss method of MAMLTrainOP
    args.loss_func = CrossEntropyLoss() #multi-class classification
    
    if torch.cuda.is_available() and args.use_cuda == 1:
        args.device = torch.device('cuda')
        print('Device: ', args.device)
    elif not(torch.cuda.is_available()) and args.use_cuda == 1:
        args.device = torch.device('cpu')
        print('Cuda is not available. Using cpu instead.')
    else:
        args.device = torch.device('cpu')
        print('Device: ', args.device)
    
    
    budget = args.budget
    runs = 5
    no_of_classes = 5 #it stands for N_way
    model = ConvMetaModelImagenet(3, no_of_classes) #instantiate the base learner
    

    no_of_datapoints_per_taskclass_list = [2, 4, 6, 8, 10,12,14, 20, 50, 100, 200, 500, 1000]
    for datapoints_per_task_per_taskclass in no_of_datapoints_per_taskclass_list:
        for run in range(runs):
            no_of_tasks = int(round(budget/(datapoints_per_task_per_taskclass*no_of_classes)))
            config = {'dataset': 'Imagenet', 'ExperimentType': 'BaselineExperiment', 'budget': budget,
                      'no_of_classes': no_of_classes, 'no_of_tasks':no_of_tasks, 
                      'datapoints_per_task_per_taskclass':datapoints_per_task_per_taskclass, 'run':run, 'inner-step-size':args.inner_step_size}            
            print(config)
            updated_args = ImagenetExperiment.update_args_from_config(args, config)
          
            model_copy = type(model)(3, no_of_classes) # get a new instance
            model_copy.load_state_dict(model.state_dict()) # copy weights
            train_op = ClassificationMAMLTrainOP(model_copy, updated_args)
            exp = ImagenetExperiment(args, config, train_op)
            
            #---training----
            exp.setup_logs()
            exp.run()
            
            #---evaluation---
            exp.evaluate()
            
            
