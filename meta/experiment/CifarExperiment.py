import sys

from meta.experiment.Experiment import Experiment
from meta.train_ops.ClassificationMAMLTrainOP import ClassificationMAMLTrainOP

#dataloader related modules
from meta.data.StaticDataset import SinusoidStaticDataset, FullBatchSampler
from torch.utils.data.dataloader import DataLoader
from meta.data.CorrectCifarDataset import CifarStaticDataset, CifarBatchSampler

from torch.nn import CrossEntropyLoss
from meta.meta_learners.ClassificationMetaModel import ConvMetaModel
from meta.utils.ClassificationExperimentLogger import ClassificationExperimentLogger
import meta.CONSTANTS as see #contains the global logger: see.logs

import torch

import numpy as np
import argparse
import json
import copy
import os
import pickle



class CifarExperiment(Experiment):
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
            see.logs.cache['train_avg_accuracy'] =[]
            see.logs.cache['train_avg_loss'] = []
            see.logs.cache['val_avg_loss'] = []
            see.logs.cache['lr_rates'] = []
            see.logs.cache['expected_meta_update_time'] = []
            see.logs.cache['total_meta_updates'] = []
            
            see.logs.cache['test_loss'] = []
            see.logs.cache['test_accuracy'] = []
        
        else:
            with open(os.path.join(see.logs.log_folder, 'log.pickle'), 'rb') as handle:
                see.logs.cache = pickle.load(handle)
        
    def run(self):
        #train for the current configuration
        CifarTrainingDataset = CifarStaticDataset(self.args.root_dir, 'train', no_of_tasks=self.args.no_of_tasks, classes_per_task=self.args.no_of_classes, no_of_data_points_per_task=self.args.datapoints_per_task_per_taskclass)
        CifarTrainingSampler = CifarBatchSampler(data_source = CifarTrainingDataset, no_of_tasks = 25, no_of_data_points_per_task = 100)
        TrainDataloader = DataLoader(CifarTrainingDataset, batch_sampler=CifarTrainingSampler, num_workers = self.args.num_workers)
        
        CifarValidationDataset = CifarStaticDataset(self.args.root_dir, 'val', no_of_tasks=500, classes_per_task=self.args.no_of_classes, no_of_data_points_per_task=10)
        CifarValidationSampler = CifarBatchSampler(data_source = CifarValidationDataset, no_of_tasks = None, no_of_data_points_per_task = None)
        ValDataloader = DataLoader(CifarValidationDataset, batch_sampler=CifarValidationSampler, num_workers = self.args.num_workers)

        self.train_op.train(TrainDataloader, ValDataloader)
        see.logs.cache['train_avg_accuracy'] = self.train_op.get_accuracy(TrainDataloader)
    def evaluate(self,):
        """this needs to be changed"""
        #--------------------------------------
        CifarTestDataset = CifarStaticDataset(self.args.root_dir, 'test', no_of_tasks = 1000, classes_per_task=self.args.no_of_classes, no_of_data_points_per_task = 10)
        CifarTestSampler = CifarBatchSampler(data_source = CifarTestDataset, no_of_tasks = None, no_of_data_points_per_task = None)
        TestDataloader = DataLoader(CifarTestDataset, batch_sampler=CifarTestSampler, num_workers = self.args.num_workers)
        
        # Update the model in the train_op
        self.train_op.model = see.logs.load_model(checkpoint_index='best')
        self.train_op.model.eval()
        
        see.logs.cache['test_loss'] = self.train_op.mean_outer_loss(TestDataloader)
        test_accuracy = self.train_op.get_accuracy(TestDataloader)
        see.logs.cache['test_accuracy'] = test_accuracy
        print('test_accuracy={}'.format(test_accuracy))
        
        #update the log file
        see.logs.write(see.logs.cache, name='log.pickle')
        #--------------------------------------
        
if __name__ == '__main__':
    # 1.) set args
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    
    parser.add_argument('--dataset', type=str, default='cifar-fs')
    
    parser.add_argument('--exp_config_dir', type=str, default='meta/experiment/config_experiment1.json',
            help='Directory of the configuration of the experiment.')

    parser.add_argument('--root-dir', type=str, default='meta/dataset/cifar100',
                        help='root directory folder')
    
    parser.add_argument('--train_test_split_inner', type=int, default=0.5,
            help='Train test split for the inner loop. Default: 0.3')
    
    parser.add_argument('--first-order', action='store_true',
            help='Use the first-order approximation of MAML.')
    
    parser.add_argument('--inner-step-size', type=float, default=0.01,
            help='Step-size for the inner loop. Default: 0.01')
    
    parser.add_argument('--train-adapt-steps', type=int, default=1,
            help='Number of inner gradient updates during training. Default: 1')
    
    parser.add_argument('--eval-adapt-steps', type=int, default=5,
            help='Number of inner gradient updates during evaluation. Default: 5')
    
    parser.add_argument('--meta_lr', type=float, default=0.001,
            help='The learning rate of the meta optimiser (outer loop lr). Default: 0.001')
    
    parser.add_argument('--hidden-size', type=int, default=64,
            help='Number of channels for each convolutional layer (default: 64).')
    
    parser.add_argument('--max-num-epochs', type=int, default=60000,
            help='Max Number of epochs')
    
    parser.add_argument('--budget', type=int, default=60000,
            help='Budget')

    parser.add_argument('--num_classes', type=int, default=5,
            help='Number of classes')
    
    parser.add_argument('--num_tasks', type=int, default=100,
            help='Number of tasks')

    parser.add_argument('--num_datapoints_per_class', type=int, default=4,
            help='Number of data points')
    
    parser.add_argument('--num-workers', type=int, default=8,
            help='Number of workers for data loading (default: 8).')
    
    parser.add_argument('--use-cuda', type=int, default=1, help='For cuda set to 1. For cpu set to 0. Default: 1.')
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
    no_of_classes = args.num_classes #it stands for N_way
    model = ConvMetaModel(3, no_of_classes) #instantiate the base learner
    

    #no_of_datapoints_per_taskclass_list = [200]
    no_of_datapoints_per_taskclass_list = [args.num_datapoints_per_class]    
    for datapoints_per_task_per_taskclass in no_of_datapoints_per_taskclass_list:
        for run in range(runs):
            #no_of_tasks = -1
            no_of_tasks = int(round(budget/(datapoints_per_task_per_taskclass*no_of_classes)))
            if no_of_tasks < 1:
                break
            config = {'dataset': 'Cifar', 'ExperimentType': 'BaselineExperiment', 'budget': budget,
	              'no_of_classes': no_of_classes, 'no_of_tasks':no_of_tasks, 
	              'datapoints_per_task_per_taskclass':datapoints_per_task_per_taskclass, 'run':run}
            
            print(config)
            
            updated_args = CifarExperiment.update_args_from_config(args, config)
            
            model_copy = type(model)(3, no_of_classes) # get a new instance
            model_copy.load_state_dict(model.state_dict()) # copy weights
            train_op = ClassificationMAMLTrainOP(model_copy, updated_args)
            exp = CifarExperiment(args, config, train_op)
            
            #---training----
            exp.setup_logs()
            exp.run()
            
            #---evaluation---
            #exp.setup_logs(init=False)
            exp.evaluate()
        
    
