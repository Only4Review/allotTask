import sys
# work_dir = '/proj/gpu_xxx/allotMeta2/salsa'
# sys.path.insert(0,work_dir)

from meta.experiment.CifarExperiment import CifarExperiment
from meta.train_ops.ClassificationMAMLTrainOP import ClassificationMAMLTrainOP

#dataloader related modules
from meta.data.StaticDataset import SinusoidStaticDataset, FullBatchSampler
from torch.utils.data.dataloader import DataLoader
from meta.data.CorrectCifarDataset import CifarStaticDatasetHierarchy, CifarBatchSampler

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



class CifarHierarchicalExperiment(CifarExperiment):
    def __init__(self, *args, **kwargs):
        super(CifarHierarchicalExperiment, self).__init__(*args, **kwargs)
        
    def run(self):
        see.logs.write_to_log_file(json.dumps(self.config))

        #train for the current configuration
        CifarTrainingDataset = CifarStaticDatasetHierarchy(
                'Train', 
                self.args.hierarchy_json, 
                no_of_easy=self.args.num_easy, 
                no_of_hard=self.args.num_hard,
                classes_per_task=self.args.no_of_classes, 
                no_data_points_hard = self.args.num_datapoints_per_class_hard,
                no_data_points_easy = self.args.num_datapoints_per_class_easy,
                )
        CifarTrainingSampler = CifarBatchSampler(data_source = CifarTrainingDataset, no_of_tasks = 25, no_of_data_points_per_task = 100)
        TrainDataloader = DataLoader(CifarTrainingDataset, batch_sampler=CifarTrainingSampler, num_workers = self.args.num_workers)

        NUM_VAL_TASKS = 500
        num_easy_val = NUM_VAL_TASKS
        num_hard_val = NUM_VAL_TASKS

        NUM_VAL_DTPTSK = 20
        num_easy_val_data = self.args.num_datapoints_per_class_easy
        num_hard_val_data = self.args.num_datapoints_per_class_hard

        CifarValidationDataset = CifarStaticDatasetHierarchy(
                'Train', 
                self.args.hierarchy_json,
                no_of_easy=num_easy_val, 
                no_of_hard=num_hard_val,
                classes_per_task=self.args.no_of_classes, 
                no_data_points_hard = num_hard_val_data,
                no_data_points_easy = num_easy_val_data,
                ) 

        CifarValidationSampler = CifarBatchSampler(data_source = CifarValidationDataset, no_of_tasks = None, no_of_data_points_per_task = None)
        ValDataloader = DataLoader(CifarValidationDataset, batch_sampler=CifarValidationSampler, num_workers = self.args.num_workers)

        self.train_op.train(TrainDataloader, ValDataloader)
        see.logs.cache['train_avg_accuracy'] = self.train_op.get_accuracy(TrainDataloader)
        
    def evaluate(self):
        """this needs to be changed"""
        #--------------------------------------

        NUM_TEST_TASKS = 1000
        num_easy_test = NUM_TEST_TASKS
        num_hard_test = NUM_TEST_TASKS

        num_easy_test_data = self.args.num_datapoints_per_class_easy 
        num_hard_test_data = self.args.num_datapoints_per_class_hard

        CifarTestDataset = CifarStaticDatasetHierarchy(
                'Test', 
                self.args.hierarchy_json,
                no_of_easy=num_easy_test, 
                no_of_hard=num_hard_test,
                classes_per_task=self.args.no_of_classes, 
                no_data_points_hard = num_hard_test_data,
                no_data_points_easy = num_easy_test_data,
                )

        CifarTestSampler = CifarBatchSampler(data_source = CifarTestDataset, no_of_tasks = None, no_of_data_points_per_task = None)
        TestDataloader = DataLoader(CifarTestDataset, batch_sampler=CifarTestSampler, num_workers = self.args.num_workers)
        
        # Update the model in the train_op
        self.train_op.model = see.logs.load_model(checkpoint_index='best')
        self.train_op.model.eval()
        
        see.logs.cache['test_loss'] = self.train_op.mean_outer_loss(TestDataloader)
        test_accuracy = self.train_op.get_accuracy(TestDataloader)
        see.logs.cache['test_accuracy'] = test_accuracy
        print('test_accuracy={}'.format(test_accuracy))
        see.logs.write_to_log_file('test_accuracy={}'.format(test_accuracy))
        
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

    parser.add_argument('--hierarchy_json', type=str, default='meta/dataset/cifar100/hierarchy_3_hiperclass.json',
                        help='root directory folder')
    
    parser.add_argument('--train_test_split_inner', type=int, default=0.5,
            help='Train test split for the inner loop. Default: 0.3')
    
    parser.add_argument('--first-order', action='store_true',
            help='Use the first-order approximation of MAML.')
    
    parser.add_argument('--inner-step-size', type=float, default=0.01,
            help='Step-size for the inner loop. Default: 0.01')
    
    parser.add_argument('--train-adapt-steps', type=int, default=1,
            help='Number of inner gradient updates during training. Default: 1')
    
    parser.add_argument('--eval-adapt-steps', type=int, default=1,
            help='Number of inner gradient updates during evaluation. Default: 1')
    
    parser.add_argument('--meta_lr', type=float, default=0.001,
            help='The learning rate of the meta optimiser (outer loop lr). Default: 0.001')
    
    parser.add_argument('--hidden-size', type=int, default=64,
            help='Number of channels for each convolutional layer (default: 64).')
    
    parser.add_argument('--max-num-epochs', type=int, default=60000,
            help='Max Number of epochs')

    parser.add_argument('--num_classes', type=int, default=5,
            help='Number of classes')
    
    parser.add_argument('--num_easy', type=int, default=500,
            help='Number of easy tasks')

    parser.add_argument('--num_hard', type=int, default=500,
            help='Number of hard tasks')

    parser.add_argument('--num_datapoints_per_class_easy', type=int, default=4,
            help='Number of data points per class for easy tasks')

    parser.add_argument('--num_datapoints_per_class_hard', type=int, default=4,
            help='Number of data points per class for hard tasks')
    
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
    
    
    runs = 5
    no_of_classes = args.num_classes #it stands for N_way
    model = ConvMetaModel(3, no_of_classes) #instantiate the base learner
    
    budget = no_of_classes * (args.num_easy * args.num_datapoints_per_class_easy + args.num_hard * args.num_datapoints_per_class_hard)

    for run in range(runs):
        config = {
                'dataset': 'Cifar', 
                'ExperimentType': 'BaselineExperiment', 
                'budget': budget,
                'no_of_classes': no_of_classes, 
                'no_of_tasks': args.num_easy + args.num_hard,
                'datapoints_per_task_per_taskclass': args.num_datapoints_per_class_easy,
                'no_of_easy_tasks': args.num_easy,
                'no_of_hard_tasks': args.num_hard,
                'no_of_datapoints_per_easy_tasks': args.num_datapoints_per_class_easy,
                'no_of_datapoints_per_hard_tasks': args.num_datapoints_per_class_hard,
                'run':run
        }
        
        print(config)
        
        updated_args = CifarHierarchicalExperiment.update_args_from_config(args, config)
        
        model_copy = type(model)(3, no_of_classes) # get a new instance
        model_copy.load_state_dict(model.state_dict()) # copy weights
        train_op = ClassificationMAMLTrainOP(model_copy, updated_args)
        exp = CifarHierarchicalExperiment(args, config, train_op)
        
        #---training----
        exp.setup_logs()
        exp.run()
        
        #---evaluation---
        #exp.setup_logs(init=False)
        exp.evaluate()
        
    
