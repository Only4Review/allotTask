import sys
from meta.experiment.Experiment import Experiment
from meta.train_ops.MAMLTrainOp import MAMLTrainOp

#dataloader related modules
from meta.data.StaticDataset import SinusoidStaticDataset, FullBatchSampler
from torch.utils.data.dataloader import DataLoader

from meta.meta_learners.RegressionMetaModel import MetaModel
from meta.utils.SinusoidExperimentLogger import SinusoidExperimentLogger
import meta.CONSTANTS as see #contains the global logger: see.logs

import torch

import numpy as np
import argparse
import json
import copy
import os
import pickle



class SinusoidExperiment(Experiment):
    def __init__(self, experimentID, args, config, dataset_params, train_op):
        self.experimentID = experimentID
        self.args = args #argument parser object
        self.config = config #dict: contains configuration info
        self.dataset_params = dataset_params #dict: contains dataset parameters
        self.train_op = train_op #train operation object
    
    def read_config(self, config):
        return config

    @staticmethod
    def update_args_from_config(args, config):

        B = config['Budget']
        p = config['p']

        args.N = int(1 / p)
        args.K = int(B / args.N)
        args.Kin = config['Kin']
        args.Nout = config['Nout']
        args.train_test_split_inner = config['train_test_split']
        
        return args

        
    def setup_logs(self, init=True):
        #re-write the global logger
        see.logs = SinusoidExperimentLogger(dataset='sinusoid', ID=self.experimentID, config_info=self.config)
        
        if init:
            see.logs.make_logdir()
    
            # save the training and configuration information
            see.logs.write(self.config, 'config.pickle')
            see.logs.write(self.args, 'args.pickle')
            
            see.logs.cache = {}
            see.logs.cache['train_tasks_hist_data'] = {}
            see.logs.cache['test_tasks_hist_data'] = {}
            see.logs.cache['train_avg_loss'] = []
            see.logs.cache['test_avg_loss'] = {}
            see.logs.cache['val_avg_loss'] = []
            see.logs.cache['lr_rates'] = []
            see.logs.cache['expected_meta_update_time'] = []
            see.logs.cache['total_meta_updates'] = []
        
        else:
            if os.path.exists(os.path.join(see.logs.log_folder, 'log.pickle'))==False:
                return False
            else:
                with open(os.path.join(see.logs.log_folder, 'log.pickle'), 'rb') as handle:
                    see.logs.cache = pickle.load(handle)
                return True
        
    def run(self):
        #train for the current configuration
        
        #create train and validation dataloader
        self.train_datasource = SinusoidStaticDataset(self.args.Nout, self.args.Kin, **self.dataset_params)
        self.train_sampler = FullBatchSampler(self.train_datasource, no_of_tasks = self.args.Nout, no_of_points_per_task = self.args.Kin)
        self.train_dataloader = DataLoader(self.train_datasource, batch_sampler = self.train_sampler, num_workers = self.args.num_workers)
        
        self.val_datasource = SinusoidStaticDataset(100, self.args.Kin, **self.dataset_params)
        self.val_sampler = FullBatchSampler(self.val_datasource)
        self.val_dataloader = DataLoader(self.val_datasource, batch_sampler = self.val_sampler, num_workers = self.args.num_workers)

        self.train_op.train(self.train_dataloader, self.val_dataloader)

    def evaluate(self,):
        train_adapt_pts = 500
        
        adapt_pts_list = []
        adapt_pts_list.append(train_adapt_pts)
        
        test_datasource = SinusoidStaticDataset(1000, 2*max(adapt_pts_list), **args.dataset_params)
        test_sampler = FullBatchSampler(test_datasource)
        test_dataloader = DataLoader(test_datasource, batch_sampler = test_sampler, num_workers = args.num_workers)

        # Update the model in the train_op
        self.train_op.model = see.logs.load_model(checkpoint_index='best')
        self.train_op.model.eval()
        for adapt_pts in adapt_pts_list:
            see.logs.cache['test_avg_loss'][adapt_pts] = self.train_op.mean_outer_loss(test_dataloader, no_of_adaptation_points = adapt_pts)
            print('%d-shot loss: %.4f' % (adapt_pts, see.logs.cache['test_avg_loss'][adapt_pts]))
            
        #update the log file
        see.logs.write(see.logs.cache, name='log.pickle')

def data_and_labels_sinusoid_splitter(data):
    assert(data.shape[1] == 2)
    return data[:, 0].unsqueeze(-1), data[:, 1].unsqueeze(-1)
        
if __name__ == '__main__':
    # 1.) set args
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    #parser.add_argument('folder', type=str,
    #        help='Path to the folder the data is downloaded to.')
    
    parser.add_argument('--exp_config_dir', type=str, default='meta/experiment/config_experiment1.json',
            help='Directory of the configuration of the experiment.')

    parser.add_argument('--data-root', type=str, default='meta/data/',
                        help='data folder')
    
    parser.add_argument('--K', type=int, default=15,
            help='Number of datapoints per task')
    
    parser.add_argument('--Kin', type=int, default=15,
            help='Inner loop batch size')
    
    parser.add_argument('--N', type=int, default=150,
            help='Number of tasks.')
    
    parser.add_argument('--Nout', type=int, default=25,
            help='Outer loop batch size')
    
    parser.add_argument('--train_test_split_inner', type=int, default=0.5,
            help='Train test split for the inner loop. Default: 0.5')
    
    parser.add_argument('--first-order', action='store_true',
            help='Use the first-order approximation of MAML.')
    
    parser.add_argument('--inner-step-size', type=float, default=0.01,
            help='Step-size for the inner loop. Default: 0.01')
    
    parser.add_argument('--num_adaptation_steps', type=int, default=1,
            help='Number of inner gradient updates during training. Default: 1')
    
    parser.add_argument('--meta_lr', type=float, default=0.001,
            help='The learning rate of the meta optimiser (outer loop lr). Default: 0.001')
    
    parser.add_argument('--hidden-size', type=int, default=40,
            help='Number of channels for each convolutional layer (default: 40).')
    
    parser.add_argument('--output-folder', type=str, default=None,
            help='Path to the output folder for saving the model (optional).')
    
    parser.add_argument('--max-num-epochs', type=int, default=5000,
            help='Max Number of epochs')
    
    parser.add_argument('--num-workers', type=int, default=0,
            help='Number of workers for data loading (default: 0).')
    
    parser.add_argument('--budget', type=int, default=60000,
            help='Budget')

    parser.add_argument('--use-cuda', action='store_true',
            help='Use CUDA if available.')
    
    args = parser.parse_args()
    
    args.data_and_labels_splitter = data_and_labels_sinusoid_splitter
    args.device = torch.device('cuda')

    # 2.) create the dataset 
    amplitude_range = (0.1, 5)
    phase_range = (0, np.pi)
    noise_range = (0, 0)
    x_range = (-5, 5)
    
    dataset_params = {'amplitude_range': amplitude_range,
                      'phase_range': phase_range,
                      'noise_std_range': noise_range,
                      'x_range': x_range}
    
    args.dataset_params = dataset_params
    
    # 3.) create the model
    model = MetaModel()

    # 5.) create the configurations of the experiment
    #with open(args.exp_config_dir) as f:
    #   configurations = json.load(f)
    
    experimentID = 5
    
    budget = args.budget
    runs = 5
   
    #training routine
    datapoints_per_task_list = [100,150,200,500,1000,2000,5000]
    #datapoints_per_task_list = [2000, 1000, 500, 200, 150, 100, 50]
    #datapoints_per_task_list = [budget]
    for datapoints_per_task in datapoints_per_task_list:
        for run in range(runs):
            no_of_tasks = int(round(budget/datapoints_per_task))
            if no_of_tasks < 1:
                break
            config = {'Budget': budget, 'Nout':no_of_tasks, 'train_test_split':args.train_test_split_inner,
                      'Kin':datapoints_per_task, 'p':1/no_of_tasks,'run':run}
            print(config)        
            updated_args = SinusoidExperiment.update_args_from_config(args, config)
            # 4.) Select the TrainOP
            model_copy = type(model)() # get a new instance
            model_copy.load_state_dict(model.state_dict()) # copy weights and stuff
            train_op = MAMLTrainOp(model_copy, updated_args)
            exp = SinusoidExperiment(experimentID, updated_args, config, dataset_params, train_op)
            exp.setup_logs()
            exp.run()
            exp.evaluate()
            del model_copy

    
