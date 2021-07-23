# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:15:18 2020

@author: xxx (MAML algorithm implementation has been copied from 
                   https://github.com/tristandeleu/pytorch-meta/tree/master/examples/maml)
"""

#This file contains the MAML training operation.
#MAMLTrainOP is derived from the base abstract class TrainOP

#project modules
import meta.CONSTANTS as see #contains the global logger: see.logs
from meta.utils.stopping_criteria import StopByAnnealing

#torch modules
import torch
import torch.nn.functional as F
from torchmeta.modules import MetaModule
from collections import OrderedDict

#miscellaneous modules
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

import random

class ClassificationMAMLTrainOP:
    def __init__(self, model, args):
        self.train_config = None
        self.device = args.device
        
        self.inner_step_size = args.inner_step_size #lr for inner loop
        self.meta_lr = args.meta_lr #lr for outer loop
        self.first_order = args.first_order #indicates the use or not of first order approximations
        
        self.train_adapt_steps = args.train_adapt_steps #number of adaptations steps in train mode
        self.eval_adapt_steps = args.eval_adapt_steps #num adapt steps in eval mode
        self.no_of_classes = args.num_classes
        #self.K_shot = args.K_shot
        self.train_test_split_inner = args.train_test_split_inner
        
        self.max_num_epochs = args.max_num_epochs #upper limit to the number of meta-updates
        self.num_workers = args.num_workers 
        
        self.model = model.to(device=self.device)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        
        #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, 'min', factor=0.5, patience=150, verbose=True)
        self.lr_rates = []
        self.stopping_criterion = StopByAnnealing(4)
        
        self.loss_func = args.loss_func.to(device=self.device)
    
    def OOM_safe_model(self, tensor_input, params=None):
        oom_safe_batch_size = 100
        batch_size = tensor_input.shape[0]
        
        if batch_size <= oom_safe_batch_size:
            return self.model(tensor_input, params=params)
        else:
            #this should be used for the evaluation mode
            num_batches = batch_size//oom_safe_batch_size
            
            logits = torch.empty((batch_size, self.no_of_classes), device=self.device)
            for batch_index in range(num_batches):
                sub_batch = tensor_input[oom_safe_batch_size*batch_index:oom_safe_batch_size*(batch_index+1)]
                sub_logits = self.model(sub_batch, params=params)
                logits[oom_safe_batch_size*batch_index:oom_safe_batch_size*(batch_index+1)] = sub_logits
            
            if oom_safe_batch_size*(batch_index+1)<batch_size:
                last_sub_batch = tensor_input[oom_safe_batch_size*(batch_index+1):]
                last_sub_logits = self.model(last_sub_batch, params=params)
                logits[oom_safe_batch_size*(batch_index+1):] = last_sub_logits
            
            return logits
            
            
        
        
    def train_test_split(self, task_data, train_test_split_ratio = None, no_of_train_points = None):
        """
        :param task_data:
        :param train_test_split_ratio: a number between 0 and 1. If None, no_of_points is used instead.
        :param no_of_train_points: positive integer. Number of points for train (rest for test). If None
                    self.train_test_split_inner is used.
        :return: two arrays - test and train obtained from splitting data according to first argument unless specified,
                    then second arg. unless specified and then stored argument in the class instance (see above).
        """

        size_of_data = task_data.shape[0]
        if train_test_split_ratio is not None:
            no_of_train_points = int((1 - train_test_split_ratio) * size_of_data)
        elif no_of_train_points is None:
            no_of_train_points = int((1 - self.train_test_split_inner) * size_of_data)

        #random.shuffle(task_data)
        task_train_points = task_data[:no_of_train_points]
        task_test_points = task_data[no_of_train_points:]
        
        return task_train_points, task_test_points

    def train_test_input_target_split(self, task_data, task_target, train_test_split_ratio = None, no_of_train_points = None):
        """
        :param task_data:
        :param train_test_split_ratio: a number between 0 and 1. If None, no_of_points is used instead.
        :param no_of_train_points: positive integer. Number of points for train (rest for test). If None
                    self.train_test_split_inner is used.
        :return: two arrays - test and train obtained from splitting data according to first argument unless specified,
                    then second arg. unless specified and then stored argument in the class instance (see above).
        """

        size_of_data = task_data.shape[0]
        if train_test_split_ratio is not None:
            no_of_train_points = int((1 - train_test_split_ratio) * size_of_data)
        elif no_of_train_points is None:
            no_of_train_points = int((1 - self.train_test_split_inner) * size_of_data)

        indices = np.arange(size_of_data)
        # Shuffling the data may be wrong for meta-learning?
        #np.random.shuffle(indices)

        task_train_data, task_test_data = task_data[indices][:no_of_train_points], task_data[indices][no_of_train_points:]
        task_train_target, task_test_target = task_target[indices][:no_of_train_points], task_target[indices][no_of_train_points:]
        return task_train_data, task_test_data, task_train_target, task_test_target
    
    def adaptation(self, train_input, train_target, return_grad = False):
        """Perform parameter adaptation using train input and train target"""
        # Inputs: 1.) train_input:  torch tensor of size: (Kin,) + input_size
        #         2.) train_target: torch tensor of size (Kin,) + output_size
        
        # Outputs: 1.) adapted parameters of the MetaModel
        
        if self.model.training:
            num_adapt_steps = self.train_adapt_steps
        else:
            num_adapt_steps = self.eval_adapt_steps
        
        params = None  
        for step in range(num_adapt_steps):
            train_logit = self.OOM_safe_model(train_input, params=params)
            inner_loss = self.loss_func(train_logit, train_target)
            
            self.model.zero_grad()
            
            if not isinstance(self.model, MetaModule):
                raise ValueError('The model must be an instance of `torchmeta.modules.'
                                 'MetaModule`, got `{0}`'.format(type(self.model)))
        
            if params is None:
                params = OrderedDict(self.model.meta_named_parameters())
        
            grads = torch.autograd.grad(inner_loss,
                                        params.values(),
                                        create_graph = not self.first_order)
        
            updated_params = OrderedDict()
        
            if isinstance(self.inner_step_size, (dict, OrderedDict)):
                for (name, param), grad in zip(params.items(), grads):
                    updated_params[name] = param - self.inner_step_size[name] * grad
        
            else:
                for (name, param), grad in zip(params.items(), grads):
                    updated_params[name] = param - self.inner_step_size * grad
            
            params = updated_params
            
        if return_grad:
            return params, grads
        else:
            return params
    
    def get_accuracy(self, dataloader, train_test_split_ratio=None, no_of_adaptation_points=None):
        mean_task_accuracy = 0
        for task_idx, (task_data, task_labels) in enumerate(dataloader):
            #train_input, train_target, test_input, test_target = self.DataloaderProcessing(task_data, self.N_way, self.K_shot)
            #train_input, test_input = self.train_test_split(task_data, train_test_split_ratio, no_of_adaptation_points)
            #train_target, test_target = self.train_test_split(task_labels, train_test_split_ratio, no_of_adaptation_points)
            train_input, test_input, train_target, test_target = self.train_test_input_target_split(task_data, task_labels, train_test_split_ratio, no_of_adaptation_points)
            train_input = train_input.to(device=self.device)
            train_target = train_target.to(device=self.device)
            test_input = test_input.to(device=self.device)
            test_target = test_target.to(device=self.device)
        
            """Adaptation"""
            params = self.adaptation(train_input, train_target)
            
            correct, total = 0, 0
            """Evaluation"""
            with torch.set_grad_enabled(False):
                test_logit = self.OOM_safe_model(test_input, params=params)
                pred = test_logit.max(1, keepdim=True)[1] # get the index of the max logit
                correct += pred.eq(test_target.view_as(pred)).sum().item()
                total += int(test_target.shape[0])
            
            task_accuracy = correct / total
            mean_task_accuracy += task_accuracy
        
        mean_task_accuracy /= task_idx + 1
        
        return mean_task_accuracy

    
    def compute_moments(self, dataloader, permutations = 10, train_test_split_ratio=None, no_of_adaptation_points=None):
        empirical_mean = torch.tensor(0., device=self.device)
        #empirical_variance = torch.tensor(0., device=self.device)
        for task_idx, (task_data, task_labels) in enumerate(dataloader):
            empirical_losses = torch.zeros(permutations)
            for i in range(permutations):
                #torch.manual_seed(i)
                permuted_indices = torch.randperm(task_data.shape[0])
                permuted_task_data = task_data[permuted_indices]
                permuted_task_labels = task_labels[permuted_indices]
                
                permuted_train_input, permuted_test_input = self.train_test_split(permuted_task_data, train_test_split_ratio, no_of_adaptation_points)
                permuted_train_target, permuted_test_target = self.train_test_split(permuted_task_labels, train_test_split_ratio, no_of_adaptation_points)
                
                permuted_train_input = permuted_train_input.to(device=self.device)
                permuted_test_input = permuted_test_input.to(device=self.device)
                permuted_train_target = permuted_train_target.to(device=self.device)
                permuted_test_target = permuted_test_target.to(device=self.device)

                params = self.adaptation(permuted_train_input, permuted_train_target, return_grad=False)
                with torch.set_grad_enabled(False):
                    permuted_test_logit = self.OOM_safe_model(permuted_test_input, params=params)
                    permuted_eval_loss = self.loss_func(permuted_test_logit, permuted_test_target)
                
                empirical_losses[i] = permuted_eval_loss
                
            empirical_mean += torch.mean(empirical_losses)
            #empirical_variance += torch.var(empirical_losses)
            
        N = task_idx + 1
        empirical_mean /= N
        #empirical_variance /= N
        #empirical_std = torch.sqrt(empirical_variance)

        return empirical_mean
    
    def mean_outer_loss(self, dataloader, train_test_split_ratio = None, no_of_adaptation_points = None):
        """
        :param dataloader: torch.utils.data.Dataloader
        :param train_test_split_ratio: a number between 0 and 1 specifying the ratio. If None no_of_adaptation_points is used
        :param no_of_adaptation_points: a pos. integer. specifying the number of adaptation points to use.
                If None, self.train_test_split_inner is used for splitting.
        :return: mean_outer_loss from adaptaion, where mean loss is an average over all test points and over the whole dataloader
                iteration
        """

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_idx, (task_data, task_labels) in enumerate(dataloader):
            #train_input, train_target, test_input, test_target = self.DataloaderProcessing(task_data, self.N_way, self.K_shot)
            #train_input, test_input = self.train_test_split(task_data, train_test_split_ratio, no_of_adaptation_points)
            #train_target, test_target = self.train_test_split(task_labels, train_test_split_ratio, no_of_adaptation_points)
            train_input, test_input, train_target, test_target = self.train_test_input_target_split(task_data, task_labels, train_test_split_ratio, no_of_adaptation_points)
            
            train_input = train_input.to(device=self.device)
            train_target = train_target.to(device=self.device)
            test_input = test_input.to(device=self.device)
            test_target = test_target.to(device=self.device)
        
            """Adaptation"""
            params = self.adaptation(train_input, train_target)
            
            """Evaluation"""
            with torch.set_grad_enabled(self.model.training):
                test_logit = self.OOM_safe_model(test_input, params=params)
                task_loss = self.loss_func(test_logit, test_target)
                mean_outer_loss += task_loss
                
        
        mean_outer_loss /= task_idx+1
        return mean_outer_loss
        
    
    def training_iteration(self, dataloader):
        """Training iteration - 1 gradient update"""
        # Inputs: batch
        
        self.model.zero_grad() 
        batch_avg_loss = self.mean_outer_loss(dataloader)
        
        """meta_update"""
        batch_avg_loss.backward()
        self.meta_optimizer.step()

        return batch_avg_loss.item()
        
    
    def training_phase(self, dataloader, num_of_training_iterations = 1):
        self.model.train() 
        
        train_avg_loss = 0
        for i in range(num_of_training_iterations):
            batch_avg_loss = self.training_iteration(dataloader)
            train_avg_loss += batch_avg_loss
        
        train_avg_loss /= i+1
        see.logs.update_cache(train_avg_loss, 'train_avg_loss')
        
        return train_avg_loss
    
    def _check_lr_annealing(self, epoch):
        # criterion relying on the monitoring the annealing of the lr rate
        lr = self.meta_optimizer.param_groups[0]['lr']
        see.logs.update_cache(lr, 'lr_rates')

        if epoch > 30:
            stop_signal = self.stopping_criterion(see.logs.cache['lr_rates'])
            return stop_signal
        else:
            return False

    def _check_max_epoch(self, epoch):
        # criterion that check we have not reached max allowed num of epochs
        if epoch > self.max_num_epochs:
            return True
        else:
            return False

    def _check_val_loss_improvement(self, epoch, checkpoints_epochs):
        # criterion that checks whether the validation loss has improved within the last 100 epochs
        if epoch - checkpoints_epochs[-1] > 2000:
            return True
        else:
            return False


    def stop_training(self, epoch, checkpoints_epochs):
        #The stopping criterion consists of the seperate check of three criteria
        #If any of the three criteria is satisfied, the training stops
        
        return self._check_lr_annealing(epoch) or \
               self._check_max_epoch(epoch) or \
               self._check_val_loss_improvement(epoch, checkpoints_epochs)
    
    def train(self, train_dataloader, val_dataloader, lr_scheduler=None, extra_dataloaders = None):
        #set the learning rate scheduler
        if lr_scheduler==None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, 'max', factor=0.5, patience=10, verbose=True)
            
        #-----------------
        start_time = datetime.datetime.now()
        def chop_microseconds(delta):
            #utility to help avoid printing the microseconds
            return delta - datetime.timedelta(microseconds=delta.microseconds)
        #-----------------
        
        #clean saved model checkpoints of previous runs with the same configuration
        see.logs.clean_model_checkpoints(-1)

        meta_update_times = []
        checkpoints_epochs = [0]
        epoch=1
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        while self.stop_training(epoch, checkpoints_epochs)==False:
            # Trainining
            
            start=time.time()
                
            train_avg_loss = self.training_phase(train_dataloader)
            
            #print(train_avg_loss)
            train_losses.append(train_avg_loss)
            
            end=time.time()
            time_diff = end-start
            meta_update_times.append(time_diff)
            
            
            
            #evaluate on the validation set
            if epoch % 50 == 0:
                if train_dataloader.dataset.infiniteTask:
                    train_dataloader.dataset.reinitialize() # this leads to infinite tasks in the dataset
                self.model.eval()
                val_avg_loss = self.mean_outer_loss(val_dataloader)
                val_losses.append(val_avg_loss)
                see.logs.update_cache(val_avg_loss, 'val_avg_loss')               

                mean_task_accuracy = self.get_accuracy(val_dataloader)
                val_accuracies.append(mean_task_accuracy)

                extra_losses=[]
                extra_accuracies=[]

                if extra_dataloaders:
                    for dataloader in extra_dataloaders:
                        extra_losses.append(self.mean_outer_loss(dataloader))
                        extra_accuracies.append(self.get_accuracy(dataloader))

                # LR scheduling
                lr_scheduler.step(mean_task_accuracy)

                elapsed_time = chop_microseconds(datetime.datetime.now() - start_time)
                print('[time:{}][{}/{}] - train_loss: {:.3f} - val_loss: {:.3f} - val_accuracy: {:.3f}, extra_losses: {}, extra_accuracies: {}, '.format(
                    elapsed_time, 
                    epoch, 
                    self.max_num_epochs, 
                    train_avg_loss, 
                    val_avg_loss, 
                    mean_task_accuracy,
                    extra_losses,
                    extra_accuracies
                    )) 
                
                #save the model if it achieves the best performance on the validation set
                if val_avg_loss == min(see.logs.cache['val_avg_loss']):
                    checkpoints_epochs.append(epoch)
                    see.logs.save_model(self.model, checkpoint_index=epoch)
                    # clean the model directory to avoid capturing too much memory
                    if len(checkpoints_epochs) > 3:
                        see.logs.clean_model_checkpoints(checkpoints_epochs[-1])
            
            
            if epoch % 500 == -1:
                plt.figure()
                plt.title('losses')
                plt.plot(train_losses, label = 'train')
                plt.plot([50*x+50 for x in range(len(val_losses))], val_losses, label = 'val')
                plt.legend()
                plt.savefig('losses.png')
                
                plt.figure()
                plt.title('Validation Accuracy')
                plt.plot([50*x+50 for x in range(len(val_accuracies))], val_accuracies)
                plt.savefig('validation_accuracy.png')
                
                see.logs.write(see.logs.cache, name='log.pickle')
                plt.close('all')
            
                
            # Data Allocation phase 
            # To be completed by Alex
            
            epoch+=1
        
        print('Convergence epoch: %d -- train_loss:%.4f - val_loss:%.4f' % (epoch, train_avg_loss, val_avg_loss))
        
        #keep only the best checkpoint for evaluation
        see.logs.clean_model_checkpoints(checkpoints_epochs[-1])
        
        # save the number of meta-updates to convergence
        see.logs.update_cache(epoch, 'total_meta_updates')
        
        # save the mean time of a meta-update
        see.logs.update_cache(np.mean(meta_update_times), 'expected_meta_update_time')
        
        # Save logs
        see.logs.write(see.logs.cache, name='log.pickle')
        
        # Save model weights at the end of training
        see.logs.save_model(self.model, checkpoint_index=epoch)

