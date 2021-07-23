import meta.CONSTANTS as see
from meta.utils.stopping_criteria import StopByAnnealing

import torch
import torch.nn.functional as F
from torchmeta.utils.gradient_based import gradient_update_parameters

import numpy as np
import time


class MAMLTrainOp:
    def __init__(self, model, args):

        # TODO: This is opaque. Refactor this
        self.train_config = None
        
        self.device = args.device
        
        self.data_root = args.data_root
        
        self.inner_step_size = args.inner_step_size #lr for inner loop
        self.meta_lr = args.meta_lr #lr for outer loop
        
        self.first_order = args.first_order #indicates the use or not of first order approximations
        
        # The following list contains all training parameters which are subject to exploration
        # -----------------------------------------
        self.N = args.N #number of tasks (int)
        self.Nout = args.Nout #num of tasks used for one meta-update (int)
        self.K = args.K #no datapoints available per task (int)
        self.Kin = args.Kin #num of datapoints per task used in the inner loop (int)
        self.B =  self.N*self.K #total available number of points (Budget: B)
        self.train_test_split_inner = args.train_test_split_inner #ratio of train points to test points used in the inner loop of MAML (float)
        self.num_adaptation_steps = args.num_adaptation_steps #number of gradient updates used in the inner loop of MAML (int)
        # -----------------------------------------
        
        self.max_num_epochs = args.max_num_epochs 
        self.num_workers = args.num_workers 
        self.data_and_labels_splitter = args.data_and_labels_splitter
        
        self.model = model.to(device=self.device)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, 'min', factor=0.5, patience=300, verbose=True)
        self.lr_rates = []
        self.stopping_criterion = StopByAnnealing(4)
        
        self.save_hist_data=False
        self.i=1
        
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

        task_train_points = task_data[:no_of_train_points]
        task_test_points = task_data[no_of_train_points:]
        
        return task_train_points, task_test_points
    
    def adaptation(self, train_input, train_target):
        """Perform parameter adaptation using train input and train target"""
        # Inputs: 1.) train_input:  torch tensor of size: (Kin,) + input_size
        #         2.) train_target: torch tensor of size (Kin,) + output_size
        
        # Outputs: 1.) adapted parameters of the MetaModel

        params = None
        for step in range(self.num_adaptation_steps):
            train_logit = self.model(train_input, params=params)
            inner_loss = F.mse_loss(train_logit, train_target)
            
            self.model.zero_grad()
            params = gradient_update_parameters(self.model,
                                                inner_loss,
                                                step_size=self.inner_step_size,
                                                first_order=self.first_order)
        return params
    
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
        for task_idx, task_data in enumerate(dataloader):
            
            inputs, labels = self.data_and_labels_splitter(task_data)
            train_input, test_input = self.train_test_split(inputs, train_test_split_ratio, no_of_adaptation_points)
            train_target, test_target = self.train_test_split(labels, train_test_split_ratio, no_of_adaptation_points)
            
            train_input = train_input.to(device=self.device)
            train_target = train_target.to(device=self.device)
            test_input = test_input.to(device=self.device)
            test_target = test_target.to(device=self.device)
        
            """Adaptation"""
            params = self.adaptation(train_input, train_target)
            
            """Sum up task losses"""
            with torch.set_grad_enabled(self.model.training):
                test_logit = self.model(test_input, params=params)
                task_loss = F.mse_loss(test_logit, test_target)
                mean_outer_loss += task_loss
                
        
        mean_outer_loss /= task_idx+1
        return mean_outer_loss
        
    
    def _training_iteration(self, dataloader):
        """Training iteration - 1 gradient update"""
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
            batch_avg_loss = self._training_iteration(dataloader)
            train_avg_loss += batch_avg_loss
        
        train_avg_loss /= num_of_training_iterations
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
        if epoch - checkpoints_epochs[-1] > 500:
            return True
        else:
            return False


    def stop_training(self, epoch, checkpoints_epochs):
        #The stopping criterion consists of the seperate check of three criteria
        #If any of the three criteria is satisfied, the training stops
        return self._check_lr_annealing(epoch) or \
               self._check_max_epoch(epoch) #or \
               #self._check_val_loss_improvement(epoch, checkpoints_epochs)
        
    def train(self, train_dataloader, val_dataloader):
        meta_update_times = []
        checkpoints_epochs = [0]
        epoch=1
        while not self.stop_training(epoch, checkpoints_epochs):
            start=time.time()
                
            train_avg_loss = self.training_phase(train_dataloader)
            
            end=time.time()
            time_diff = end-start
            meta_update_times.append(time_diff)
            
            # LR scheduling
            self.lr_scheduler.step(train_avg_loss)
            
            #evaluate on the validation set
            if epoch % 20 == 10:
                self.model.eval()
                val_avg_loss = self.mean_outer_loss(val_dataloader)
                see.logs.update_cache(val_avg_loss, 'val_avg_loss')
                
                print('epoch: %d -- train_loss:%.4f - val_loss:%.4f' % (epoch, train_avg_loss, val_avg_loss))
                
                #save the model if it achieves the best performance on the validation set
                if val_avg_loss == min(see.logs.cache['val_avg_loss']):
                    checkpoints_epochs.append(epoch)
                    see.logs.save_model(self.model, checkpoint_index=epoch)

                    # clean the model directory to avoid capturing too much memory
                    if len(checkpoints_epochs) > 3:
                        see.logs.clean_model_checkpoints(checkpoints_epochs[-1])
            
            epoch+=1
        
        # save the number of meta-updates to convergence
        see.logs.update_cache(epoch, 'total_meta_updates')
        
        # save the mean time of a meta-update
        see.logs.update_cache(np.mean(meta_update_times), 'expected_meta_update_time')
        
        # Save logs
        see.logs.write(see.logs.cache, name='log.pickle')
        
        # Save model weights at the end of training
        see.logs.save_model(self.model, checkpoint_index=epoch)

