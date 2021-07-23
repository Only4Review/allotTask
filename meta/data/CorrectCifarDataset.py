# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:19:28 2020

@author: xxx
"""

from torch.utils.data.dataset import Dataset
from meta.data.StaticDataset import StaticMetaDataset, FullBatchSampler
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import random
import numpy as np
from PIL import Image
import os
import operator as op
from functools import reduce
import json
import math
import itertools
import random

def n_choose_r(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom 

def load_data_in_memory(data_root):
    class_index2name = {}
    class_name2index = {}
    class_data_dict = {}
    class_data_names_dict = {}
    for index,_class in enumerate(sorted(os.listdir(data_root))): 
        class_root_dir = os.path.join(data_root, _class)
        class_images_names = os.listdir(class_root_dir)
        class_data_names_dict[index] = class_images_names
        
        class_images = []
        for image in class_images_names:
            img = np.array(Image.open(os.path.join(class_root_dir, image)))
            img = np.rollaxis(img, 2, 0)  
            class_images.append(img.astype(np.float32))
        
        class_data_dict[index] = class_images
        class_index2name[index] = _class
        class_name2index[_class] = index
        
    return class_data_dict, class_data_names_dict, class_index2name, class_name2index

def ReadClasses(split_root, phase, class_name2index):
    with open(os.path.join(split_root, '%s.txt' % phase)) as f:
        lines = f.readlines()
        
    class_names = []
    for line in lines:
        class_names.append(line[:-1])
    
    return [class_name2index[name] for name in class_names]


root_dir = 'meta/dataset/cifar100'
class_images, class_images_names, class_index2name, class_name2index = load_data_in_memory(os.path.join(root_dir, 'data'))

class CifarStaticDataset(StaticMetaDataset):
    '''
    Implementation of a static Sinusoid Dataset.
    '''
    def __init__(self, root_dir, mode, no_of_tasks, classes_per_task, no_of_data_points_per_task):
        '''
        no_of_tasks: positive integer - initial number of tasks to sample.
        no_of_data_points_per_task - positive int.
        '''
        super(CifarStaticDataset, self).__init__()
        self.split_root = os.path.join(root_dir, 'splits', 'bertinetto')
        self.available_classes = ReadClasses(self.split_root, mode, class_name2index)
        
        self.classes_per_task = classes_per_task
        self.task_seed = 0
        if no_of_tasks == -1:
            self.no_of_tasks = 2000
            self.infiniteTask = True
        else:
            self.no_of_tasks = no_of_tasks
            self.infiniteTask = False
        self.no_of_data_points_per_task = no_of_data_points_per_task
        self.task_parametrization_array = self.generate_task_parametrizations(self.no_of_tasks)
        self.task_dataset_array = self.generate_task_datasets(self.task_parametrization_array, self.no_of_data_points_per_task)
            
    ###########################################################################   
    def reinitialize(self):
        self.task_parametrization_array = self.generate_task_parametrizations(self.no_of_tasks)
        self.task_dataset_array = self.generate_task_datasets(self.task_parametrization_array, self.no_of_data_points_per_task)
             
    ###########################################################################        
    def generate_task_parametrizations(self, no_of_tasks:int):
        '''
        Implementation of generate task parametrizations in base class
        '''
        # all_tasks_list = list(itertools.combinations(self.available_classes, self.classes_per_task))
        # random.shuffle(all_tasks_list)
        # task_parameterisation_array = all_tasks_list[0:no_of_tasks]

        try:
            available_tasks = n_choose_r(len(self.available_classes), self.classes_per_task)
            assert available_tasks >= no_of_tasks, "no_of_tasks greater than available"
        
        except AssertionError:
            no_of_tasks = available_tasks
            print('no_of_tasks set to num of available tasks: %d' % no_of_tasks)
            
            if no_of_tasks == 0:
                return []
        
        task_parameterisation_array = []
        for i in range(no_of_tasks):
            task_classes = random.sample(self.available_classes, self.classes_per_task)
            task_parameterisation_array.append(task_classes)
        
        return task_parameterisation_array
    
    #############################################################################
    def generate_task_datasets(self, task_parametrization_array, no_of_points_per_task):
        '''
        Input:
            task_parametrization_array: an array of true task parameters.
            no_of_points_per_task: pos. int
        Output:
            array of SinusoidStaticTask datasets, corresponding to the input parameters.
        '''
        if not task_parametrization_array:
            return []
        else:
            task_dataset_array = np.empty(len(task_parametrization_array), dtype = CifarStaticTask)
    
            for i, task_parametrization in enumerate(task_parametrization_array):
                task_dataset_array[i] = CifarStaticTask(task_parametrization, no_of_points_per_task)
                
            return task_dataset_array

    ############################################################################    
    def add_new_tasks(self, no_of_tasks, no_of_points_per_task):
        '''Adds additional tasks to MetaDataset.
        Input: 
            no_of_tasks: pos. int
            no_of_poinmts_per_task: pos. int
        Output:
            None. Adds to the class variables.
        '''
        new_tasks = self.generate_task_parametrizations(no_of_tasks)
        new_task_dataset_array = self.generate_task_datasets(new_tasks, no_of_points_per_task)
        
        if len(new_task_dataset_array)==0:
            print('No tasks have been added. No available tasks')
            return
        else:
            self.task_parametrization_array =  np.concatenate([self.task_parametrization_array, new_tasks], axis = 0)
            self.task_dataset_array = np.concatenate([self.task_dataset_array,new_task_dataset_array], axis = 0)
        
    
    ##########################################################################
    def add_data_per_task(self, no_of_data_points_per_task):
        '''
        Adds data points to all tasks in the meta dataset
        Input: pos. int
        '''
        for task in self.task_dataset_array:
            task.add_data(no_of_data_points_per_task)

    ##########################################################################
    def increase_tasks_and_data_per_task(self, task_increase, data_increase):
        '''
        Increases the number of tasks and data points per task by specified number. 
        For newly created tasks will create the same number of data points as in first task 
        in the existing meta-dataset. And then add additional data points to all tasks - new and old.
        '''
        if data_increase > 0:
            self.add_data_per_task(data_increase)
        data_for_new_tasks = self.get_task_length(task_index = 0)
        if task_increase > 0:
            self.add_new_tasks(task_increase, data_for_new_tasks//self.classes_per_task) # debugged by wangq 20/12/2020

class CifarStaticDatasetHierarchy(CifarStaticDataset):
    '''
    Implementation of a static Cifar Dataset that takes into account the data structure to create hard and easy tasks
    '''
    def __init__(self, mode, hierarchy_json, no_of_easy, no_of_hard, classes_per_task, no_data_points_hard, no_data_points_easy):
        '''
        root_dir: directory containing the data. CAREFUL: If mode is Mix you would like to use diferent data directories.
        no_of_easy: positive integer - initial number of easy tasks to sample.
        no_of_data_points_per_task - positive int.
        root_dir - data folder
        mode - string 'Train', 'Test','Val', 'Mix_Train','Mix_Test', 'Mix_Val'. 
        With 'Mix' prefix clases are share between, train, test and validation  
        '''

        self.classes_per_task = classes_per_task
        
        self.classes_per_task = classes_per_task
        self.set_hierarchy(hierarchy_json,mode)
        
        if (no_of_easy == -1) or (no_of_hard == -1):
            no_of_easy = 1000
            no_of_datapoints = 1000
            self.infiniteTask = True

        else:
            self.infiniteTask = False
        
        self.no_of_tasks = no_of_easy + no_of_hard
        self.task_parametrization_array_hard, self.task_parametrization_array_easy = self.generate_task_parametrizations(no_of_hard, no_of_easy)
        self.task_parametrization_array = self.task_parametrization_array_hard + self.task_parametrization_array_easy
        self.task_dataset_array_hard = self.generate_task_datasets(self.task_parametrization_array_hard, no_data_points_hard) if no_of_hard != 0 else []
        self.task_dataset_array_easy = self.generate_task_datasets(self.task_parametrization_array_easy, no_data_points_easy) if no_of_easy != 0 else []
        self.task_dataset_array = np.concatenate((self.task_dataset_array_hard,self.task_dataset_array_easy), axis=0)
        
    def set_hierarchy(self, hierarchy_json, mode):
        with open(hierarchy_json) as json_file: 
            data = json.load(json_file)
        hyperclass_strucutre = data[mode] if 'Mix' not in mode else data['Mix'] 
        hard_tasks_list = []
        _tasks_classes = []
        for key, value in hyperclass_strucutre.items():
            hard_tasks_list +=list(itertools.combinations(value, self.classes_per_task))
            _tasks_classes += value
        
        self.hard_tasks_list = hard_tasks_list
        all_tasks_list = list(itertools.combinations(_tasks_classes, self.classes_per_task))
        self.easy_tasks_list = list (set(all_tasks_list)- set(hard_tasks_list))
        
    def generate_task_parametrizations(self, no_of_hard:int, no_of_easy:int):
        hard,easy =  self.hard_tasks_list, self.easy_tasks_list

        try:
            if len(hard) <= no_of_hard and not (len(easy) <= no_of_easy ):
                raise ValueError('Not enougt hard tasks available')
            elif len(hard) <= no_of_hard and (len(easy) <= no_of_easy ):
                raise ValueError('Not enougt hard tasks and not enought easy tasks available')
            elif not (len(hard) <= no_of_hard) and (len(easy) <= no_of_easy ):
                raise ValueError('Not enougt easy tasks available')
        except ValueError as ve:
            print(ve)      
        
        hard_list = random.choices(self.hard_tasks_list, k = no_of_hard)
        hard_list = [list(x) for x in hard_list]
        for item in hard_list:
            random.shuffle(item)
        easy_list = random.choices(self.easy_tasks_list, k = no_of_easy)
        easy_list = [list(x) for x in easy_list]
        for item in easy_list:
            random.shuffle(item)
        return hard_list, easy_list


class NoisyCifarStaticDataset(CifarStaticDataset):
    '''
    Implementation of static Cifar dataset with hard and easy tasks where hard tasks have noisy labels. 
    '''
    def __init__(self, mode,data_json, noise_percent, no_of_easy, no_of_hard, classes_per_task, no_data_points_hard, no_data_points_easy):
        '''
        data_json: json file that contains, train, test, val, split
        no_of_easy: positive integer - initial number of easy tasks to sample.
        no_of_hard: positive integer - initial number of easy tasks to sample
        no_of_data_points_hard: positive integer - number of data points in hard tasks
        no_of_data_points_easy:  positive integer - number of data points per easy tasks
        mode - string 'Train', 'Test','Val'
        noise_percent: positive integer - that represent the percent of hard task with noisy labels
        '''
        self.classes_per_task = classes_per_task
        self.data_json=data_json
        self.mode = mode
        self.noise_percent = noise_percent

        self.classes_per_task = classes_per_task        
        if no_of_easy == -1 :
            self.no_of_tasks = 2000
            self.infiniteTask = True
        else:
            self.no_of_tasks = no_of_easy + no_of_hard
            self.infiniteTask = False

        self.task_parametrization_array_hard, self.task_parametrization_array_easy = self.generate_task_parametrizations(no_of_hard, no_of_easy)
        self.task_parametrization_array = self.task_parametrization_array_hard + self.task_parametrization_array_easy
        self.task_dataset_array_hard = self.generate_noisy_task_datasets(self.task_parametrization_array_hard, no_data_points_hard) if no_of_hard != 0 else []
        self.task_dataset_array_easy = self.generate_task_datasets(self.task_parametrization_array_easy, no_data_points_easy) if no_of_easy != 0 else []
        self.task_dataset_array = np.concatenate((self.task_dataset_array_hard,self.task_dataset_array_easy), axis=0)

    def generate_task_parametrizations(self,no_of_hard, no_of_easy):
        with open(self.data_json) as json_file:
            data = json.load(json_file)
        _tasks_classes = data[self.mode] if 'Mix' not in self.mode else data['Mix']
        all_tasks = list(itertools.combinations(_tasks_classes, self.classes_per_task))

        all_tasks = random.choices(all_tasks, k = no_of_hard + no_of_easy)

        all_tasks = [list(x) for x in all_tasks]
        for item in all_tasks:
            random.shuffle(item)

        return all_tasks[0:no_of_hard], all_tasks[no_of_hard:no_of_hard+no_of_easy]

    def generate_noisy_task_datasets(self, task_parametrization_array, no_of_points_per_task):
        '''
        Input:
            task_parametrization_array: an array of true task parameters.
            no_of_points_per_task: pos. int
        Output:
            array of SinusoidStaticTask datasets, corresponding to the input parameters.
        '''
        if not task_parametrization_array:
            return []
        else:
            task_dataset_array = np.empty(len(task_parametrization_array), dtype = CifarStaticTask)
    
            for i, task_parametrization in enumerate(task_parametrization_array):
                task_dataset_array[i] = CifarStaticNoisyTask(task_parametrization, no_of_points_per_task, self.noise_percent)
                
            return task_dataset_array


class CifarStaticTask(Dataset):
    def __init__(self, task_parametrization: list, no_of_samples : int):
        self._task = task_parametrization #task is a list of integers. Each integer corresponds to one class.
        self.task_classes = task_parametrization

        self.available_images = {}
        for task_class in self.task_classes:
            self.available_images[task_class] = list(np.arange(len(class_images[task_class])))
        
        if no_of_samples > 0:
            self.index2class={}
            self.index2label={}
            self.index2class_index={}
            _ = self.sample_datapoints(no_of_samples)
    
    @property
    def task(self):
        return self._task
    
    
    def sample_datapoints(self, no_of_datapoints = 1):
        
        try:
            available_images = self.available_images.copy()
            
            min_num_available_images_accros_task_classes = min([len(self.available_images[x]) for x in self.task_classes])
            #make sure available datapoints are greater than or equal 
            #to the num of datapoints asked to be sampled
            
            assert min_num_available_images_accros_task_classes >= no_of_datapoints, "no_of_datapoints to sample greater than available for some task class"
        
        except AssertionError:
            no_of_datapoints = min_num_available_images_accros_task_classes
            print('no_of_datapoints set to num of min_num_available_images_accros_task_classes, i.e., %d' % no_of_datapoints)
            if no_of_datapoints == 0:
                augmented = False
                return augmented
        
        if bool(self.index2class) == False:
            current_index = 0
        else:
            current_index = len(list(self.index2class.keys()))
        
        for label, task_class in enumerate(self.task_classes):
            for _ in range(no_of_datapoints):
                random_index = random.randint(0, len(available_images[task_class])-1) #random.randint is inclusive for both ends
                task_class_index = available_images[task_class].pop(random_index)
                self.index2class[current_index] = task_class
                self.index2label[current_index] = label
                self.index2class_index[current_index] = task_class_index
                current_index+=1
        
        self.available_images = available_images
        
        augmented=True
        return augmented

    def __getitem__(self, index):
        image_class = self.index2class[index]
        image_label = self.index2label[index]
        image_class_index = self.index2class_index[index]
        image = class_images[image_class][image_class_index]
        label = image_label
        return image, label

    def __len__(self):
        return len(self.index2class_index)
    
    def add_data(self, no_of_samples):
        augmented = self.sample_datapoints(no_of_samples)
        if not augmented:
            print('No more available datapoints for task:', self._task)
            return

class CifarStaticNoisyTask(CifarStaticTask):
    def __init__(self, task_parametrization: list, no_of_samples : int, noise_percent: int):
        super().__init__(task_parametrization, no_of_samples)
        self.noise_percent = noise_percent
    def __getitem__(self, index):
        image_class = self.index2class[index]
        image_label = self.index2label[index]
        label = image_label
        image_class_index = self.index2class_index[index]
        image = class_images[image_class][image_class_index]
        if random.randrange(100) < self.noise_percent:
            _val_list=(list(range(0,len(self.task_classes)))) 
            _val_list.remove(label)
            label= random.choice(_val_list) 
        return image, label
    

class CifarBatchSampler(Sampler):
    '''
    Sampler that samples data from static dataset as follows:
    A sample will contain a given number of tasks (specified at sampler creation), or all tasks if 
    not specified and will sample a given number of data points from a task
    or all data points from a task if data points per task was not specified. 
    This is a batch sampler, and each batch will contain data for a single! task. In other words,
    iterating over this sampler will yield all data corresponding to one task in a single iteration 
    until the whole sample data is exhausted. 
    '''
    ###################################################################
    def __init__(self, data_source, no_of_tasks = None, no_of_data_points_per_task = None):
        super(CifarBatchSampler, self).__init__(data_source)
        self._data_source = data_source
        self._no_of_tasks = no_of_tasks
        self._no_of_data_points_per_task = no_of_data_points_per_task

    ###################################################################       
    def __iter__(self):
        if self._no_of_tasks is None:
            no_of_tasks = len(self._data_source)
        else:
            try:
                no_of_tasks = self._no_of_tasks
                assert no_of_tasks <= len(self._data_source), 'no of tasks greater than available'
            except AssertionError:
                no_of_tasks = len(self._data_source)
                #print('no of tasks set to available tasks: %d' % len(self._data_source))
        
        task_indices = np.random.choice(a = len(self._data_source), size = no_of_tasks, replace = False)
        for task in task_indices:
            indices_list = []
            task_length = self._data_source.get_task_length(task)
            
            if self._no_of_data_points_per_task is None:
                no_of_points = task_length
            else:
                try:
                    no_of_points = self._no_of_data_points_per_task
                    assert no_of_points <= task_length, 'no_of_datapoints more than available'
                
                except AssertionError:
                    no_of_points = task_length
                    #print('_no_of_data_points_per_task set to num available points: %d' % task_length)
            
            #print('(task, task_length) = (%d,%d)' % (task, no_of_points))
            task_data_index_list = np.random.choice(a = task_length, size = no_of_points, replace = False)

            for task_data_point in task_data_index_list:
                indices_list.append((task, task_data_point))

            yield indices_list

            
            
"""
CifarDataset = CifarStaticDataset(root_dir, 'train', no_of_tasks=12, classes_per_task=5, no_of_data_points_per_task=42)
CifarSampler = CifarBatchSampler(data_source = CifarDataset, no_of_tasks = None, no_of_data_points_per_task = None)
dataloader = DataLoader(CifarDataset, batch_sampler=CifarSampler, num_workers = 0)
for data, labels in dataloader:
    print(data.shape)
    print(labels.shape)
"""




"""
print(len(CifarDataset.task_dataset_array))

for taskDataset in CifarDataset.task_dataset_array[:10]:
    print(taskDataset.data.shape)

CifarDataset.increase_tasks_and_data_per_task(2, 30)


for taskDataset in CifarDataset.task_dataset_array[:10]:
    print(taskDataset.data.shape)

print(len(CifarDataset))
print(len(CifarDataset.task_dataset_array))

for task in CifarDataset.task_parametrization_array:
    print(task)
"""





#CifarSampler = CifarBatchSampler(data_source = CifarDataset, no_of_tasks = 1, N_way = 5, no_of_points_per_class = None)
#dataloader = DataLoader(CifarDataset, batch_sampler=CifarSampler, num_workers=0)
