# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:19:28 2020

@author: Georgios
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
import pandas as pd
import torchvision.transforms as transforms

def n_choose_r(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom 

# transform = torchvision.transforms.Resize([256,256])
transform = transforms.Compose([transforms.Resize((84, 84)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
def load_data_in_memory(data_root):
    print(data_root)
    df = pd.DataFrame()
    for phase in ['train','test','val']:
        df = pd.concat((df,pd.read_csv(os.path.join(data_root,phase+'.csv'))),axis=0)
        print("dataframe created...")
        
    class_index2name = {}
    class_name2index = {}
    class_data_dict = {}
    class_data_names_dict = {}

    for index, _class in enumerate(df.label.unique()): ##index, names of classes
        class_images_names = df.loc[df.label == _class].filename.values ## list of image filenames belonging to _class
        class_data_names_dict[index] = class_images_names
        class_images = [] ;  #print(index)
        for image in class_images_names: ## for each image belonging to _class
                img = np.array(transform(Image.open(os.path.join(data_root,'images', image)).convert('RGB'))) ## convert image to np.array
                # img = np.rollaxis(img, 2, 0)  ## reformat in correct format NO NEED IF ALREADY TENSOR
                class_images.append(img.astype(np.float32)) ## add image to list

        class_data_dict[index] = class_images ## list of images belonging to _class [images]
        class_index2name[index] = _class ## _class name [label]
        class_name2index[_class] = index ## index of class [label in integer form] 
        # print(class_data_dict[index][0].shape)
    return class_data_dict, class_data_names_dict, class_index2name, class_name2index

# class_images, class_images_names, class_index2name, class_name2index = load_data_in_memory()


def ReadClasses(root_dir, phase, class_name2index):
    print(root_dir)
    class_names = pd.read_csv(os.path.join(root_dir,phase+'.csv')).label.unique()    
    return [class_name2index[name] for name in class_names] ## list of index [integers] of classes for the phase

root_dir = 'meta/dataset/miniimagenet'
class_images, class_images_names, class_index2name, class_name2index = load_data_in_memory(root_dir)


class ImagenetStaticDataset(StaticMetaDataset):
    '''
    Implementation of a static Sinusoid Dataset.
    '''
    def __init__(self, root_dir, mode, no_of_tasks, classes_per_task, no_of_data_points_per_task):
        '''
        no_of_tasks: positive integer - initial number of tasks to sample.
        no_of_data_points_per_task - positive int.
        '''
        super(ImagenetStaticDataset, self).__init__()
        self.split_root = root_dir
        self.available_classes = ReadClasses(self.split_root, mode, class_name2index) ## available classes for a phase (list of integers representing class number)
        self.classes_per_task = classes_per_task ## input classes per task
        self.task_additions = 0
        self.task_seed = 0
        if no_of_tasks == -1:
            self.no_of_tasks = 2000
            self.infiniteTask = True
        else:
            self.no_of_tasks = no_of_tasks
            self.infiniteTask = False     
        self.no_of_data_points_per_task =  no_of_data_points_per_task  
        self.task_parametrization_array = self.generate_task_parametrizations(self.no_of_tasks)
        self.task_dataset_array = self.generate_task_datasets(self.task_parametrization_array, no_of_data_points_per_task)

    def reinitialize(self):
        self.task_parametrization_array = self.generate_task_parametrizations(self.no_of_tasks)
        self.task_dataset_array = self.generate_task_datasets(self.task_parametrization_array, self.no_of_data_points_per_task)


    ###########################################################################        
    def generate_task_parametrizations(self, no_of_tasks : int):
        '''
        Implementation of generate task parametrizations in base class
        '''
        
        try:
            available_tasks = n_choose_r(len(self.available_classes), self.classes_per_task) - self.task_additions ## number of available tasks
            assert available_tasks >= no_of_tasks, "no_of_tasks greater than available"
        
        except AssertionError:
            no_of_tasks = available_tasks
            print('no_of_tasks set to num of available tasks: %d' % no_of_tasks)
            
            if no_of_tasks == 0:
                return []
        
        task_parameterisation_array = []
        for i in range(no_of_tasks):
            #each task should exist only once in task parametrisation array
            task_sampled = False
            task_in_memory = False
            while not(task_sampled==True and task_in_memory==False):
                self.task_seed += 1 ## why change the seed? Remove THIS! (1)Make sure tasks don't repeat when adding new tasks
                random.seed(self.task_seed)## REMOVE THIS
                
                task_classes = random.sample(self.available_classes, self.classes_per_task)  ## sample classes (list of integers) for each task, ie each task has a random sample of e.g. 5 classes per task
                task_sampled = True ## task was sampled, this is always true
                
                if task_classes in task_parameterisation_array: ## if already sampled then re-sample this combination of classes 
                    task_in_memory = True
                else:
                    task_in_memory=False
            ### [[1,2,5,3],[1,2,5,3]] NOT POSSIBLE
                
            task_parameterisation_array.append(task_classes) ## append to list of [ assignment of classes (integers) for each task], e.g. [[1, 3, 2,10,5],[7...] ]
        
        self.task_additions += len(task_parameterisation_array) ## this is outside loop: number of tasks

        return task_parameterisation_array ## return set of [ assignment of classes (integers) for each task]
    
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
            task_dataset_array = np.empty(len(task_parametrization_array), dtype = ImagenetStaticTask)
    
            for i, task_parametrization in enumerate(task_parametrization_array):
                task_dataset_array[i] = ImagenetStaticTask(task_parametrization, no_of_points_per_task)
                
            return task_dataset_array ## array of ImagenetStaticTask datasets

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
            self.add_new_tasks(task_increase, data_for_new_tasks)


class ImagenetStaticTask(Dataset):
    def __init__(self, task_parametrization: list, no_of_samples : int):
        self._task = task_parametrization #task is a list of integers. Each integer corresponds to one class.
        self.task_classes = task_parametrization ## same as above

        self.available_images = {}
        for task_class in self.task_classes: ## for each class (integer) for that task
            self.available_images[task_class] = list(np.arange(len(class_images[task_class]))) ## [0,1,2,...,number of images for that class] -- get all the list of images for that class
        
        if no_of_samples > 0:
            self.index2class={}
            self.index2label={}
            self.index2class_index={}
            _ = self.sample_datapoints(no_of_samples) ## sample data points
    
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
        
        for label, task_class in enumerate(self.task_classes): ## iterate over list of classes for that task
            for _ in range(no_of_datapoints):
                random_index = random.randint(0, len(available_images[task_class])-1) #random.randint is inclusive for both ends
                task_class_index = available_images[task_class].pop(random_index) ## index of image for the list of images in that class and pops out element in random index, so you cant sample the same image again for that task
                self.index2class[current_index] = task_class ## each data point for true class has that class
                self.index2label[current_index] = label ## label is restricted to be [0,1,2...] when classes are [1, 3, 2,10,12]
                self.index2class_index[current_index] = task_class_index ## true index of the image [e.g. 120] for that class in the global class_images dataset
                current_index+=1
        
        self.available_images = available_images
        
        augmented=True
        return augmented

    def __getitem__(self, index):
        image_class = self.index2class[index]
        image_label = self.index2label[index]
        image_class_index = self.index2class_index[index]

        image = class_images[image_class][image_class_index] ##  actual image : in class_images for that class at index k, e.g. 120th image of class 52
        label = image_label ## label from [0,1,2,3,4] within the task
        
        return image, label

    def __len__(self):
        return len(self.index2class_index)
    
    def add_data(self, no_of_samples):
        augmented = self.sample_datapoints(no_of_samples)
        if not augmented:
            print('No more available datapoints for task:', self._task)
            return

class ImagenetBatchSampler(Sampler):
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
        super(ImagenetBatchSampler, self).__init__(data_source)
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
