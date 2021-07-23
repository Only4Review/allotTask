# How to distribute data across tasks for meta-learning?

This repository is the official implementation of [How to distribute data across tasks for meta-learning?](https://arxiv.org/abs/2030.12345). 

Meta-learning models transfer the knowledge acquired from previous tasks to quickly learn new ones. They are trained on benchmarks with a fixed number of data points per task. This number is usually arbitrary and it is unknown how it affects performance at testing. Since labelling of data is expensive, finding the optimal allocation of labels across training tasks may reduce costs. Given a fixed budget of labels, should we use a small number of highly labelled tasks, or many tasks with few labels each? Should we allocate more labels to some tasks and less to others?

This anonymous repository is constructed to explore this questions using a linear regression experiment, and also real word data as CIFAR-FS, and mini Imagenet. 

## Requirements

- We provide the *requirements.txt* for users to set up the working environment. To install requirements:

```setup
pip install -r requirements.txt
```

- We also provide a **DOCKERFILE** for users to create a docker environment.
- The CIFAR-FS and mini-ImageNet datasets, are public well-known datasets and therefore not included in this repository.

## Training

The repository contains experimental code to run all the experiments  described in the paper. Each point of the results showed in the plots of the papers represent a single run of experimental code with different input variables. The code below show examples of how to run this experiments.   

- To run linear regression experiments: 

```train linear regression
python ./experiment/LinearRegression.py
```

 (*Hyperparameters are defined in the first lines and the last line saves the loss (train and test) and the hyperparameters in a single file.*)

- To run Sinusoid regression experiments: 

```train linear regression
python ./experiment/SinusoidExperiment.py [--budget 1000]
```

- To run CIFAR-FS experiments: 

```train Cifar-FS
python ./experiment/CifarExperiment.py [--budget 1000 --num_datapoints_per_class 10]
```

- To run mini-ImageNet experiments:

```train imagenet
 python ./experiment/ImagenetExperiment.py [--budget 1000]
```

- To run CIFAR-FS "Hierarchy class" hard tasks experiments:

```train hierarchy
 python ./experiment/CifarHierarchicalExperiment.py --num_easy 333 --num_hard 333 --num_datapoints_per_class_easy 28 --num_datapoints_per_class_hard 2 --hierarchy_json meta/dataset/cifar100/hierarchy_3_hiperclass.json
```

- To run CIFAR-FS  ""Noisy labels" hard tasks"  experiments:

```train noisy
python ./experiment/CifarNoisyLabelsExperiment.py --num_easy 333 --num_hard 333 --num_datapoints_per_class_easy 28 --num_datapoints_per_class_hard 2 --noise_percent 20
```

To specifically reproduce the results in the paper use the training conditions specified in appendix A of the paper.

#### About evaluation

All the experiments scripts include a function for evaluation in the test dataset that is automatically run at the end of the training. These evaluation functions can be called for evaluation outside the training script.

 We do not provide pretrained models, as more than 1500 individual training runs were done in the experiments described in the paper, and each model is individually fast to train ; all experiments ran on a single Nvidia 2080 GPU with an average runtime of 1.5 hours per run of the CIFAR-FS dataset and 5 hours for miniImageNet.



## Results

These experiments show that:

 1) If tasks are homogeneous, there is a uniform optimal allocation, whereby all tasks get the same amount of data. 

2) At fixed budget, there is a trade-off between number of tasks and number of data points per task, with a unique and constant optimum *(This two results can be tested by running  CifarExperiment.py,  ImagenetExperiment.py, or LinearRegression.py; to check specific training conditions please check appendix A1 and A2 of the paper.)*  

 3) When trained separately, harder task should get more data, at the cost of a smaller number of tasks;

 4) When training on a mixture of easy and hard tasks, more data should be allocated to easy tasks. *(This two results can be tested by running  CifarNoisyLabelsExperiment.py  or CifarHierarchicalExperiment.py; to check specific training conditions please check appendix A2 and A3 of the paper.)*
