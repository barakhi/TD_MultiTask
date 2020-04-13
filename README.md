# Multi-Task Learning by a Top-Down Control Network
This is the official implementation repository for our paper  `Multi-Task Learning by a Top-Down Control Network`.

**Abstract:**
```
A general problem that received considerable recent attention is how to perform multiple tasks in the same
network, maximizing both prediction accuracy and efficiency of training. Recent approaches address this
problem by branching networks, or by a channel-wise modulation of the feature-maps with task specific vectors. 
We propose a novel architecture that uses a top-down network to modify the main network according to the task
in a channel-wise, as well as spatial-wise, image-dependent computation scheme. We show the effectiveness of
our scheme by achieving better results than alternative state-of-the-art approaches to multi-task learning. 
We also demonstrate our advantages in terms of task selectivity, scaling the number of tasks, learning from
fewer examples and interpretability.
```
## Architecture
Our architecture consists of three streams: BU2 (bottom-up) is our main recognition network. It performs multi recognition tasks **with only one branch** (requires high task selectivity). The two other streams gather information that will be used to modify the activations along BU2: 
- BU1 (share weights with BU2) extracts the image information. 
- The TD (top-down) stream combines the image information (in a top-down manner) with the task information. Its outputs multiply the activations along BU2 and control the recognition process (tensor-wised, task-dependent and image-aware).

The information passes between the streams through two sets of lateral connections (illustrated only once for clarity).

A detailed illustration of our architecture: ![:](https://github.com/barakhi/TD_MultiTask/blob/master/images/detailed_arch__.png)


## Prerequisites
- python 3
- pytorch 1.0 
- torchvision
- numpy
- tqdm

## Getting Started
Run:
```
python train_multi_task_counter_stream.py --param_file=./params/mmnist_counter_9tasks_params.json --epochs 100
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

**If you find this repository useful cite this paper:**
```
To Add
```
## Acknowledgments
```
To Add
```

