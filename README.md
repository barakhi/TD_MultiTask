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
Our architecture consists of three streams: TBD

Here is a detailed illustration based on a LeNet backbone ![:](https://github.com/barakhi/TD_MultiTask/blob/master/images/detailed_arch__.png)

## Prerequisites
- python 3
- pytorch 1.0 
- torchvision
- numpy
- tqdm

## Getting Started
Run:
```
python train_multi_task_counter_stream.py --param_file=./params/mmnist_counter_9tasks_params.json
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

**If you find this repository useful cite this paper:**
```
To Add
```
## Acknowledgments


