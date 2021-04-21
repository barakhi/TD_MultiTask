import sys
import torch
#import click
import json
import datetime
from timeit import default_timer as timer

import numpy as np
import argparse
import json
import argparse
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import losses
from data import datasets
from data import metrics
from models.model_selector import get_model


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--param_file', default='', type=str, metavar='PATH',
                    help='param file location')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_tasks', default=9, type=int, metavar='N',
                    help='number of tasks')
parser.add_argument('--n_eval', default=20, type=int, metavar='N',
                    help='number of epochs for evaluation')

def train_multi_task_counter_stream():

    args = parser.parse_args()

    if args.param_file:
        if os.path.isfile(args.param_file):
            print("=> loading params '{}'".format(args.param_file))
            with open(args.param_file) as json_params:
                params = json.load(json_params)
        else:
            print("=> no param_file found at '{}'".format(args.param_file))

    NUM_TASKS = args.num_tasks
    if 'tasks' not in params.keys():
        params['tasks'] = [str(k) for k in range(NUM_TASKS)]

    train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params)
    metric = metrics.get_metrics(params)#{'dataset': 'rightof'})

    model = get_model(params, NUM_TASKS)
    model = model.cuda()
    model_params = model.parameters()

    start_epoch = 0
    best_acc1 = 0.0

    if 'RMSprop' in params['optimizer']:
        optimizer = torch.optim.RMSprop(model_params, lr=params['lr'])
    elif 'Adam' in params['optimizer']:
        optimizer = torch.optim.Adam(model_params, lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model_params, lr=params['lr'], momentum=0.9)

    tasks = params['tasks']
    all_tasks = [str(k) for k in range(1, 10)]

    savefile = 'saved_models/{}_model.pkl'.format(params['exp_id'])
    if os.path.isfile(savefile):
        print("=> loading checkpoint '{}'".format(savefile))
        checkpoint = torch.load(savefile)
        start_epoch = checkpoint['epoch']
        cur_acc1 = checkpoint['cur_acc']
        best_acc1 = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_rep'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(savefile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(savefile))



    print('Starting training with parameters \n \t{} \n'.format(str(params)))


    n_iter = 0

    loss_init = {}
    rand_task_one_hot = torch.FloatTensor(params['batch_size'], 2).cuda()
    for epoch in tqdm(range(start_epoch, args.epochs)):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        if (epoch+1) % int(args.epochs*30/100) == 0:
            # Every 50 epoch, half the LR
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print('Half the learning rate{}'.format(n_iter))

        model.train()

        for batch in train_loader:
            n_iter += 1
            # First member is always images
            images = batch[0]
            images = images.cuda()

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i+1]
                labels[t] = labels[t].cuda()

            #transforms.ToPILImage()(transforms.Normalize(mean=[-0.1307 / 0.3081], std=[1 / 0.3081])(images[5, :, :, :].float().cpu())).show()


            stacked_labels = torch.stack((torch.stack((labels['1'], labels['2'], labels['3']), dim=0),
                                          torch.stack((labels['4'], labels['5'], labels['6']), dim=0),
                                          torch.stack((labels['7'], labels['8'], labels['9']), dim=0)), dim=1)

            right_of_valid = [(torch.sum(torch.sum(stacked_labels[:, :2, :] == digit, dim=0), dim=0)) * (
                torch.sum(torch.sum(stacked_labels == digit, dim=0), dim=0)) == 1 for digit in range(10)]

            indices = [((stacked_labels == digit) * right_of_valid[digit].view(1, 1, -1)).nonzero() for digit in range(10)]
            indices_sorted = [indices[digit][indices[digit][:, 2].sort(dim=-1)[1]] for digit in range(10)]
            labels_rightof = [-torch.ones(images.size(0)).cuda().long() for digit in range(10)]
            for digit in range(10):
                indices_rightof = indices_sorted[digit][:, 0] * 3 * images.size(0) + (1 + indices_sorted[digit][:, 1]) * images.size(0) + indices_sorted[digit][:, 2]
                labels_rightof[digit][right_of_valid[digit].nonzero().view(-1)] = torch.index_select(stacked_labels.view(-1), 0, (
                            indices_rightof).view(-1))



            with torch.no_grad():



                weights = torch.transpose(torch.stack([lab for lab in right_of_valid]), 0, 1)
                weights[torch.sum(weights, dim=1) == 0, 0] = 1
                rand_task = torch.multinomial(weights.float(), 1)  # was 16
                rand_task_one_hot = torch.FloatTensor(images.size(0), 10)
                rand_task_one_hot.zero_()
                rand_task_one_hot.scatter_(1, rand_task.cpu(), 1)
                rand_task_one_hot = rand_task_one_hot.cuda()

                labels_com = torch.masked_select(
                    torch.stack([lab_line for lab_line in labels_rightof]).transpose(0, 1),
                    rand_task_one_hot.byte()).view(images.size(0), -1).long().squeeze()

            optimizer.zero_grad()
            bu1_log, bu2_log, seg_sig = model(images, rand_task_one_hot)

            loss_bu2 = nn.CrossEntropyLoss(ignore_index=-1)(bu2_log, labels_com)
            loss = loss_bu2


            loss.backward()
            optimizer.step()


        cur_acc1 = -1.0
        is_best = False
        if (epoch + 1) % 100 == 0:
            model.eval()

            tot_loss = {}
            tot_loss['all'] = 0.0
            met = {}

            with torch.no_grad():

                for t in all_tasks:
                    tot_loss[t] = 0.0
                    met[t] = 0.0

                num_val_batches = 0
                for batch_val in val_loader:
                    val_images = batch_val[0].cuda()
                    labels_val = {}
                    mask1 = None
                    mask2 = None

                    for i, t in enumerate(all_tasks):
                        labels_val[t] = batch_val[i + 1]
                        labels_val[t] = labels_val[t].cuda()

                    stacked_labels_val = torch.stack((torch.stack((labels_val['1'], labels_val['2'], labels_val['3']), dim=0),
                                                  torch.stack((labels_val['4'], labels_val['5'], labels_val['6']), dim=0),
                                                  torch.stack((labels_val['7'], labels_val['8'], labels_val['9']), dim=0)), dim=1)

                    right_of_valid_val = [(torch.sum(torch.sum(stacked_labels_val[:, :2, :] == digit, dim=0), dim=0)) * (
                        torch.sum(torch.sum(stacked_labels_val == digit, dim=0), dim=0)) == 1 for digit in range(10)]

                    # torch.index_select(stacked_labels, 0, (stacked_labels[:, :2, :] == 0).nonzero()[:, 0])
                    indices_val = [((stacked_labels_val == digit) * right_of_valid_val[digit].view(1, 1, -1)).nonzero() for digit in
                               range(10)]
                    indices_sorted_val = [indices_val[digit][indices_val[digit][:, 2].sort(dim=-1)[1]] for digit in range(10)]
                    labels_rightof_val = [-torch.ones(val_images.size(0)).cuda().long() for digit in range(10)]
                    for digit in range(10):
                        indices_rightof_val = indices_sorted_val[digit][:, 0] * 3 * val_images.size(0) + (1 + indices_sorted_val[digit][:, 1]) * val_images.size(0) + \
                                          indices_sorted_val[digit][:, 2]
                        labels_rightof_val[digit][right_of_valid_val[digit].nonzero().view(-1)] = torch.index_select(
                            stacked_labels_val.view(-1), 0, (
                                indices_rightof_val).view(-1))

                    for i, t in enumerate(range(10)):
                        rand_task = i * torch.ones(val_images.size(0), 1).long().cuda()
                        rand_task_one_hot = torch.FloatTensor(val_images.size(0), 10).cuda()
                        rand_task_one_hot.zero_()
                        rand_task_one_hot.scatter_(1, rand_task, 1)
                        rand_task_one_hot = rand_task_one_hot.cuda()

                        labels_val_com = torch.masked_select(
                            torch.stack([lab_line for lab_line in labels_rightof_val]).transpose(0, 1),
                            rand_task_one_hot.byte()).view(val_images.size(0), -1).long().squeeze()


                        bu1_log, bu2_log, seg_sig = model(val_images, rand_task_one_hot)

                        loss_bu2 = nn.CrossEntropyLoss(ignore_index=-1)(bu2_log, labels_val_com)


                        if torch.any(labels_val_com >= 0):
                            metric[t].update(bu2_log[labels_val_com >= 0], labels_val_com[labels_val_com >= 0])

                    num_val_batches += 1

                metric_results_list = []
                for t in range(10):
                    metric_results = metric[t].get_result()
                    print(t, metric_results)
                    metric_results_list.append(metric_results['acc'])
                    metric[t].reset()
                print(float(torch.mean(torch.FloatTensor(metric_results_list))))


                cur_acc1 = float(torch.mean(torch.FloatTensor(metric_results_list)))
                is_best = cur_acc1 > best_acc1
                best_acc1 = max(cur_acc1, best_acc1)

        state = {'epoch': epoch+1,
                'model_rep': model.state_dict(),
                'optimizer_state' : optimizer.state_dict()}
        state['cur_acc'] = cur_acc1 #metric_results_list
        state['best_acc'] = best_acc1
        
        torch.save(state, "saved_models/{}_model.pkl".format(params['exp_id']))

        if is_best:
            shutil.copyfile("saved_models/{}_model.pkl".format(params['exp_id']), "saved_models/{}_model_best.pkl".format(params['exp_id']))



            end = timer()
            print('Epoch ended in {}s'.format(end - start))

    model.eval()

    tot_loss = {}
    tot_loss['all'] = 0.0
    met = {}

    with torch.no_grad():

        for t in all_tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        for batch_val in val_loader:
            val_images = batch_val[0].cuda()
            labels_val = {}
            mask1 = None
            mask2 = None

            for i, t in enumerate(all_tasks):
                labels_val[t] = batch_val[i + 1]
                labels_val[t] = labels_val[t].cuda()

            stacked_labels_val = torch.stack((torch.stack((labels_val['1'], labels_val['2'], labels_val['3']), dim=0),
                                              torch.stack((labels_val['4'], labels_val['5'], labels_val['6']), dim=0),
                                              torch.stack((labels_val['7'], labels_val['8'], labels_val['9']), dim=0)),
                                             dim=1)

            right_of_valid_val = [(torch.sum(torch.sum(stacked_labels_val[:, :2, :] == digit, dim=0), dim=0)) * (
                torch.sum(torch.sum(stacked_labels_val == digit, dim=0), dim=0)) == 1 for digit in range(10)]

            # torch.index_select(stacked_labels, 0, (stacked_labels[:, :2, :] == 0).nonzero()[:, 0])
            indices_val = [((stacked_labels_val == digit) * right_of_valid_val[digit].view(1, 1, -1)).nonzero() for
                           digit in
                           range(10)]
            indices_sorted_val = [indices_val[digit][indices_val[digit][:, 2].sort(dim=-1)[1]] for digit in range(10)]
            labels_rightof_val = [-torch.ones(val_images.size(0)).cuda().long() for digit in range(10)]
            for digit in range(10):
                indices_rightof_val = indices_sorted_val[digit][:, 0] * 3 * val_images.size(0) + (
                            1 + indices_sorted_val[digit][:, 1]) * val_images.size(0) + \
                                      indices_sorted_val[digit][:, 2]
                labels_rightof_val[digit][right_of_valid_val[digit].nonzero().view(-1)] = torch.index_select(
                    stacked_labels_val.view(-1), 0, (
                        indices_rightof_val).view(-1))

            for i, t in enumerate(range(10)):
                rand_task = i * torch.ones(val_images.size(0), 1).long().cuda()
                rand_task_one_hot = torch.FloatTensor(val_images.size(0), 10).cuda()
                rand_task_one_hot.zero_()
                rand_task_one_hot.scatter_(1, rand_task, 1)
                rand_task_one_hot = rand_task_one_hot.cuda()

                labels_val_com = torch.masked_select(
                    torch.stack([lab_line for lab_line in labels_rightof_val]).transpose(0, 1),
                    rand_task_one_hot.byte()).view(val_images.size(0), -1).long().squeeze()

                bu1_log, bu2_log, seg_sig = model(val_images, rand_task_one_hot)

                loss_bu2 = nn.CrossEntropyLoss(ignore_index=-1)(bu2_log, labels_val_com)

                if torch.any(labels_val_com >= 0):
                    metric[t].update(bu2_log[labels_val_com >= 0], labels_val_com[labels_val_com >= 0])

            num_val_batches += 1

        metric_results_list = []
        for t in range(10):
            metric_results = metric[t].get_result()
            print(t, metric_results)
            metric_results_list.append(metric_results['acc'])
            metric[t].reset()
        print(float(torch.mean(torch.FloatTensor(metric_results_list))))

    cur_acc1 = float(torch.mean(torch.FloatTensor(metric_results_list)))
    is_best = cur_acc1 > best_acc1
    best_acc1 = max(cur_acc1, best_acc1)

    #print(t, metric_results)



if __name__ == '__main__':
    train_multi_task_counter_stream()
