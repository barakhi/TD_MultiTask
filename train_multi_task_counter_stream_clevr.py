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
parser.add_argument('--num_tasks', default=40, type=int, metavar='N',
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
    metric = metrics.get_metrics(params)

    model = get_model(params, NUM_TASKS)
    if 'mnist' not in params['dataset']:
        model = torch.nn.DataParallel(model)
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

    # TRAINING
    ##########
    print('Starting training with parameters \n \t{} \n'.format(str(params)))
    n_iter = 0

    for epoch in tqdm(range(start_epoch, args.epochs)):
        print('Epoch {} Started'.format(epoch))
        if (epoch+1) % int(args.epochs*0.3) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print('Half the learning rate{}'.format(n_iter))

        model.train()

        for batch in train_loader:


            n_iter += 1
            images = batch[0]
            images = images.cuda()

            labels = {}
            xy_task = {}
            # Read all targets of all tasks
            for i, t in enumerate(tasks):
                labels[t] = batch[1][:, i]
                labels[t] = labels[t].cuda()
                xy_task[t] = batch[1][:, (NUM_TASKS + i * 2):(NUM_TASKS + 2 + i * 2)]
                xy_task[t] = xy_task[t].float().cuda()

            with torch.no_grad():
                weights = torch.transpose(torch.stack([lab for lab in labels.values()]), 0, 1)
                weights[weights >= 0] = 1
                weights[weights == -1] = 0
                weights[torch.sum(weights, dim=1) == 0, 0] = 1
                rand_task = torch.multinomial(weights.float(), 1)  # was 16
                rand_task_one_hot = torch.FloatTensor(images.size(0), NUM_TASKS)
                rand_task_one_hot.zero_()
                rand_task_one_hot.scatter_(1, rand_task.cpu(), 1)
                rand_task_one_hot = rand_task_one_hot.cuda()
                labels_com = torch.masked_select(
                    torch.stack([lab_line for lab_line in labels.values()]).transpose(0, 1),
                    rand_task_one_hot.byte()).view(images.size(0), -1).long().squeeze()
                xy_task_com = torch.masked_select(
                    torch.stack([lab_line for lab_line in xy_task.values()]).transpose(0, 1),
                    rand_task_one_hot.byte().unsqueeze(2)).view(images.size(0), -1)
                xy_task_com[:, 0::2] = (xy_task_com[:, 0::2] * 224 / 480)
                xy_task_com[:, 1::2] = (xy_task_com[:, 1::2] * 224 / 320)


            optimizer.zero_grad()
            bu2_log, seg_log = model(images, rand_task_one_hot)

            loss_bu2 = nn.CrossEntropyLoss(ignore_index=-1)(bu2_log, labels_com)
            if torch.any(labels_com >= 0):
                loss_seg = losses.SpatialClassificationLoss(seg_log[labels_com>=0], xy_task_com[labels_com>=0])

            loss = loss_bu2 + loss_seg

            loss.backward()
            optimizer.step()
            print(loss)

        # EVALUATION
        ############
        cur_acc1 = -1.0
        is_best = False
        if (epoch + 1) % args.n_eval == 0:
            model.eval()

            tot_loss = {}
            tot_loss['all'] = 0.0
            met = {}

            with torch.no_grad():

                for t in tasks:
                    tot_loss[t] = 0.0
                    met[t] = 0.0

                num_val_batches = 0
                for batch_val in val_loader:

                    val_images = batch_val[0].cuda()
                    labels_val = {}
                    for i, t in enumerate(tasks):
                        labels_val[t] = batch_val[1][:, i]
                        labels_val[t] = labels_val[t].cuda()


                    for i, t in enumerate(tasks):
                        rand_task = i * torch.ones(val_images.size(0), 1).long().cuda()
                        rand_task_one_hot = torch.FloatTensor(val_images.size(0), NUM_TASKS).cuda()
                        rand_task_one_hot.zero_()
                        rand_task_one_hot.scatter_(1, rand_task, 1)
                        rand_task_one_hot = rand_task_one_hot.cuda()

                        labels_val_com = torch.masked_select(
                            torch.stack([lab_line for lab_line in labels_val.values()]).transpose(0, 1),
                            rand_task_one_hot.byte()).view(val_images.size(0), -1).long().squeeze()

                        bu2_log, _ = model(val_images, rand_task_one_hot)

                        loss_bu2 = nn.CrossEntropyLoss(ignore_index=-1)(bu2_log, labels_val_com)
                        loss_t = loss_bu2
                        tot_loss['all'] += loss_t
                        tot_loss[t] += loss_t

                        if torch.any(labels_val_com >= 0):
                            metric[t].update(bu2_log[labels_val_com >= 0], labels_val_com[labels_val_com >= 0])
                    num_val_batches += 1

                metric_results_list = []
                for t in tasks:
                    metric_results = metric[t].get_result()
                    print(t, metric_results)
                    metric_results_list.append(metric_results['acc'])
                    metric[t].reset()

                cur_acc1 = float(torch.mean(torch.FloatTensor(metric_results_list)))
                is_best = cur_acc1 > best_acc1
                best_acc1 = max(cur_acc1, best_acc1)
                print(cur_acc1, best_acc1)

        state = {'epoch': epoch + 1,
                 'model_rep': model.state_dict(),
                 'optimizer_state': optimizer.state_dict()}
        state['cur_acc'] = cur_acc1  # metric_results_list
        state['best_acc'] = best_acc1

        torch.save(state, "saved_models/{}_model.pkl".format(params['exp_id']))

        if is_best:
            shutil.copyfile("saved_models/{}_model.pkl".format(params['exp_id']),
                            "saved_models/{}_model_best.pkl".format(params['exp_id']))


    #'''
    #----
    print('evaluating model')

    model.eval()

    tot_loss = {}
    tot_loss['all'] = 0.0
    met = {}

    with torch.no_grad():

        for t in tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        for batch_val in val_loader:

            val_images = batch_val[0].cuda()
            labels_val = {}
            for i, t in enumerate(tasks):
                labels_val[t] = batch_val[1][:, i]
                labels_val[t] = labels_val[t].cuda()

            for i, t in enumerate(tasks):
                rand_task = i * torch.ones(val_images.size(0), 1).long().cuda()
                rand_task_one_hot = torch.FloatTensor(val_images.size(0), NUM_TASKS).cuda()
                rand_task_one_hot.zero_()
                rand_task_one_hot.scatter_(1, rand_task, 1)
                rand_task_one_hot = rand_task_one_hot.cuda()

                labels_val_com = torch.masked_select(
                    torch.stack([lab_line for lab_line in labels_val.values()]).transpose(0, 1),
                    rand_task_one_hot.byte()).view(val_images.size(0), -1).long().squeeze()
                # labels_val_com[labels_val_com >= 0] = labels_val_com[labels_val_com >= 0] - 1

                bu2_log, _ = model(val_images, rand_task_one_hot)

                loss_bu2 = nn.CrossEntropyLoss(ignore_index=-1)(bu2_log, labels_val_com)
                loss_t = loss_bu2  # + loss_seg
                tot_loss['all'] += loss_t
                tot_loss[t] += loss_t

                if torch.any(labels_val_com >= 0):
                    metric[t].update(bu2_log[labels_val_com >= 0], labels_val_com[labels_val_com >= 0])
            num_val_batches += 1

        metric_results_list = []
        for t in tasks:
            metric_results = metric[t].get_result()
            print(t, metric_results)
            metric_results_list.append(metric_results['acc'])
            metric[t].reset()

        cur_acc1 = float(
            torch.mean(torch.FloatTensor(metric_results_list)))  # torch.mean(torch.stack(metric_results_list))
        print(cur_acc1)


    #'''

if __name__ == '__main__':
    train_multi_task_counter_stream()
