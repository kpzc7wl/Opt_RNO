#!/usr/bin/env python
#-*- coding:utf-8 _*-
import sys
import os
sys.path.append('../..')
sys.path.append('..')


import re
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import dgl


from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

from args import get_args
from data_utils import get_dataset, get_model, get_loss_func, \
         MIODataLoader, RefDataLoader
from utils import get_seed, get_num_params, MultipleTensors, plot_ref_query_error, shape_obj_batch, chunk_as
from models.optimizer import Adam, AdamW
# from models.cgpt_geo import GNO_H



'''
    A general code framework for training neural operator on irregular domains
'''

EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']






def train(model, loss_func, metric_func,
              train_loader, valid_loader,
              optimizer, lr_scheduler,
              epochs=10,
              writer=None,
              device="cuda",
              patience=10,
              grad_clip=0.999,
              sobolev=False,
              lamb=0.3,
              no_ref=False,
              start_epoch: int = 0,
              print_freq: int = 20,
              model_save_path='./data/checkpoints/',
              save_mode='state_dict',  # 'state_dict' or 'entire'
              model_name='model.pt',
              result_name='result.pt'):
    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    result = None
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = np.inf
    best_val_epoch = None
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__)for s in EPOCH_SCHEDULERS)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        torch.cuda.empty_cache()
        for batch in train_loader:

            loss = train_batch(model, loss_func, batch, optimizer, lr_scheduler, device, 
                               x_norm=train_loader.dataset.x_normalizer, 
                               y_norm=train_loader.dataset.y_normalizer, 
                               sobolev=sobolev, lamb=lamb, no_ref=no_ref,
                               grad_clip=grad_clip)

            loss = np.array(loss)
            loss_epoch.append(loss)
            it += 1
            lr = optimizer.param_groups[0]['lr']
            lr_history.append(lr)
            log = f"epoch: [{epoch+1}/{end_epoch}]"
            if loss.ndim == 0:  # 1 target loss
                _loss_mean = np.mean(loss_epoch)
                log += " loss: {:.6f}".format(_loss_mean)
            else:
                _loss_mean = np.mean(loss_epoch, axis=0)
                for j in range(len(_loss_mean)):
                    log += " | loss {}: {:.6f}".format(j, _loss_mean[j])
            log += " | current lr: {:.3e}".format(lr)

            if it % print_freq==0:
                print(log)

            if writer is not None:
                for j in range(len(_loss_mean)):
                    writer.add_scalar("train_loss_{}".format(j),_loss_mean[j], it)    #### loss 0 seems to be the sum of all loss



        loss_train.append(_loss_mean)
        loss_epoch = []

        val_result = validate_epoch(model, metric_func, valid_loader, device, sobolev=sobolev, 
                                    x_norm=train_loader.dataset.x_normalizer, 
                                    y_norm=train_loader.dataset.y_normalizer)

        loss_val.append(val_result["metric"])
        val_metric = val_result["metric"].sum()


        if val_metric < best_val_metric:
            best_val_epoch = epoch
            best_val_metric = val_metric

            checkpoint = {'args':args, 'model':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(checkpoint, os.path.join('./data/checkpoints/{}'.format(model_name)))


        if lr_scheduler and is_epoch_scheduler:
            if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                lr_scheduler.step(val_metric)
            else:
                lr_scheduler.step()


        if val_result["metric"].size == 1:
            log = "| val metric 0: {:.6f} ".format(val_metric)

        else:
            log = ''
            for i, metric_i in enumerate(val_result['metric']):
                log += '| val metric {} : {:.6f} '.format(i, metric_i)

        if writer is not None:
            if val_result["metric"].size == 1:
                writer.add_scalar('val loss {}'.format(metric_func.component),val_metric, epoch)
            else:
                for i, metric_i in enumerate(val_result['metric']):
                    writer.add_scalar('val loss {}'.format(i), metric_i, epoch)

        print(log)
        log += "| best val: {:.6f} at epoch {} | current lr: {:.3e}".format(best_val_metric, best_val_epoch+1, lr)

        desc_ep = ""
        if _loss_mean.ndim == 0:  # 1 target loss
            desc_ep += "| loss: {:.6f}".format(_loss_mean)
        else:
            for j in range(len(_loss_mean)):
                if _loss_mean[j] > 0:
                    desc_ep += "| loss {}: {:.3e}".format(j, _loss_mean[j])

        desc_ep += log
        print(desc_ep)

        result = dict(
            best_val_epoch=best_val_epoch,
            best_val_metric=best_val_metric,
            loss_train=np.asarray(loss_train),
            loss_val=np.asarray(loss_val),
            lr_history=np.asarray(lr_history),
            # best_model=best_model_state_dict,
            optimizer_state=optimizer.state_dict()
        )
        pickle.dump(result, open(os.path.join(model_save_path, result_name),'wb'))
    return result


# normalize sensitivity
def normalize_sens_batch(sens, gs): 
    xs = [g.ndata['x'] for g in gs]   
    ss = chunk_as(sens, xs)
    ss = [s / (s.norm(p=2, dim=1, keepdim=True) + 1e-8) for s in ss]
    # print('normalized sens: ', [s.norm(p=2) for s in ss])
    return torch.cat(ss, dim=0)
    

def train_batch(model, loss_func, data, optimizer, lr_scheduler, device, x_norm=None, y_norm=None, sobolev=False, lamb=0.3, grad_clip=0.999, no_ref=0.3):
    optimizer.zero_grad()

    g, g_r, u_p, g_u = data
    g, g_r, u_p, g_u = g.to(device), g_r.to(device), u_p.to(device), g_u.to(device)
    gs, gs_r = dgl.unbatch(g), dgl.unbatch(g_r)
    if sobolev:
        for g_ in gs:
            x_ = g_.ndata['x']
            x_.requires_grad = True

    y_pred, y_ref = model(gs, gs_r, u_p, g_u, no_ref=no_ref) 
    y_pred, y = y_pred.squeeze(), g.ndata['y'].squeeze()

    dim_sens = y.shape[1] - y_pred.shape[1] # dim of sensitivity
    if dim_sens > 0:
        sens = y[:, -dim_sens:]
        y_pred = torch.cat([y_pred, sens], dim=-1)
        if sobolev:
            dJdx = shape_obj_batch(gs, y_pred, x_norm.to(device), y_norm.to(device))
            dJdx = torch.cat(dJdx, dim=0)
            assert sens.shape == dJdx.shape, f'sens.shape{sens.shape} and dJdx.shape{dJdx.shape} not matching.'
            # domain_mask = torch.cat([g_.ndata['x'][:, -1, None].bool() for g_ in gs])
            # dJdx = dJdx * domain_mask   
            sens = normalize_sens_batch(sens, gs) # normalize sensitivity
            # print('dJdx has nan: ', dJdx.isnan().any())
            y = torch.cat([y[:, :-dim_sens], sens], dim=-1)
            y_pred = torch.cat([y_pred[:, :-dim_sens], dJdx], dim=-1)            
            
    losses, reg, _ = loss_func(g, y_pred, y)
    # print(losses)
    loss = losses.mean()
    if dim_sens > 0:
        loss = losses[:-dim_sens].mean()

    loss_total = loss + reg
    if dim_sens > 0:
        loss_grad = losses[-dim_sens:].mean()
        # if loss_grad.item() < 0.1: lamb = 0
        loss_total = loss_total + lamb * loss_grad
    # print(loss_total, loss_grad)

    
    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


    if lr_scheduler:
        lr_scheduler.step()

    if sobolev: return (loss.item(), loss_grad.item())
    return (loss.item(), reg.item())



def validate_epoch(model, metric_func, valid_loader, device, sobolev=False, save_fig=None, x_norm=None, y_norm=None, no_ref=0.0):
    model.eval()
    metric_val = []
    # metric_interp_error_val = []
    metric_interp_difference_val = []
    dist = []
    for i, data in enumerate(valid_loader):
        # print(i)
        # with torch.enable_grad():
        with torch.no_grad():
            g, g_r, u_p, g_u = data
            g, g_r, u_p, g_u = g.to(device), g_r.to(device), u_p.to(device), g_u.to(device)
            gs, gs_r = dgl.unbatch(g), dgl.unbatch(g_r)
            if sobolev:
                for g_ in gs:
                    x_ = g_.ndata['x']
                    x_.requires_grad = True
                with torch.enable_grad():
                    y_pred, y_ref = model(gs, gs_r, u_p, g_u, no_ref=no_ref)
                    y_pred, y = y_pred.squeeze(), g.ndata['y'].squeeze()    
            else:
                y_pred, y_ref = model(gs, gs_r, u_p, g_u, no_ref=no_ref)
                y_pred, y = y_pred.squeeze(), g.ndata['y'].squeeze()       
            

            dim_sens = y.shape[1] - y_pred.shape[1] # dim of sensitivity
            if dim_sens > 0:
                sens = y[:, -dim_sens:]
                
                if sobolev:
                    with torch.enable_grad():
                        y_pred = torch.cat([y_pred, sens], dim=-1)
                        dJdx = shape_obj_batch(gs, y_pred, x_norm.to(device), y_norm.to(device), create_graph=False, retain_graph=False) 
                        dJdx = torch.cat(dJdx, dim=0)
                        assert sens.shape == dJdx.shape, f'sens.shape={sens.shape} and dJdx.shape={dJdx.shape} not matching.'
                        # domain_mask = torch.cat([g_.ndata['x'][:, -1, None].bool() for g_ in gs])
                        # dJdx = dJdx * domain_mask  
                        # sens = sens / sens.reshape(-1).norm(p=2) 
                        sens = normalize_sens_batch(sens, gs) # normalize sensitivity
                        y = torch.cat([y[:, :-dim_sens], sens], dim=-1)
                        y_pred = torch.cat([y_pred[:, :-dim_sens], dJdx], dim=-1)
                else:
                    y_pred = torch.cat([y_pred, sens], dim=-1)
                    
                    
            _, _, metrics = metric_func(g, y_pred, y)
            
            metric_val.append(metrics)
        # break
    
    # if save_fig:
    if False:
        print('Saving plots...')
        plot_ref_query_error(g.to('cpu'), g_u.to('cpu'), 
                             y_interp.cpu(), y_pred.detach().clone().cpu(), 
                            #  y_interp_pred.detach().clone().cpu(), 
                            save_path=save_fig
                             )
        np.save('Error_dist.npy',
                {
                    'interp_diff': np.array(metric_interp_difference_val),
                    'pred_error': np.array(metric_val),
                    'distance': np.array(dist)
                }
                )

    return dict(metric=np.mean(metric_val, axis=0))    



if __name__ == "__main__":
    args = get_args()
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)


    train_dataset, test_dataset = get_dataset(args)
    # test_dataset = get_dataset(args)

    train_loader = RefDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = RefDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    args.space_dim = int(re.search(r'\d', args.dataset).group())
    args.normalizer =  train_dataset.y_normalizer.to(device) if train_dataset.y_normalizer is not None else None

    #### set random seeds
    get_seed(args.seed)
    torch.cuda.empty_cache()

    loss_func = get_loss_func(name=args.loss_name,args= args, regularizer=True,normalizer=args.normalizer)
    metric_func = get_loss_func(name='rel2', args=args, regularizer=False, normalizer=args.normalizer)

    model = get_model(args)
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")


    path_prefix = args.dataset  + '_{}_'.format(args.component) + model.__name__ + args.comment + time.strftime('_%m%d_%H_%M_%S')
    model_path, result_path = path_prefix + '.pt', path_prefix + '.pkl'

    print(f"Saving model and result in ./../models/checkpoints/{model_path}\n")


    if args.use_tb:
        writer_path =  './data/logs/' + path_prefix
        log_path = writer_path + '/params.txt'
        writer = SummaryWriter(log_dir=writer_path)
        fp = open(log_path, "w+")
        sys.stdout = fp

    else:
        writer = None
        log_path = None


    # print(model)
    # print(config)

    epochs = args.epochs
    lr = args.lr


    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay,betas=(0.9,0.999))
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay,betas=(0.9, 0.999))
    else:
        raise NotImplementedError



    if args.lr_method == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, pct_start=0.2, final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=epochs)
    elif args.lr_method == 'step':
        print('Using step learning rate schedule')
        scheduler = StepLR(optimizer, step_size=args.lr_step_size*len(train_loader), gamma=0.7)
    elif args.lr_method == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = LambdaLR(optimizer, lambda steps: min((steps+1)/(args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader)/float(steps + 1), 0.5)))


    time_start = time.time()

    result = train(model, loss_func, metric_func,
                       train_loader, test_loader,
                       optimizer, scheduler,
                       epochs=epochs,
                       grad_clip=args.grad_clip,
                       sobolev=args.sobolev,
                       no_ref=args.noref,
                       lamb=args.lamb,
                       patience=None,
                       model_name=model_path,
                       model_save_path='./data/checkpoints/',
                       result_name=result_path,
                       writer=writer,
                       device=device)

    print('Training takes {} seconds.'.format(time.time() - time_start))

    # result['args'], result['config'] = args, config
    checkpoint = {'args':args, 'model':model.state_dict(),'optimizer':optimizer.state_dict()}
    torch.save(checkpoint, os.path.join('./data/checkpoints/{}'.format(model_path)))
    model.eval()
    val_metric = validate_epoch(model, metric_func, test_loader, device, sobolev=args.sobolev, 
                                x_norm=train_loader.dataset.x_normalizer, 
                                y_norm=train_loader.dataset.y_normalizer,
                                save_fig='fig/{}'.format(model_path))
    print(f"\nBest model's validation metric in this run: {val_metric}")




