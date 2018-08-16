import torch
import numpy as np 
import os
import argparse
from models.posenet import PoseNet
import torch.backends.cudnn as cudnn
import data.dataloader as dataload
from torch.nn import DataParallel

#training parameters and options#

def options():
    opts=argparse.ArgumentParser()
    
    opts.add_argument('-c','--continue_exp',type=bool, default=False, help='continue the experiment')
    opts.add_argument('-e','--exp',     type=str,          default='pose',help='begin a new experiment')
    ##
    opts.add_argument('-nstack',         '--nstack',    type=int,  default=4,   help='the number of hourglass modules')
    opts.add_argument('-inp_dim',       '--inp_dim',    type=int,  default=256, help='the channels of the input for hourglass')
    opts.add_argument('-oup_dim',       '--oup_dim',    type=int,  default=68,  help='the channels of the output for prediction')
    opts.add_argument('-lr',                 '--lr',    type=float,default=2e-4,help='learning rate')
    opts.add_argument('-batchsize',   '--batchsize',    type=int,  default=32  ,help='batchsize')
    opts.add_argument('-input_res',   '--input_res',    type=int,  default=512 ,help='the resolution size of input')
    opts.add_argument('-output_res', '--output_res',    type=int,  default=128 ,help='the resolution size of output')
    opts.add_argument('-num_workers','--num_workers',   type=int,  default=2  , help='number of workers')
    opts.add_argument('-train_iters','--train_iters',   type=int,  default=1000,help='')
    opts.add_argument('-valid_iters','--valid_iters',   type=int,  default=10  ,help='')

    return opts.parse_args()

def adjust_lr(optimizer, epoch, gamma=0.9):
    schedule = list(range(3,32,2))
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']

class Model_Checkpoints:
    def __init__(self,options):
        self.opts=options

    def save_checkpoints(self,state,checkpoint_dir=None,filename='checkpoint.pth.tar'):
        '''
        param 'state' saves the state of the model parameters, example as :
        state={ 'epoch':      epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict(),}
        '''
        opts=self.opts
        continue_exp=opt.continue_exp
        if not continue_exp:
            if not os.path.exists('checkpoint/'+opts.exp):
                os.makedirs('checkpoint/'+opts.exp)
            checkpoint_dir='checkpoint/'+opts.exp
        else:
            checkpoint_dir='checkpoint/'+ continue_exp
        filename = 'epoch'+str(state['epoch']) + filename
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(state, filepath)
    
    def load_checkpoints(self,train_net,train_optimizer):
        '''
        load the checkpoint , return the epoch of last saves
        '''
        opts=self.opts
        if opts.continue_exp:
            resume = os.path.join('checkpoint', opts.continue_exp)
            f_list = os.listdir(resume)
            resume_file={'resume_file':i  for i in f_list if os.path.splitext(i)[-1] == '.tar'}['resume_file']
            if os.path.isfile(resume_file):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume_file)
                train_net.load_state_dict(checkpoint['state_dict'])
                train_optimizer.load_state_dict(checkpoint['optimizer'])
                begin_epoch = checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))
                exit(0)
        return begin_epoch

def train_func(opts,model,phase,**inputs):
    
    for i in inputs:
        inputs[i] = make_input(inputs[i])

    if phase == 'train':
        net = model.train()
    else:
        net = model.eval()

    forward_net=DataParallel(net.cuda())

    if phase !='inference':
        
        output = forward_net(inputs['imgs'])    
        
        losses = model.calc_loss(output,**{i:inputs[i] for i in inputs if i!='imgs'})
        losses ={'push_loss':losses[0]*opts.push_loss,'pull_loss':losses[1]*opts.pull_loss,'detection_loss':losses[2]*opts.detection_loss}
        loss = 0
            
        for i in losses:
            loss = loss + torch.mean(losses[i])

            my_loss = make_output( losses[i] )
            my_loss = my_loss.mean(axis = 0)

            if my_loss.size == 1:
                toprint += ' {}: {}'.format(i, format(my_loss.mean(), '.8f'))
            else:
                toprint += '\n{}'.format(i)
                for j in my_loss:
                    toprint += ' {}'.format(format(j.mean(), '.8f'))

        if phase == 'train':
            optimizer = train_cfg['optimizer']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return None
    else:
        out = {}
        net = net.eval()
        result = net(**inputs)
        if type(result)!=list and type(result)!=tuple:
            result = [result]
        out['preds'] = [make_output(i) for i in result]
        return out

def main():
    cudnn.benchmark = True
    cudnn.enabled = True

    opts=options()
    continue_exp=opts.continue_exp

    model=PoseNet(nstack=opts.nstack,inp_dim=opts.inp_dim,oup_dim=opts.oup_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    ##train datas and valid datas loader generator##
    data_load_func=dataload.init(opts)

    begin_epoch=0
    #choose whether continue the specified experiment checkpoint that was saved last time or not#
    if continue_exp:
        begin_epoch=Model_Checkpoints(opts).load_checkpoints(model,optimizer)
        print('Start training # epoch{}'.format(begin_epoch))

    for epoch in range(begin_epoch,500):
        print ('-------------Training Epoch {}-------------'.format(epoch))
        lr = adjust_lr(optimizer, epoch)

        #training and validation
        for phase in ['train', 'valid']:
            num_step = opts['{}_iters'.format(phase)]
            generator = data_load_func(phase)
            print('start', phase)

            show_range = range(num_step)
            show_range = tqdm.tqdm(show_range, total = num_step, ascii=True)
            
            for i in show_range:
                datas = next(generator)
                outs = train_func(opts, phase, **datas)

        Model_Checkpoints.save_checkpoints({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),})

if __name__ == "__main__":
    main()