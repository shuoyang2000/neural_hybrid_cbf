# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, utils.modules as modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./train/logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=-1.1, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.0, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_end', type=int, default=10000, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--numpoints', type=int, default=65000, required=False, help='Number of samples at each time step')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
p.add_argument('--use_cuda', default=True, type = bool, help='Use GPU or not.')
p.add_argument('--network_scale_mean', type=float, default=-20.0, required=False, help='neural network scale mean')
p.add_argument('--network_scale_factor', type=float, default=20.0, required=False, help='neural network scale factor')
p.add_argument('--state_dim', type=int, default=7, required=False, help='state dimension of the system')
p.add_argument('--left_right_wall_flag', type=int, default=1, help='flag variable to decide train left wall or right wall')
opt = p.parse_args()

print("Training begins!")

dataset = dataio.ReachabilityGymSource(numpoints=opt.numpoints, pretrain=opt.pretrain, tMin=opt.tMin, network_mean=opt.network_scale_mean,
                                          network_factor=opt.network_scale_factor, use_cuda=opt.use_cuda, state_dim=opt.state_dim,
                                          tMax=opt.tMax, counter_end=opt.counter_end,
                                          pretrain_iters=opt.pretrain_iters,
                                          num_src_samples=opt.num_src_samples, left_right_wall_flag=opt.left_right_wall_flag)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=1+opt.state_dim, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
if opt.use_cuda:
  model.cuda()


# Define the loss
loss_fn = loss_functions.initialize_hji_gym(dataset)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

def val_fn(model, ckpt_dir, epoch):
  # Time values at which the function needs to be plotted
  times = [opt.tMin + 0.1, - 0.75 , - 0.5, - 0.25 , opt.tMax]
  num_times = len(times)

  # position slices to be plotted
  thetas = [-0.9,  0.,  0.9]
  num_thetas = len(thetas)

  # Create a figure
  fig = plt.figure(figsize=(5*num_times,5*num_thetas))

  # Get the meshgrid in the (x, y) coordinate
  sidelen = 200
  mgrid_coords = dataio.get_mgrid(sidelen)

  # Start plotting the results
  for i in range(num_thetas):
    theta_coords = torch.ones(mgrid_coords.shape[0], 1) * thetas[i]

    for j in range(num_times):
      time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[j]
      coords = torch.cat((time_coords, mgrid_coords, torch.zeros(mgrid_coords.shape[0], 1), theta_coords , torch.zeros(mgrid_coords.shape[0], 1), torch.zeros(mgrid_coords.shape[0], 1), torch.zeros(mgrid_coords.shape[0], 1)), dim=1)
      if opt.use_cuda:
        model_in = {'coords': coords.cuda()}
      else:
        model_in = {'coords': coords}
      model_out = model(model_in)['model_out']

      # Detatch model ouput and reshape
      model_out = model_out.detach().cpu().numpy()
      model_out = model_out.reshape((sidelen, sidelen))

      # Unnormalize the value function
      mean = opt.network_scale_mean
      normalization_factor = opt.network_scale_factor
      model_out = (model_out * normalization_factor) + mean

      # Plot the zero level sets
      model_out = (model_out <= 0.001) * 1.

      # Plot the actual data
      ax = fig.add_subplot(num_thetas, num_times,i*num_times+j+1)
      ax.set_title('theta = %0.2f, t = %0.2f' % (thetas[i], times[j]))
      s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
      fig.colorbar(s) 

  fig.savefig(os.path.join(ckpt_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch))
  
training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr, use_cuda = opt.use_cuda,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, validation_fn=val_fn, start_epoch=opt.checkpoint_toload)
