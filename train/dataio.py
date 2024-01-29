import numpy as np
import torch
from torch.utils.data import Dataset
import modules
from scipy.interpolate import interpn
import random
from utils.frenet_utils import cartesian_to_frenet
from utils.cubic_spline import CubicSpline2D


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords



class ReachabilityACCSource(Dataset):
    def __init__(self, numpoints, network_mean, network_factor, use_cuda, state_dim,
        pretrain=False, tMin=0.0, tMax=0, counter_end=100e3, dataset_dir = "train/train_data/", first = np.arange(0, 205, 5), second = np.arange(11.5, 33.4, 0.3), third = np.arange(0, 101, 1),
        pretrain_iters=2000, num_src_samples=1000,lowbound = [0., 11.8, 0.],upbound = [200.0, 32.8, 100.0], switch_sample_number = 25000, switch_sample_number_initial_time = 5000):
        super().__init__()

        torch.manual_seed(seed = 0)
        self.pretrain = pretrain  ## Flag for whether pretrain
        self.numpoints = numpoints  ## sample number
        self.num_states = state_dim  ## state dimension
        self.tMax = tMax
        self.tMin = tMin
        assert self.tMax == 0, "Please confirm if tMax is zero!"
        assert self.tMin < 0, "Please confirm if tMin is nageative!"
        self.N_src_samples = num_src_samples  #sample number for t=tMax
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters #iteration number of pretrain
        self.counter = 0
        self.full_count = counter_end # number of step for curriculum training to reach t=tMin
        self.use_cuda =use_cuda
        self.points = (first,second,third)  #para for function interpn
        self.HJR_dry_table = np.load(dataset_dir + 'acc_HJR_dry_boundary_value.npy')  ##computed value for dry condition
        self.HJR_ice_table = np.load(dataset_dir + 'acc_HJR_ice_boundary_value.npy')  ##computed value for ice condition
        self.network_mean = network_mean  ##NN scale coe
        self.network_factor = network_factor ##NN scale code
        self.lowbound = lowbound  ## low bound for state p v d
        self.upbound = upbound ## up bound for state p v d
        self.mean = []
        self.width = []
        for i in range(len(self.lowbound)):
            self.mean.append((self.lowbound[i] + self.upbound[i]) / 2.0)
            self.width.append((-self.lowbound[i] + self.upbound[i]) / 2.0)
        self.switch_sample_number = switch_sample_number    ## the least sample number near switching region
        self.switch_sample_number_initial_time = switch_sample_number_initial_time ## the least sample number near switching region at initial time
    def __len__(self):
        return 1


    def __getitem__(self, idx):
        start_time = 0.  # time to apply initial conditions
        # uniformly sample domain
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
            pos = torch.zeros(self.switch_sample_number, 1).uniform_(-0.012,0.012)
            coords[0:self.switch_sample_number, 1]= pos[0:self.switch_sample_number,0]
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0(tMax) and min time value is tMin
            time = - torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1) # concatenate time and states

            # make sure we always have training samples at the t=tMax
            coords[-self.N_src_samples:, 0] = start_time

            # make sure we always have training samples near swithing region for t=tMax and other time
            pos = torch.zeros(self.switch_sample_number, 1).uniform_(-0.012,0.012)
            coords[-self.switch_sample_number_initial_time:, 1]= pos[-self.switch_sample_number_initial_time:,0]
            coords[0:self.switch_sample_number-self.switch_sample_number_initial_time, 1]= pos[0:self.switch_sample_number-self.switch_sample_number_initial_time,0]

        # set up the boundary value function
        real_data = coords[:,1:4] * torch.tensor(self.width)+ torch.tensor(self.mean)  ## transform to origin rdinate
        dist_fun = torch.abs(real_data[:, 0, None]-100)-1  ## <=0 when state is in switching region
        real_data = real_data.numpy()
        ice_outcome = interpn(self.points,self.HJR_ice_table[0],real_data)  ## compute ice value by insert function
        ice_outcome = torch.unsqueeze(torch.tensor(ice_outcome),1)
        dry_outcome = interpn(self.points, self.HJR_dry_table[0], real_data)  ## compute dry value by insert function
        dry_outcome = torch.unsqueeze(torch.tensor(dry_outcome), 1)
        boundary_values = torch.min(dry_outcome, torch.max(dist_fun, ice_outcome*dry_outcome))  ##boundary value is a mixture of ice condition, dry condition and switching function
        boundary_values = (boundary_values - self.network_mean) / self.network_factor  ## scale the boundary value
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class ReachabilityGymSource(Dataset):
    def __init__(self, numpoints, network_mean, network_factor, use_cuda, state_dim, left_right_wall_flag,
                 pretrain=False, tMin=0.0, tMax=0, counter_start=0, counter_end=100e3, track_halfw=1.5, dataset_dir = "train/train_data/",
                 pretrain_iters=2000, num_src_samples=1000, safety_margin = 0.58, max_brake_lat_dry = 500.0, max_brake_lat_ice = 0.01,
                 lowbound = [0.0, -1.0, -0.21, 2.5, -1.5, -5.0, -0.2], upbound = [16.0, 1.0, 0.21, 7.5,  1.5, 5.0, 0.2]):
        super().__init__()

        torch.manual_seed(seed=0)
        self.pretrain = pretrain
        self.numpoints = numpoints
        self.num_states = state_dim
        self.tMax = tMax
        self.tMin = tMin
        assert self.tMax == 0, "Please confirm if tMax is zero!"
        assert self.tMin < 0, "Please confirm if tMin is nageative!"
        self.N_src_samples = num_src_samples
        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.use_cuda = use_cuda
        self.lowbound = lowbound ## lowbound of state; state is: s,d,yaw,vel, phi, dot phi, beta
        self.upbound = upbound
        self.mean = []
        self.width = []
        for i in range(len(self.lowbound)):
            self.mean.append((self.lowbound[i] + self.upbound[i]) / 2.0)
            self.width.append((-self.lowbound[i] + self.upbound[i]) / 2.0)
        self.track_halfw = track_halfw  ## safety parameter
        self.safety_margin = safety_margin  ## safety parameter
        self.max_brake_lat_dry = max_brake_lat_dry  ## safety parameter
        self.max_brake_lat_ice = max_brake_lat_ice  ## safety parameter
        self.table = np.load(dataset_dir+"race_boundary_value.npy") ## include uniform point for (s,d,yaw,curvature)
        self.table[:,0] = (self.table[:,0] - self.mean[0])/self.width[0]
        self.table[:,1] = (self.table[:, 1] - self.mean[1]) / self.width[1]
        self.table = torch.tensor(self.table)
        self.all = [i for i in range(self.table.shape[0])]
        self.choice = random.sample(self.all, self.numpoints)
        self.network_mean = network_mean
        self.network_factor = network_factor
        self.left_right_flag = left_right_wall_flag #1 will train left wall, 0 will train right wall


    def __len__(self):
        return 1


    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample points in table
        self.choice = random.sample(self.all, self.numpoints)
        coords = torch.cat((self.table[self.choice,0:2], torch.zeros(self.numpoints, 5).uniform_(-1, 1)), dim=1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(coords.shape[0], 1) * start_time
            coords = torch.cat((time, coords), dim=1)

        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0(tMax) and min time value is tMin
            time = - torch.zeros(coords.shape[0], 1).uniform_(0,
                                                             (self.tMax - self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)  # concatenate time and states
            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        coords = coords.to(torch.float32) # switch the data type

        # set up the initial value function
        true_data = coords[:, 1:8] * torch.tensor(self.width) + torch.tensor(self.mean)  ## transform to origin coordinate
        dey = true_data[:, 3] * torch.sin(true_data[:, 4] + true_data[:, 6] - self.table[self.choice, 2])  ## compute derivative of d(second dimension of frenet coordinate)
        if self.left_right_flag:  ## compute boundary value when training left wall
            dist_left = self.track_halfw - self.table[self.choice, 1]
            Dv_left = torch.clamp(- dey, -80.0 ,0.0)
            hx_left_ice = dist_left - (Dv_left ** 2) / (2*self.max_brake_lat_ice) - self.safety_margin
            hx_left_dry = dist_left - (Dv_left ** 2) / (2 * self.max_brake_lat_dry) - self.safety_margin
            hx_left = (true_data[:, 0]<=15.3)*hx_left_dry+(true_data[:, 0]>15.3)*hx_left_ice
            boundary_values = torch.unsqueeze(hx_left, 1)
        else: ## compute boundary value when training right wall
            dist_right = self.track_halfw + self.table[self.choice, 1]
            Dv_right = torch.clamp(dey, -80.0 ,0.0)
            hx_right_ice = dist_right - (Dv_right ** 2) / (2*self.max_brake_lat_ice) - self.safety_margin
            hx_right_dry = dist_right - (Dv_right ** 2) / (2 * self.max_brake_lat_dry) - self.safety_margin
            hx_right = torch.min(hx_right_dry,torch.max(hx_right_ice,25.0*(15.3-true_data[:, 0])))
            boundary_values = torch.unsqueeze(hx_right, 1)

        boundary_values = boundary_values.to(torch.float32)
        boundary_values = (boundary_values - self.network_mean) / self.network_factor ## scale the NN

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}