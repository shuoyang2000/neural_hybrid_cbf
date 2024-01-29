import torch
import diff_operators


def initialize_hji_acc(dataset):
    # Initialize the loss function for the adaptive cruise control
    # The dynamics parameters
    c_dry = 0.3 # 0.3
    m = 1650
    g = 9.81
    c_0 = 0.3
    c_1 = 15
    c_2 = 0.75
    v0 = 14

    ## aining parameter
    time_scale = 25.0  #value of tMin
    cor_mean = dataset.mean
    cor_width = dataset.width

    def hji_acc(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4), x is the transoformed state
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]


        du, status = diff_operators.jacobian(y, x) # y is the value function, x is the system state with time
        dudt = du[..., 0, 0] # dy/dt
        dudx = du[..., 0, 1:]  # dy/dp dy/dv dy/dd
        x_v = x[..., 2] * cor_width[1] + cor_mean[1] # original velocity

        # Compute the hamiltonian for the ego vehicle
        ham = c_dry*g/cor_width[1] * torch.abs(dudx[..., 1])  # Control component
        ham = ham + dudx[..., 0]/cor_width[0] * x_v -1/m/cor_width[1]*(c_0*x_v*x_v+c_1* x_v+c_2)*dudx[..., 1]+ (v0-x_v)/cor_width[2]*dudx[..., 2]  # Constant component
        ham = ham*time_scale

        if torch.all(dirichlet_mask):  ##pretrain stage which only trains for t=tMax
            diff_constraint_hom = torch.Tensor([0])
        else:   ##stage which consider loss for time except t=tMax
            diff_constraint_hom = -dudt[:,:,None] - ham[:,:, None]
            diff_constraint_hom = torch.max(diff_constraint_hom, y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]  ## the loss term for t=tMax
        loss_tradeoff_factor = batch_size / 15e2  ## trade-off factor
        return {'dirichlet': torch.abs(dirichlet).sum() * loss_tradeoff_factor, 'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_acc


def initialize_hji_gym(dataset):
    # Initialize the loss function for the gym simulator
    # The dynamics parameters
    params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74,
              'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_max': 3.2,
              'a_max': 9.51}
    mu = params['mu']
    C_Sf = params['C_Sf']
    C_Sr = params['C_Sr']
    lf = params['lf']
    lr = params['lr']
    h = params['h']
    m = params['m']
    I = params['I']
    sv_max = params['sv_max']
    a_max = params['a_max']
    g = 9.81

    ## aining parameter
    time_scale = 20.0  ## value of tMin
    cor_mean = dataset.mean
    cor_width = dataset.width


    def hji_gym(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 8), x is the transoformed state
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x) # y is the value function, x is the system state with time
        dudt = du[..., 0, 0] #  dev for time
        dudx = du[..., 0, 1:] # dev for system state

        ## transform to origin coordinate
        x_2 = x[..., 3] * cor_width[2] + cor_mean[2]
        x_3 = x[..., 4] * cor_width[3] + cor_mean[3]
        x_4 = x[..., 5] * cor_width[4] + cor_mean[4]
        x_5 = x[..., 6] * cor_width[5] + cor_mean[5]
        x_6 = x[..., 7] * cor_width[6] + cor_mean[6]

        ## compute dev of x_6 and x_7 in control-affine formulation
        fx_6_f = -mu * m / (x_3 * I * (lr + lf)) * (lf ** 2 * C_Sf * (g * lr) + lr ** 2 * C_Sr * (g * lf)) * x_5 \
                 + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf) - lf * C_Sf * (g * lr)) * x_6 \
                 + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr) * x_2
        fx_6_g = -mu * m / (x_3 * I * (lr + lf)) * (lf ** 2 * C_Sf * (-h) + lr ** 2 * C_Sr * (h)) * x_5 \
                 + mu * m / (I * (lr + lf)) * (lr * C_Sr * (h) - lf * C_Sf * (-h)) * x_6 \
                 + mu * m / (I * (lr + lf)) * lf * C_Sf * (-h) * x_2
        fx_7_f = (mu / (x_3 ** 2 * (lr + lf)) * (C_Sr * (g * lf) * lr - C_Sf * (g * lr) * lf) - 1) * x_5 \
                 - mu / (x_3 * (lr + lf)) * (C_Sr * (g * lf) + C_Sf * (g * lr)) * x_6 \
                 + mu / (x_3 * (lr + lf)) * (C_Sf * (g * lr)) * x_2
        fx_7_g = (mu / (x_3 ** 2 * (lr + lf)) * (C_Sr * (h) * lr - C_Sf * (-h) * lf)) * x_5 \
                 - mu / (x_3 * (lr + lf)) * (C_Sr * (h) + C_Sf * (-h)) * x_6 \
                 + mu / (x_3 * (lr + lf)) * (C_Sf * (-h)) * x_2

        # Compute the hamiltonian for the ego vehicle
        # Control component
        ham = sv_max * torch.abs(dudx[..., 2]) / cor_width[2]
        ham += a_max * torch.abs(dudx[..., 3]) / cor_width[3]
        ham += a_max * torch.abs(dudx[..., 5] * fx_6_g)/cor_width[5]
        ham += a_max * torch.abs(dudx[..., 6] * fx_7_g)/cor_width[6]

        # Constant component
        ham = ham + x_3 * torch.cos(x_6 + x_4) * dudx[..., 0] / cor_width[0]
        ham = ham + x_3 * torch.sin(x_6 + x_4) *dudx[..., 1] / cor_width[1]
        ham = ham + x_5 * dudx[..., 4] / cor_width[4]
        ham = ham + fx_6_f * dudx[..., 5] / cor_width[5]
        ham = ham + fx_7_f * dudx[..., 6] / cor_width[6]
        ham = ham * time_scale

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = - dudt[:, :, None]-ham[:, :, None]
            diff_constraint_hom = torch.max(diff_constraint_hom, y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
        loss_tradeoff_factor = batch_size / 15e2
        dirichlet_loss = torch.abs(dirichlet).sum() * loss_tradeoff_factor
        return {'dirichlet': dirichlet_loss, 'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_gym
