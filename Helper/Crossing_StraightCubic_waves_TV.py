# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import meshgrid
import torch.optim as optim
from scipy import sparse
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import gradcheck
from numpy import exp, mod, meshgrid, cos, sin, exp, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from math import sqrt
from scipy.sparse import diags
from scipy.linalg import cholesky

import os
import argparse

# -----------------------------------------------------------------------------
# NEW: allow a list of lambda_TV values from the command line
# Example:
#   python3 Crossing_StraightCubic_waves_TV.py --lambda_TV 0.1 1.0 10.0
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--lambda_TV",
    type=float,
    nargs="+",
    default=[1.0],
    help="List of lambda_TV values to sweep over"
)
args = parser.parse_args()
lambda_TV_list = args.lambda_TV

impath = "./data/StraightCubic_wave_TV/"
immpath = "./plots/StraightCubic_wave_TV/"
os.makedirs(impath, exist_ok=True)
os.makedirs(immpath, exist_ok=True)

"""# Crossing waves example

## Create data for example model
"""

seed = 51
np.random.seed(seed)
torch.manual_seed(seed)

dtype  = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_load = False
print(device)

Nx, Nt = 400, 200
t_start = -10.0
t_end = 10.0

x = np.arange(0, Nx)
t = np.linspace(t_start, t_end, Nt)
x_device = torch.tensor(x.copy(),  dtype=dtype, device=device)
t_device = torch.tensor(t.copy(),  dtype=dtype, device=device)
coefficients1 = torch.tensor([0.15,0,0.8,1.5], dtype=dtype, device=device)
coefficients2 = torch.tensor([-18,2], dtype=dtype, device=device)
sigma = torch.tensor(4.0, dtype=dtype, device=device)
center_matrix = torch.tensor(200.0, dtype=dtype, device=device)


@torch.jit.script
def torch_polyval(coeffs, t_array):
    results = torch.zeros_like(t_array)
    for i in range(t_array.shape[0]):
        t = t_array[i]
        result = 0.0
        for coeff in coeffs:
            result = result * t + coeff
        results[i] = result
    return results


@torch.jit.script
def torch_gaussian(x, mu, sigma):
    return torch.exp(-torch.pow(x - mu, 2.0) / (2 * torch.pow(sigma, 2.0)))


def central_FD2Matrix(N, h=1.0):
    D = sparse.lil_matrix((N, N), dtype=float)

    # interior: central second-order
    for i in range(1, N - 1):
        D[i, i - 1] = 1.0 / h**2
        D[i, i] = -2.0 / h**2
        D[i, i + 1] = 1.0 / h**2

    # first row: forward second-order derivative
    D[0, 0] = 2.0 / h**2
    D[0, 1] = -5.0 / h**2
    D[0, 2] = 4.0 / h**2
    D[0, 3] = -1.0 / h**2

    # last row: backward second-order derivative
    D[-1, -4] = -1.0 / h**2
    D[-1, -3] = 4.0 / h**2
    D[-1, -2] = -5.0 / h**2
    D[-1, -1] = 2.0 / h**2

    D_1 = (D.transpose()).tocsr()

    return D_1


def generate_data_crossing_wave(coefficients1, coefficients2, x, t, center_of_matrix, sigma):
    shift1 = torch_polyval(coefficients1, t)
    shift2 = torch_polyval(coefficients2, t)
    X1, MU1 = torch.meshgrid(x, center_of_matrix + shift1)
    X2, MU2 = torch.meshgrid(x, center_of_matrix + shift2)

    Q1 = torch_gaussian(X1, MU1, sigma)
    Q2 = torch_gaussian(X2, MU2, sigma)

    Q = Q1 + Q2

    return Q, Q1, Q2, shift1, shift2

Q, Q1, Q2, shift1, shift2 = generate_data_crossing_wave(coefficients1, coefficients2, x_device, t_device, center_matrix, sigma)


"""## Define a model"""

@torch.jit.script
def nuclear_norm(input_matrix: torch.Tensor) -> torch.Tensor:
    # Forward computation
    return torch.linalg.matrix_norm(input_matrix, ord="nuc")


@torch.jit.script
def nuclear_norm_grad(input_matrix: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    u, s, v = torch.linalg.svd(input_matrix, full_matrices=False)
    rank = (s > 0).sum().item()
    eye_approx = torch.diag_embed((s > 0).to(input_matrix.dtype)[:rank])
    grad_input = torch.matmul(u[:, :rank], eye_approx)
    grad_input = torch.matmul(grad_input, v[:, :rank].transpose(-2, -1))
    return grad_input * grad_output.unsqueeze(-1).unsqueeze(-1)


class NuclearNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix):
        ctx.save_for_backward(input_matrix)
        return nuclear_norm(input_matrix)

    @staticmethod
    def backward(ctx, grad_output):
        input_matrix, = ctx.saved_tensors
        return nuclear_norm_grad(input_matrix, grad_output)

class ShapeShiftNet(nn.Module):
    def __init__(self, N_in, N_in_c, N_out, N_out_c, N_hidden, N_layers, x, t, center_matrix):
        super(ShapeShiftNet, self).__init__()

        self.register_buffer('x_flat', x)
        self.register_buffer('t_flat', t)
        self.register_buffer('center_matrix', center_matrix)

        activation = torch.nn.Tanh

        # network for f^1
        self.fc_in_frame1 = torch.nn.Sequential(*[torch.nn.Linear(N_in, N_hidden), activation()])
        # Hidden layers.
        self.hidden_layers_frame1 = torch.nn.Sequential(*[torch.nn.Sequential(*[torch.nn.Linear(N_hidden, N_hidden), activation()])
                                                          for _ in range(N_layers - 1)])
        # Output layer.
        self.fc_out_frame1 = torch.nn.Linear(N_hidden, N_out)


        # network for f^2
        self.fc_in_frame2 = torch.nn.Sequential(*[torch.nn.Linear(N_in, N_hidden), activation()])
        # Hidden layers.
        self.hidden_layers_frame2 = torch.nn.Sequential(*[torch.nn.Sequential(*[torch.nn.Linear(N_hidden, N_hidden), activation()])
                                                          for _ in range(N_layers - 1)])
        # Output layer.
        self.fc_out_frame2 = torch.nn.Linear(N_hidden, N_out)


        # Shift 1
        # Input layer.
        self.shift_in1 = torch.nn.Sequential(*[torch.nn.Linear(N_in_c, N_hidden), activation()])
        # Hidden layers.
        self.hidden_layers1 = torch.nn.Sequential(*[torch.nn.Sequential(*[torch.nn.Linear(N_hidden, N_hidden), activation()])
                                                          for _ in range(N_layers - 1)])
        # Output layer.
        self.shift_out1 = torch.nn.Linear(N_hidden, N_out_c)


        # Shift 2
        # Input layer.
        self.shift_in2 = torch.nn.Sequential(*[torch.nn.Linear(N_in_c, N_hidden), activation()])
        # Hidden layers.
        self.hidden_layers2 = torch.nn.Sequential(*[torch.nn.Sequential(*[torch.nn.Linear(N_hidden, N_hidden), activation()])
                                                          for _ in range(N_layers - 1)])
        # Output layer.
        self.shift_out2 = torch.nn.Linear(N_hidden, N_out_c)


    def forward(self):
        ################################ Shifts ####################################
        c1 = self.shift_in1(self.t_flat)
        for layer in self.hidden_layers1:
            c1 = layer(c1)
        c1 = self.shift_out1(c1)

        c2 = self.shift_in2(self.t_flat)
        for layer in self.hidden_layers2:
            c2 = layer(c2)
        c2 = self.shift_out2(c2)

        ################################ Frames ####################################
        x_shifted1 = self.x_flat - c1 * self.t_flat - self.center_matrix
        f1 = self.fc_in_frame1(torch.cat((x_shifted1, self.t_flat), dim=1))
        for layer in self.hidden_layers_frame1:
            f1 = layer(f1)
        f1 = self.fc_out_frame1(f1)

        x_shifted2 = self.x_flat - c2 * self.t_flat - self.center_matrix
        f2 = self.fc_in_frame2(torch.cat((x_shifted2, self.t_flat), dim=1))
        for layer in self.hidden_layers_frame2:
            f2 = layer(f2)
        f2 = self.fc_out_frame2(f2)



        f1_without_shift = self.fc_in_frame1(torch.cat((self.x_flat - self.center_matrix, self.t_flat), dim=1))
        for layer in self.hidden_layers_frame1:
            f1_without_shift = layer(f1_without_shift)
        f1_without_shift = self.fc_out_frame1(f1_without_shift)

        f2_without_shift = self.fc_in_frame2(torch.cat((self.x_flat - self.center_matrix, self.t_flat), dim=1))
        for layer in self.hidden_layers_frame2:
            f2_without_shift = layer(f2_without_shift)
        f2_without_shift = self.fc_out_frame2(f2_without_shift)

        return f1, f2, c1, c2, f1_without_shift, f2_without_shift

"""## Define inputs"""

x_flat = (x_device).repeat_interleave(Nt).to(device=device, dtype=dtype).view(-1, 1)
t_flat = (t_device).repeat(Nx).to(device=device, dtype=dtype).view(-1, 1)
Q = torch.tensor(Q, dtype=dtype, device=device)

lr = 0.0005
num_epochs = 25000
lambda_star = 0.005


"""## Create the TV matrix"""
DCSR = central_FD2Matrix(N=Nt)
coo = DCSR.tocoo()
indices = np.vstack((coo.row, coo.col))
i = torch.LongTensor(indices)  # indices need to be of type LongTensor
v = torch.FloatTensor(coo.data)  # values as FloatTensor
shape = coo.shape
D = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)


"""## Call the model"""

for lambda_TV in lambda_TV_list:
    print(f"\n============================================================")
    print(f"Running lambda_TV = {lambda_TV}")
    print(f"============================================================")

    model = ShapeShiftNet(2, 1, 1, 1, 32, 4, x_flat, t_flat, center_matrix)

    if pretrained_load:
        state_dict_original = torch.load("./data/StraightCubic_wave_TV/StraightCubic_wave_TV.pth")
        state_dict_new = model.state_dict()

        for name, param in state_dict_original.items():
            if name in state_dict_new:
                state_dict_new[name].copy_(param)
        model.load_state_dict(state_dict_new, strict=False)
        jit_model = torch.jit.script(model)
        jit_model.to(device)
        delta = 1e-1
    else:
        jit_model = torch.jit.script(model)
        jit_model.to(device)
        delta = 1e-5

    optimizer = torch.optim.Adam(jit_model.parameters(), lr=lr)

    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()

        # Function call for the model
        f1_full, f2_full, shift1_pred, shift2_pred, f1_full_nos, f2_full_nos = jit_model()
        T1Q1 = f1_full.view(Nx, Nt)
        T2Q2 = f2_full.view(Nx, Nt)
        Q1 = f1_full_nos.view(Nx, Nt)
        Q2 = f2_full_nos.view(Nx, Nt)
        shift1_mat = shift1_pred.view(Nx, Nt)
        shift2_mat = shift2_pred.view(Nx, Nt)

        frobenius_loss = torch.linalg.norm(Q - T1Q1 - T2Q2, 'fro')/ torch.linalg.norm(Q, 'fro')
        nuclear_loss = lambda_star * (NuclearNormAutograd.apply(Q1) + NuclearNormAutograd.apply(Q2))
        TV_loss = lambda_TV * (torch.linalg.norm(shift1_mat @ D, ord=1)
                               + torch.linalg.norm(shift2_mat @ D, ord=1))
        total_loss = frobenius_loss + nuclear_loss + TV_loss

        total_loss.backward()

        optimizer.step()

        if frobenius_loss < delta:
            print("Early stopping is triggered")
            break
        with torch.no_grad():
            if epoch % 10 == 0:
                print("\n**************************************************************")
                print(f'lambda_TV={lambda_TV} | Epoch {epoch}/{num_epochs}, F: {frobenius_loss.item():.4f}, '
                      f'N: {nuclear_loss.item():.4f}, '
                      f'TV: {TV_loss.item():.4f}, '
                      f'T: {total_loss.item():.4f}')

    # Bring everything back to CPU
    Q_np = Q.cpu().detach().numpy()
    Q_tilde_np = (T1Q1 + T2Q2).cpu().detach().numpy()
    T1Q1_np = T1Q1.cpu().detach().numpy()
    T2Q2_np = T2Q2.cpu().detach().numpy()
    Q1_np = Q1.cpu().detach().numpy()
    Q2_np = Q2.cpu().detach().numpy()
    shift1_np = shift1.cpu().detach().numpy()
    shift2_np = shift2.cpu().detach().numpy()
    shift1_numpy = shift1_pred.cpu().detach().numpy()
    shift2_numpy = shift2_pred.cpu().detach().numpy()
    shift1_numpy_mat = shift1_numpy.reshape(Nx, Nt)
    shift2_numpy_mat = shift2_numpy.reshape(Nx, Nt)
    shift1_pred_np = shift1_numpy * t_flat.cpu().detach().numpy()
    shift2_pred_np = shift2_numpy * t_flat.cpu().detach().numpy()
    shift1_pred_mat = shift1_pred_np.reshape(Nx, Nt)
    shift2_pred_mat = shift2_pred_np.reshape(Nx, Nt)
    shift1_val = shift1_pred_mat.max(axis=0)
    shift2_val = shift2_pred_mat.max(axis=0)

    rec_err = np.linalg.norm(Q_np - T1Q1_np - T2Q2_np) / np.linalg.norm(Q_np)
    print(f"RecErr: {rec_err}")

    """## Saving the results"""

    # NEW: unique suffix per lambda_TV so that all runs are preserved
    lam_str = f"{lambda_TV:g}".replace('.', 'p')

    torch.save(model.state_dict(), impath + f'StraightCubic_wave_TV_lambdaTV_{lam_str}.pth')
    np.save(impath + f'Q_lambdaTV_{lam_str}.npy', Q_np)
    np.save(impath + f'Q_tilde_lambdaTV_{lam_str}.npy', Q_tilde_np)
    np.save(impath + f'T1Q1_lambdaTV_{lam_str}.npy', T1Q1_np)
    np.save(impath + f'T2Q2_lambdaTV_{lam_str}.npy', T2Q2_np)
    np.save(impath + f'Q1_lambdaTV_{lam_str}.npy', Q1_np)
    np.save(impath + f'Q2_lambdaTV_{lam_str}.npy', Q2_np)
    np.save(impath + f'shifts1_lambdaTV_{lam_str}.npy', shift1_val)
    np.save(impath + f'shifts2_lambdaTV_{lam_str}.npy', shift2_val)
    np.save(impath + f'shifts1_true_lambdaTV_{lam_str}.npy', shift1_np)
    np.save(impath + f'shifts2_true_lambdaTV_{lam_str}.npy', shift2_np)
