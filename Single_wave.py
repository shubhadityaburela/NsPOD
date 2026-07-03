# -*- coding: utf-8 -*-

import torch.nn as nn
import numpy as np
import torch
import os

# =============================================================================
# Universal Device Router & Fail-Safe Hardware Setup
# =============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"--> Dynamic hardware router initialized. Active device: {device.type.upper()}")

dtype = torch.float32
pretrained_load = True


impath = "./data/Single_wave/"
immpath = "./plots/Single_wave/"
os.makedirs(impath, exist_ok=True)
os.makedirs(immpath, exist_ok=True)


"""# Single wave example

## Create data for example model
"""

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

Nx, Nt = 300, 300
t_start = -10.0
t_end = 10.0

x = np.arange(0, Nx)
t = np.linspace(t_start, t_end, Nt)
x_device = torch.tensor(x.copy(),  dtype=dtype, device=device)
t_device = torch.tensor(t.copy(),  dtype=dtype, device=device)
coefficients = torch.tensor([-10.0, 1.5], dtype=dtype, device=device)
sigma = torch.tensor(4.0, dtype=dtype, device=device)
center_matrix = torch.tensor(150.0, dtype=dtype, device=device)


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


def generate_data_single_wave(coefficient, x, t, center_of_matrix, sigma):
    shift = torch_polyval(coefficient, t_device)
    X, MU = torch.meshgrid(x, center_of_matrix + shift)
    Q = torch_gaussian(X, MU, sigma)
    return Q, shift


Q, shift = generate_data_single_wave(coefficients, x_device, t_device, center_matrix, sigma)


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


        # Shift
        # Input layer.
        self.shift_in = torch.nn.Sequential(*[torch.nn.Linear(N_in_c, N_hidden), activation()])
        # Hidden layers.
        self.hidden_layers = torch.nn.Sequential(*[torch.nn.Sequential(*[torch.nn.Linear(N_hidden, N_hidden), activation()])
                                                          for _ in range(N_layers - 1)])
        # Output layer.
        self.shift_out = torch.nn.Linear(N_hidden, N_out_c)


    @torch.jit.export
    def forward(self):
        ################################ Shifts ####################################
        c = self.shift_in(self.t_flat)
        for layer in self.hidden_layers:
            c = layer(c)
        c = self.shift_out(c)

        ################################ Frames ####################################
        x_shifted1 = self.x_flat - c * self.t_flat - self.center_matrix
        f1 = self.fc_in_frame1(torch.cat((x_shifted1, self.t_flat), dim=1))
        for layer in self.hidden_layers_frame1:
            f1 = layer(f1)
        f1 = self.fc_out_frame1(f1)

        f1_without_shift = self.fc_in_frame1(torch.cat((self.x_flat - self.center_matrix, self.t_flat), dim=1))
        for layer in self.hidden_layers_frame1:
            f1_without_shift = layer(f1_without_shift)
        f1_without_shift = self.fc_out_frame1(f1_without_shift)

        return f1, f1_without_shift, c

"""## Define inputs"""

x_flat = (x_device).repeat_interleave(Nt).to(device=device, dtype=dtype).view(-1, 1)
t_flat = (t_device).repeat(Nx).to(device=device, dtype=dtype).view(-1, 1)
Q = torch.tensor(Q, dtype=dtype, device=device)

lr = 0.0005
num_epochs = 75000
lambda_star = 0.005

"""## Call the model"""

model = ShapeShiftNet(2, 1, 1, 1, 32, 4, x_flat, t_flat, center_matrix)

if pretrained_load:
    state_dict_original = torch.load("./trained_weights/Single_wave/Single_wave.pth", map_location=device)
    state_dict_new = model.state_dict()

    for name, param in state_dict_original.items():
        if name in state_dict_new:
            state_dict_new[name].copy_(param)
    model.load_state_dict(state_dict_new, strict=False)
    jit_model = torch.jit.script(model)
    jit_model.to(device)

    # --- EVALUATION ONLY MODE ---
    print("--> Pretrained model loaded successfully. Evaluating weights safely...")
    jit_model.eval()
    with torch.no_grad():
        f1_full, f1_full_nos, c_raw = jit_model()

        # Adding .clone() breaks the dangerous TorchScript view dependencies instantly!
        T1Q1 = f1_full.view(Nx, Nt).clone()
        Q1 = f1_full_nos.view(Nx, Nt).clone()
        c = c_raw.clone()

        # Calculate evaluation loss for tracking
        frobenius_loss = torch.linalg.norm(Q - T1Q1, 'fro') / torch.linalg.norm(Q, 'fro')
        nuclear_loss = lambda_star * NuclearNormAutograd.apply(Q1)
        total_loss = frobenius_loss + nuclear_loss

        print(
            f"Pretrained Model Evaluation -> F: {frobenius_loss.item():.4f}, N: {nuclear_loss.item():.4f}, T: {total_loss.item():.4f}")
else:
    # --- TRAINING MODE ---
    jit_model = torch.jit.script(model)
    jit_model.to(device)
    delta = 1e-5

    optimizer = torch.optim.Adam(jit_model.parameters(), lr=lr)

    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()

        # Function call for the model
        f1_full, f1_full_nos, c = jit_model()
        T1Q1 = f1_full.view(Nx, Nt)
        Q1 = f1_full_nos.view(Nx, Nt)

        frobenius_loss = torch.linalg.norm(Q - T1Q1, 'fro') / torch.linalg.norm(Q, 'fro')
        nuclear_loss = lambda_star * NuclearNormAutograd.apply(Q1)
        total_loss = frobenius_loss + nuclear_loss

        total_loss.backward()
        optimizer.step()

        if frobenius_loss < delta:
            print("Early stopping is triggered")
            break

        with torch.no_grad():
            if epoch % 10 == 0:
                print("\n**************************************************************")
                print(f'Epoch {epoch}/{num_epochs}, F: {frobenius_loss.item():.4f}, '
                      f'N: {nuclear_loss.item():.4f}, '
                      f'T: {total_loss.item():.4f}')

# Bring everything back to CPU
Q = Q.cpu().detach().numpy()
Q_tilde = T1Q1.cpu().detach().numpy()
T1Q1 = T1Q1.cpu().detach().numpy()
Q1 = Q1.cpu().detach().numpy()
shift = shift.cpu().detach().numpy()
c_numpy = c.cpu().detach().numpy()
c_numpy_mat = c_numpy.reshape(Nx, Nt)
shift_pred = c_numpy * t_flat.cpu().detach().numpy()
shift_pred_mat = shift_pred.reshape(Nx, Nt)
shift_val = shift_pred_mat.max(axis=0)

rec_err = np.linalg.norm(Q - T1Q1) / np.linalg.norm(Q)
print(f"RecErr: {rec_err}")

"""## Saving the results"""
if not pretrained_load:
    torch.save(model.state_dict(), impath + 'Single_wave.pth')

np.save(impath + 'Q.npy', Q)
np.save(impath + 'Q_tilde.npy', Q_tilde)
np.save(impath + 'T1Q1.npy', T1Q1)
np.save(impath + 'Q1.npy', Q1)
np.save(impath + 'shift.npy', shift_val)
np.save(impath + 'shift_true.npy', shift)
