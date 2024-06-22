import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat


# Verifica se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# arquitetura da rede neural: 5 camadas hidden layers com 100 neuronios cada
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.tanh(self.layer4(x))
        x = self.tanh(self.layer5(x))
        x = self.output_layer(x)
        return x
    

def loss_function(model, x_init, t_init, u_init, t_boundary, x_collocation, t_collocation):
    # Condições iniciais
    u_init_pred = model(torch.cat((x_init, t_init), dim=1))
    mse_initial = nn.MSELoss()(u_init_pred, u_init)

    # Rastreia as operações sobre os tensores, permitindo autodiff
    x_collocation.requires_grad = True
    t_collocation.requires_grad = True

    u = model(torch.cat((x_collocation, t_collocation), dim=1)).view(-1, 2)
    u_real = u[:, 0:1]
    u_imag = u[:, 1:2]

    u_real_t = torch.autograd.grad(u_real, t_collocation, grad_outputs=torch.ones_like(u_real), create_graph=True)[0]
    u_imag_t = torch.autograd.grad(u_imag, t_collocation, grad_outputs=torch.ones_like(u_imag), create_graph=True)[0]
    u_real_x = torch.autograd.grad(u_real, x_collocation, grad_outputs=torch.ones_like(u_real), create_graph=True)[0]
    u_imag_x = torch.autograd.grad(u_imag, x_collocation, grad_outputs=torch.ones_like(u_imag), create_graph=True)[0]
    u_real_xx = torch.autograd.grad(u_real_x, x_collocation, grad_outputs=torch.ones_like(u_real_x), create_graph=True)[0]
    u_imag_xx = torch.autograd.grad(u_imag_x, x_collocation, grad_outputs=torch.ones_like(u_imag_x), create_graph=True)[0]

    x_collocation.requires_grad = False
    t_collocation.requires_grad = False

    u_abs_square = u_real**2 + u_imag**2
    f_real = -u_imag_t + 0.5 * u_real_xx + u_abs_square * u_real
    f_imag = u_real_t + 0.5 * u_imag_xx + u_abs_square * u_imag

    f = torch.cat((f_real, f_imag), dim=0)
    mse_pde = nn.MSELoss()(f, torch.zeros_like(f))

    # Condições de fronteira
    x_boundary1 = 5.0 * torch.ones_like(t_boundary)
    x_boundary2 = -5.0 * torch.ones_like(t_boundary)

    x_boundary1.requires_grad = True
    x_boundary2.requires_grad = True

    u_boundary_pred1 = model(torch.cat((x_boundary1, t_boundary), dim=1))
    u_boundary_pred2 = model(torch.cat((x_boundary2, t_boundary), dim=1))
    mse_boundary1 = nn.MSELoss()(u_boundary_pred1, u_boundary_pred2)

    # Derivada nas fronteiras
    u_real_x_boundary1 = torch.autograd.grad(u_boundary_pred1[:, 0], x_boundary1, grad_outputs=torch.ones_like(u_boundary_pred1[:, 0]), create_graph=True)[0]
    u_imag_x_boundary1 = torch.autograd.grad(u_boundary_pred1[:, 1], x_boundary1, grad_outputs=torch.ones_like(u_boundary_pred1[:, 1]), create_graph=True)[0]
    u_real_x_boundary2 = torch.autograd.grad(u_boundary_pred2[:, 0], x_boundary2, grad_outputs=torch.ones_like(u_boundary_pred2[:, 0]), create_graph=True)[0]
    u_imag_x_boundary2 = torch.autograd.grad(u_boundary_pred2[:, 1], x_boundary2, grad_outputs=torch.ones_like(u_boundary_pred2[:, 1]), create_graph=True)[0]

    mse_boundary2 = nn.MSELoss()(u_real_x_boundary1, u_real_x_boundary2) + nn.MSELoss()(u_imag_x_boundary1, u_imag_x_boundary2)

    x_boundary1.requires_grad = False
    x_boundary2.requires_grad = False

    # Loss total
    loss = mse_initial + mse_pde + mse_boundary1 + mse_boundary2
    return loss


# Parâmetros
input_size = 2
hidden_size = 100
output_size = 2
model = PINN(input_size, hidden_size, output_size).to(device)

# Quantidade de exemplos de treinamento
n_boundary = 100
n_initial = 200
n_collocation = 20000

# Dados para treinamento
sampler = scipy.stats.qmc.LatinHypercube(d=1) # metodo Latin Hypercube

x_init = torch.Tensor(2 * sampler.random(n=n_initial) - 1).reshape(-1,1).to(device)
t_init = torch.zeros_like(x_init).to(device)
u_init = torch.cat((2/torch.cosh(x_init), torch.zeros_like(x_init)), dim=1).to(device)

t_boundary = torch.Tensor(np.pi/2 * sampler.random(n=n_boundary)).to(device)

sampler = scipy.stats.qmc.LatinHypercube(d=2)
collocation = torch.Tensor(sampler.random(n_collocation))
x_collocation = collocation[:,0].reshape(-1,1).to(device)
x_collocation = 10 * x_collocation - 5
t_collocation = collocation[:,1].reshape(-1,1).to(device)


# Otimizador
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=50000, max_eval=50000, history_size=50)

# Função de fechamento necessária para L-BFGS
def closure():
    optimizer.zero_grad()  # Zera os gradientes dos parâmetros
    loss = loss_function(model, x_init, t_init, u_init, t_boundary, x_collocation, t_collocation)
    loss.backward()  # Backward pass para calcular gradientes
    return loss

# Treinamento
model.train()

for epoch in range(50):
    loss = optimizer.step(closure)
    print(f"Epoca {epoch}, Loss: {loss.item()}")


# Visualização e plot
model.eval()
with torch.no_grad():
    x_plot = torch.linspace(-5, 5, 100).to(device)
    t_plot = torch.linspace(0, np.pi/2-0.2, 100).to(device)
    x_plot, t_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
    x_plot_flat = x_plot.reshape(-1, 1)
    t_plot_flat = t_plot.reshape(-1, 1)
    u_plot = model(torch.cat((x_plot_flat, t_plot_flat), dim=1))
    u_real_plot = u_plot[:, 0].reshape(100, 100)
    u_imag_plot = u_plot[:, 1].reshape(100, 100)
    u_modulus_plot = torch.sqrt(u_real_plot**2 + u_imag_plot**2)

# Plotando a solução
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot.cpu().numpy(), t_plot.cpu().numpy(), u_modulus_plot.cpu().numpy(), cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('|u(x, t)|')
plt.title('Solution of the Schrödinger Equation using PINN (Modulus)')
plt.show()


# Certifique-se de que o modelo está em modo de avaliação
model.eval()

# Cria subplots com 3 gráficos
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Define o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tempos para plotagem
times = [0.59, 0.79, 0.98]

for i, time in enumerate(times):
    with torch.no_grad():
        x_plot = torch.linspace(-5, 5, 100).reshape(-1, 1).to(device)
        t_plot = torch.tensor([time]).expand_as(x_plot).to(device)  # Solução no tempo especificado
        u_plot = model(torch.cat((x_plot, t_plot), dim=1))

        u_real_plot = u_plot[:, 0]
        u_imag_plot = u_plot[:, 1]
        u_modulus_plot = torch.sqrt(u_real_plot**2 + u_imag_plot**2)

    axes[i].plot(x_plot.cpu().numpy(), u_modulus_plot.cpu().numpy(), label='Predicted Solution (PINN)', color='blue')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel(f'|u(x, t={time})|')
    axes[i].set_title(f'Solution of the Schrödinger Equation using PINN (t = {time})')
    axes[i].legend()
    axes[i].set_ylim(-0, 5.0)  # Ajusta os eixos para terem a mesma escala

# Mostra os gráficos
plt.show()

print(f"Final Loss: {loss.item()}")