import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat


# Verifica se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Arquitetura da rede neural: 5 camadas hidden layers com 100 neuronios cada
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

        # Parâmetros alpha e beta como parâmetros treináveis
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.tanh(self.layer4(x))
        x = self.tanh(self.layer5(x))
        x = self.output_layer(x)
        return x


def loss_function(model, x_train, t_train, u_train):
    # Rastreia as operações sobre os tensores, permitindo autodiff
    x_train.requires_grad = True
    t_train.requires_grad = True

    u = model(torch.cat((x_train, t_train), dim=1))
    u_t = torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Usando o alpha e beta da classe PINN
    alpha = model.alpha
    beta = model.beta
    f_pred = u_t + alpha * u*u_x -  beta * u_xx
    mse_pde = nn.MSELoss()(f_pred, torch.zeros_like(f_pred))
    mse_eq = nn.MSELoss()(u, u_train)

    x_train.requires_grad = False
    t_train.requires_grad = False
    # Loss total
    loss = mse_pde + mse_eq
    return loss


# Carregar os dados do arquivo KdV.mat
data = loadmat('burgers_shock.mat')
u_data = torch.Tensor(data['usol'])
t_data = torch.Tensor(data['t']).squeeze()  # Remove a dimensão extra se existir
x_data = torch.Tensor(data['x']).squeeze()   # Remove a dimensão extra se existir


# Número de exemplos de treinamento
n_train = 10000

# Geração de índices de amostra
total_size = 100 * 256  # Máximo índice permitido
train_choices = np.random.choice(np.arange(total_size), size=n_train, replace=False)
ind = np.zeros(total_size, dtype=bool)
ind[train_choices] = True
rest = ~ind
test_choices = np.arange(total_size)[rest]

# Calcular t_choices e x_choices
train_t_choices = train_choices % 100
train_x_choices = (train_choices // 100) 

# Calcular t_choices e x_choices
test_t_choices = test_choices % 100
test_x_choices = (test_choices // 100)  

# Obter valores de u, x, e t para o treinamento
u_train = u_data[train_x_choices, train_t_choices].reshape(-1, 1).to(device)
x_train = x_data[train_x_choices].reshape(-1, 1).to(device)
t_train = t_data[train_t_choices].reshape(-1, 1).to(device)

# Obter valores de u, x, e t para o treinamento
u_test = u_data[test_x_choices, test_t_choices].reshape(-1, 1).to(device)
x_test = x_data[test_x_choices].reshape(-1, 1).to(device)
t_test = t_data[test_t_choices].reshape(-1, 1).to(device)


# Parâmetros
input_size = 2
hidden_size = 100
output_size = 1
model = PINN(input_size, hidden_size, output_size).to(device)

# Otimizador
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)

# Função de fechamento necessária para L-BFGS
def closure():
    optimizer.zero_grad()
    loss = loss_function(model, x_train, t_train, u_train)
    loss.backward()
    return loss

# Treinamento
model.train()

for epoch in range(500):
    loss = optimizer.step(closure)
    print(f"Época {epoch}, Loss: {loss.item()}, Alpha: {model.alpha.item()}, Beta: {model.beta.item()}")


model.eval()
u_pred = model(torch.cat((x_test, t_test), dim=1))
mse = nn.MSELoss()(u_pred, u_test)
print(f"Loss no conjunto de teste: {mse.item()}")
print(f"Loss no conjunto de treino: {loss.item()}")

# Visualização e plot
with torch.no_grad():
    x_plot = torch.linspace(-1, 1, 100).to(device)
    t_plot = torch.linspace(0, 1, 100).to(device)
    x_plot, t_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
    u_plot = model(torch.cat((x_plot.reshape(-1, 1), t_plot.reshape(-1, 1)), dim=1))
    u_plot = u_plot.reshape(100, 100)

# Plotando a solução
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot.cpu().numpy(), t_plot.cpu().numpy(), u_plot.cpu().numpy(), cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x, t)')
plt.title('Eq. Burgers com PINN descoberta')
plt.show()

# Certifique-se de que o modelo está em modo de avaliação
model.eval()

# Cria subplots com 3 gráficos
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Define o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tempos para plotagem
times = [0.25, 0.5, 0.75]

for i, time in enumerate(times):
    with torch.no_grad():
        x_plot = torch.linspace(-1, 1, 100).reshape(-1, 1).to(device)
        t_plot = torch.tensor([time]).expand_as(x_plot).to(device)  # Solução no tempo especificado
        u_plot = model(torch.cat((x_plot, t_plot), dim=1))

    axes[i].plot(x_plot.cpu().numpy(), u_plot.cpu().numpy(), label='Predicted Solution (PINN)', color='blue')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel(f'u(x, t={time})')
    axes[i].set_title(f'Discovery equação de Burgers (t = {time})')
    axes[i].legend()
    axes[i].set_ylim(-1.0, 1.0)  # Ajusta os eixos para terem a mesma escala

# Mostra os gráficos
plt.show()