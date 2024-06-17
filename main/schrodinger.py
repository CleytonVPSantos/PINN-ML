import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


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
    

def loss_function(model, x_init, t_init, u_init, x_boundary, t_boundary, u_boundary, x_collocation, t_collocation):
    # Condições iniciais
    u_init_pred = model(torch.cat((x_init, t_init), dim=1))
    mse_initial = nn.MSELoss()(u_init_pred, u_init)

    # Condições de fronteira
    u_boundary_pred = model(torch.cat((x_boundary, t_boundary), dim=1))
    mse_boundary = nn.MSELoss()(u_boundary_pred, u_boundary)

    # Rastreia as operações sobre os tensores, permitindo autodiff
    x_collocation.requires_grad = True
    t_collocation.requires_grad = True

    u = model(torch.cat((x_collocation, t_collocation), dim=1))
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

    f = torch.cat((f_real, f_imag))
    mse_pde = nn.MSELoss()(f, torch.zeros_like(f))

    # Loss total
    loss = mse_initial + mse_pde + mse_boundary
    return loss


def train_model(device):
    # Parâmetros
    input_size = 2
    hidden_size = 100
    output_size = 2
    model = PINN(input_size, hidden_size, output_size).to(device)

    # Quantidade de exemplos de treinamento
    n_boundary = 30
    n_initial = 30
    n_collocation = 100
    
    # Dados para treinamento
    x_init = torch.linspace(-5, 5, n_initial).reshape(-1, 1).to(device)
    t_init = torch.zeros_like(x_init).to(device)
    u_init = 2/torch.cosh(x_init).to(device)

    x_boundary = torch.cat([-5.0*torch.ones(n_boundary), 5.0*torch.ones(n_boundary)]).reshape(-1, 1).to(device)
    t_boundary = torch.linspace(0, np.pi/2, n_boundary).reshape(-1, 1).repeat(2, 1).to(device)
    u_boundary = torch.zeros_like(t_boundary).to(device)

    x_collocation, t_collocation = torch.meshgrid(
        torch.linspace(-5, 5, n_collocation),
        torch.linspace(0, np.pi/2, n_collocation),
        indexing='ij'
    )
    x_collocation = x_collocation.reshape(-1, 1).to(device)
    t_collocation = t_collocation.reshape(-1, 1).to(device)

    # Otimizador
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50000, max_eval=50000, history_size=50)

    # Função de fechamento necessária para L-BFGS
    def closure():
        optimizer.zero_grad()  # Zera os gradientes dos parâmetros
        loss = loss_function(model, x_init, t_init, u_init, x_boundary, t_boundary, u_boundary, x_collocation, t_collocation)
        loss.backward()  # Backward pass para calcular gradientes
        return loss

    # Treinamento
    model.train()

    for epoch in range(50):
        loss = optimizer.step(closure)
        print(f"Epoch {epoch}, Loss: {loss.item()}")


def visualize_result(device, model, loss):
    # Visualização e plot
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 100)
        t_plot = torch.linspace(0, 1, 100)
        x_plot, t_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
        u_plot = model(torch.cat((x_plot.reshape(-1, 1), t_plot.reshape(-1, 1)), dim=1))
        u_plot = u_plot.reshape(100, 100)

    # Plotando a solução
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_plot.numpy(), t_plot.numpy(), u_plot.numpy(), cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x, t)')
    plt.title('Solution of the Heat Equation using PINN')
    plt.show()

    print(f"Final Loss: {loss.item()}")


def main():
    # Verifica se a GPU está disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, loss = train_model(device)
    visualize_result(device, model, loss)
