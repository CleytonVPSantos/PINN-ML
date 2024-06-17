import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# arquitetura da rede neural: 3 hidden layers, ativacao tanh
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.output_layer(x)
        return x
    
def loss_function(model, t, u_init, t_collocation):
    # condicoes iniciais
    u_pred = model(t)
    mse_initial = nn.MSELoss()(u_pred, u_init)

    # rastreia as operacoes sobre os tensores, permitindo autodiff
    t_collocation.requires_grad = True

    u = model(t_collocation)
    u_t = torch.autograd.grad(u, t_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    f_pred = u_t + 15.0*u
    mse_pde = nn.MSELoss()(f_pred, torch.zeros_like(f_pred))

    # loss total
    loss = mse_initial + mse_pde
    return loss

def train_model():
    # init
    input_size = 1
    hidden_size = 100
    output_size = 1
    model = PINN(input_size, hidden_size, output_size)

    # dados p/ treinamento (condições iniciais e pontos de colocation)
    t_init = torch.tensor([[0.0]], requires_grad=True)
    u_init = torch.tensor([[1.0]], requires_grad=True)
    t_collocation = torch.linspace(0, 2, 50).reshape(-1, 1)

    # usamos o otimizador L-BFGS (porque o cara do artigo quis)
    optimizer = optim.LBFGS(model.parameters(), lr=0.01)

    # função de fechamento necessária para L-BFGS
    def closure():
        optimizer.zero_grad()  # Zera os gradientes dos parâmetros
        loss = loss_function(model, t_init, u_init, t_collocation)
        loss.backward()  # Backward pass para calcular gradientes
        return loss

    # lets freakin train this shheet
    model.train()
    for epoch in range(50):
        loss = optimizer.step(closure)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, loss

def visualize_result(model, loss):
    # Visualização e plot
    model.eval()
    with torch.no_grad():
        t_plot = torch.linspace(0, 2, 100).view(-1, 1)  # 100 pontos igualmente espaçados em t
        u_plot = model(t_plot)  # Avalia o modelo nos pontos t_plot

    # Solução analítica
    t_analytical = t_plot.numpy()
    u_analytical = np.exp(-15.0*t_analytical)

    # Plotando a solução
    plt.plot(t_plot.numpy(), u_plot.numpy(), label='Predicted Solution (PINN)', color='blue')
    plt.plot(t_analytical, u_analytical, label='Analytical Solution', color='red', linestyle='dashed')
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title('Solution of the Differential Equation using PINN')
    plt.legend()
    plt.show()

    print(f"Final Loss: {loss.item()}")

def main():
    model, loss = train_model()
    visualize_result(model, loss)

main()