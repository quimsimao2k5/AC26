import torch
import torch.nn as nn

# 1. Dados de treino simples
# X é o input, y é o output real (seguindo a lógica y = 2x + 1)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0], [13.0]])

# 2. Definição do modelo
# nn.Linear é o equivalente exato à DenseLayer que estavas a construir.
# Tem 1 feature de entrada e 1 feature de saída.
model = nn.Linear(in_features=1, out_features=1)

# 3. Função de perda e Otimizador
# Mean Squared Error (Erro Quadrático Médio) e Stochastic Gradient Descent
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Ciclo de treino
epochs = 100
for epoch in range(epochs):
    
    # Forward propagation
    predictions = model(X)
    
    # Cálculo da loss
    loss = criterion(predictions, y)
    
    # Backward propagation
    optimizer.zero_grad() # Limpar os gradientes da iteração anterior
    loss.backward()       # Calcular os gradientes atuais (equivalente ao TODO 8)
    optimizer.step()      # Atualizar pesos e biases (equivalente ao opt.update)

    # Imprimir o progresso a cada 20 épocas
    if (epoch + 1) % 20 == 0:
        print(f'Época [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Fazer uma previsão com dados novos (x = 5.0)
novo_dado = torch.tensor([[11.0]])
previsao = model(novo_dado)

print(f'\nPrevisão do modelo para x=7.0: {previsao.item():.4f}')
print('Resultado matemático esperado: 11.0')