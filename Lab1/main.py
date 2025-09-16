import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Läs in data
data = np.loadtxt("maintenance.txt")

X = data[:, :16]   # 16 features
y = data[:, 16:]   # 2 decay coefficients

# 2. Dela upp data i train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)

# 3. Normalisera features
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val   = x_scaler.transform(X_val)
X_test  = x_scaler.transform(X_test)

y_train_original = y_train.copy()
y_val_original   = y_val.copy()
y_test_original  = y_test.copy()

y_train = y_scaler.fit_transform(y_train)
y_val   = y_scaler.transform(y_val)
y_test  = y_scaler.transform(y_test)

# 4. Konvertera till PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train, dtype=torch.float64, device=device)
y_train = torch.tensor(y_train, dtype=torch.float64, device=device)
X_val   = torch.tensor(X_val, dtype=torch.float64, device=device)
y_val   = torch.tensor(y_val, dtype=torch.float64, device=device)
X_test  = torch.tensor(X_test, dtype=torch.float64, device=device)
y_test  = torch.tensor(y_test, dtype=torch.float64, device=device)

# 5. Skapa DataLoader för mini-batches
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# 6. Definiera nätverket
class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc3 = nn.Linear(256, 128, dtype=torch.float64)
        self.fc4 = nn.Linear(128, 64, dtype=torch.float64)
        self.fc5 = nn.Linear(64, 32, dtype=torch.float64)
        self.fc6 = nn.Linear(32, 2, dtype=torch.float64)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)  # regression → ingen aktivering
        return x

model = Net(n_features=X_train.shape[1]).to(device)

# 7. Loss och optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500)

# 8. Träning
epochs = 10000
best_val_loss = float("inf")
patience = 2000
trigger_times = 0

for epoch in range(epochs):
    # Training
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

    #Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.2e} - val_loss: {val_loss:.2e}- lr: {current_lr:.2e}")

     # Scheduler
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

# Ladda bästa modellen
model.load_state_dict(best_model_state)

# 9. Test
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_test_original
    
mse = ((y_true - y_pred) ** 2).mean(axis=0)

print("\n--- Slutresultat ---")
print(f"Test MSE compressor: {mse[0]:.2e}")
print(f"Test MSE turbine:    {mse[1]:.2e}")

