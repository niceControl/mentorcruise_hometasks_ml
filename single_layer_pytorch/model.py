import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

log = logging.getLogger(__name__)

# Choose device for pytorch. Kept as
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# load input data
data = torch.load("data/wine-white.pt")
log.info(f"Initial slice of data: {data}")
log.info(f"data size is {data.shape}")
index = torch.randperm(data.shape[0])
data = data[index]
X, y = data[:, :-1], data[:, -1:]
X_train, X_val = X[:3500], X[3500:]
y_train, y_val = y[:3500], y[3500:]
log.info(f"Train dataset: {X_train.shape}, {y_train.shape}")
log.info(f"Validation dataset: {X_val.shape}, {y_val.shape}")

# Normalize input data
mean = X_train.mean(0)
std = X_train.std(0)
X_train_norm = (X_train - mean) / std
X_val_norm = (X_val - mean) / std
log.info(f"Normalized data: X_train: {X_train_norm} \n X_val: {X_val_norm}")

# move all data to device - UNUSED due to train speed degradation on current example
# X_train_norm = X_train_norm.to(device)
# X_val_norm = X_val_norm.to(device)
# y_train = y_train.to(device)
# y_val = y_val.to(device)

# create empty ADAM state
adam_state = {}


def init_params(in_features: int, hidden: int, out_features: int):
    # Kaiming initialization to avoid vanishing or exploding gradients
    W1 = (torch.randn(in_features, hidden) * ((2.0 / in_features) ** 0.5)).requires_grad_(True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(hidden, out_features) * ((2.0 / hidden) ** 0.5)).requires_grad_(True)
    b2 = torch.zeros(out_features, requires_grad=True)

    # UNUSED due to train speed degradation on current example
    # W1, b1, W2, b2 = (p.to(device) for p in (W1, b1, W2, b2))
    return W1, b1, W2, b2


def forward(X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor):

    # Calculate z1 - preactivation of hidden layer
    z1 = torch.matmul(X, W1) + b1

    # Calculate h1 - post-activation of hidden layer
    h1 = torch.relu(z1)

    # calculate prediction y hat
    y_hat = torch.matmul(h1, W2) + b2

    return y_hat


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor):

    loss = ((y_pred - y_true) ** 2).mean()
    loss.requires_grad_(True)
    return loss


def zero_grads(params):
    # reset grads for each parameters to avoid grad accumulation
    for p in params:
        p.grad = None


# Base gradient descent implementation without optimizer
@torch.no_grad()
def sgd_step(params, lr: float):
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad


@torch.no_grad()
def adam_step(params, adam_state, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
    for p in params:
        # skip params without gradient
        if p.grad is None:
            continue

        state = adam_state.setdefault(
            id(p),
            {'m': torch.zeros_like(p),  # EMA of gradients
             'v': torch.zeros_like(p),  # EMA of squared gradients
             't': 0}
        )
        state['t'] = state['t'] + 1  # step counter for bias correction

        # Update "m" and "v"
        state['m'] = beta1 * state['m'] + (1.0 - beta1) * p.grad
        state['v'] = beta2 * state['v'] + (1.0 - beta2) * (p.grad * p.grad)

        # bias correction
        b1c = 1.0 - beta1 ** state['t']
        b2c = 1.0 - beta2 ** state['t']
        m_hat = state['m'] / b1c
        v_hat = state['v'] / b2c

        denominator = v_hat.sqrt() + eps
        p.data = p.data - lr * (m_hat / denominator)


@torch.no_grad()
def evaluate(X: torch.Tensor, y: torch.Tensor, W1, b1, W2, b2, batch_size: int = 512):
    # unction to measure model perf (calculate cost) on Validation dataset without changing weight
    N = X.shape[0]  # number of examples
    total_loss = 0.0

    for i in range(0, N, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        z1 = torch.matmul(X_batch, W1) + b1
        h1 = torch.relu(z1)

        y_hat_batch = torch.matmul(h1, W2) + b2

        batch_loss = ((y_hat_batch - y_batch) ** 2).mean()

        total_loss += (batch_loss.item() * X_batch.shape[0]) # to avoid different batch size contribution diff

    total_loss = total_loss / N
    return total_loss


def train(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor,
          y_val: torch.Tensor, epochs: int = 400, lr: float = 0.05, batch_size: int = 128):

    W1, b1, W2, b2 = init_params(X_train.shape[1], 5, 1)
    params = [W1, b1, W2, b2]
    N = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        train_loss_sum = 0.0
        for i in range(0, N, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            prediction = forward(X_batch, W1, b1, W2, b2)

            loss = mse_loss(prediction, y_batch)

            zero_grads(params) # reset old grads
            loss.backward()
            adam_step(params, adam_state, lr)
            train_loss_sum = loss.item() * X_batch.shape[0]

        if (epoch == 1) or (epoch == epochs) or (epoch % 25 == 0):
            train_mse = train_loss_sum / N 
            val_mse = evaluate(X_val, y_val, W1, b1, W2, b2, batch_size=512)
            log.info(f"Epoch: {epoch}, train_MSE: {train_mse}, val_mse: {val_mse}")

    return W1, b1 , W2, b2


if __name__ == "__main__":
    log.info('Starting model training')
    model_params = train(X_train_norm, y_train, X_val_norm, y_val)
    log.info(f"Model params: {model_params}")