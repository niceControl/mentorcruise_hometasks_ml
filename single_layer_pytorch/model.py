import torch
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

log = logging.getLogger(__name__)

# Choose device for pytorch. Kept as
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def load_data(filename):
    dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir, "data", filename)
    if not os.path.exists(data_path):
        data_path = os.path.join(dir, "single_layer_pytorch", "data", filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(filename)
    return torch.load(data_path)


def init_params(in_features: int, hidden: int, out_features: int):
    # Kaiming initialization to avoid vanishing or exploding gradients
    W1 = (torch.randn(in_features, hidden) * ((2.0 / in_features) ** 0.5)).requires_grad_(True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(hidden, out_features) * ((2.0 / hidden) ** 0.5)).requires_grad_(True)
    b2 = torch.zeros(out_features, requires_grad=True)

    # UNUSED due to train speed degradation on current example
    # W1, b1, W2, b2 = (p.to(device) for p in (W1, b1, W2, b2))
    return W1, b1, W2, b2



def forward(X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
            W2: torch.Tensor, b2: torch.Tensor, training: bool = True, p_drop: float = 0.5):
    # hidden preactivation
    z1 = torch.matmul(X, W1) + b1
    h1 = torch.relu(z1)

    # apply dropout only during training
    h1 = dropout(h1, p=p_drop, training=training)

    # output
    y_hat = torch.matmul(h1, W2) + b2
    return y_hat

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor):

    loss = ((y_pred - y_true) ** 2).mean()
    loss.requires_grad_(True)
    return loss

def cross_entropy_loss(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)             # convert logits to probabilities
    N = logits.shape[0]
    correct_class_probs = probs[torch.arange(N), y_true]
    loss = -torch.log(correct_class_probs)
    return loss.mean()


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
def adam_step(params, adam_state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
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


def dropout(X: torch.Tensor, p: float = 0.5, training: bool = True):
    if not training or p == 0.0:
        return X
    mask = (torch.rand_like(X) > p).float()
    # scale activations so expected value remains the same
    return (X * mask) / (1.0 - p)


@torch.no_grad()
def evaluate(X, y, W1, b1, W2, b2, batch_size=512, task="regression"):
    N = X.shape[0]
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for i in range(0, N, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        out = forward(X_batch, W1, b1, W2, b2, training=False)

        if task == "regression":
            batch_loss = ((out - y_batch) ** 2).mean()
            total_loss += batch_loss.item() * X_batch.shape[0]
        else:
            y_idx = torch.clamp(y_batch.view(-1).to(torch.long) - 1, 0, 9)
            batch_loss = cross_entropy_loss(out, y_idx)
            total_loss += batch_loss.item() * X_batch.shape[0]

            preds = out.argmax(dim=1)
            total_correct += (preds == y_idx).sum().item()

        total_seen += X_batch.shape[0]

    avg_loss = total_loss / max(1, total_seen)
    if task == "regression":
        return avg_loss
    else:
        return avg_loss, total_correct / total_seen


def train(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor,
          y_val: torch.Tensor, epochs=4000, lr: float = 0.055, batch_size=128,
          task: str = "regression"):
    """
    task: "regression" (default) or "classification"
    Classification expects y to contain integer marks 1..10.
    """

    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")

    # choose output size
    out_features = 1 if task == "regression" else 10

    W1, b1, W2, b2 = init_params(X_train.shape[1], 5, out_features)
    params = [W1, b1, W2, b2]
    N = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        train_loss_sum = 0.0

        for i in range(0, N, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            prediction = forward(X_batch, W1, b1, W2, b2, training=True, p_drop=0.3)

            if task == "regression":
                loss = mse_loss(prediction, y_batch)
            else:
                y_idx = torch.clamp(y_batch.view(-1).to(torch.long) - 1, 0, 9)
                loss = cross_entropy_loss(prediction, y_idx)

            zero_grads(params)
            loss.backward()
            adam_step(params, adam_state, lr)

            train_loss_sum += loss.item() * X_batch.shape[0]

        # periodic eval
        if (epoch == 1) or (epoch == epochs) or (epoch % 25 == 0):
            if task == "regression":
                train_mse = train_loss_sum / N
                val_mse = evaluate(X_val, y_val, W1, b1, W2, b2, batch_size=512, task="regression")
                log.info(f"[{task}] Epoch: {epoch}, train_MSE: {train_mse:.6f}, val_mse: {val_mse:.6f}")
            else:
                train_ce = train_loss_sum / N
                val_ce, val_acc = evaluate(X_val, y_val, W1, b1, W2, b2, batch_size=512, task="classification")
                log.info(f"[{task}] Epoch: {epoch}, train_CE: {train_ce:.6f}, val_CE: {val_ce:.6f}, val_acc: {val_acc:.4f}")

    return W1, b1, W2, b2



if __name__ == "__main__":

    # load input data
    data = load_data('wine-white.pt')
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
    log.info('Starting model training - REGRESSION')

    W1_reg, b1_reg, W2_reg, b2_reg = train(X_train_norm, y_train, X_val_norm, y_val, task='regression')
    model_params_regression = {
        "layer1.weight": W1_reg,
        "layer1.bias": b1_reg,
        "layer2.weight": W2_reg,
        "layer2.bias": b2_reg,
    }


    log.info('Starting model training - Classification')

    W1_cls, b1_cls, W2_cls, b2_cls = train(X_train_norm, y_train, X_val_norm, y_val, task='classification')
    model_params_cls = {
        "layer1.weight": W1_cls,
        "layer1.bias": b1_cls,
        "layer2.weight": W2_cls,
        "layer2.bias": b2_cls,
    }
    for name, param in model_params_regression.items():
        log.info(f"{name}:\n{param.data}\n")

    for name, param in model_params_cls.items():
        log.info(f"{name}:\n{param.data}\n")
