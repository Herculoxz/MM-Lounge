'''
add the following libs into your venv 
- jax ( ye bas numpy ka faster version hain. )
- scikit-learn 
- optax 
-pandas 

'''



import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('AAPL.csv')
print(df.head())

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
print(scaled_data[:5])

import jax
import jax.numpy as jnp
from jax import random
import optax

class LSTM:
    def __init__(self, input_dim, hidden_dim, key):
        self.hidden_dim = hidden_dim

        self.params = {
            'Wf': random.normal(key, (hidden_dim, hidden_dim + input_dim)),
            'bf': jnp.zeros((hidden_dim, 1)),

            'Wi': random.normal(key, (hidden_dim, hidden_dim + input_dim)),
            'bi': jnp.zeros((hidden_dim, 1)),

            'Wc': random.normal(key, (hidden_dim, hidden_dim + input_dim)),
            'bc': jnp.zeros((hidden_dim, 1)),

            'Wo': random.normal(key, (hidden_dim, hidden_dim + input_dim)),
            'bo': jnp.zeros((hidden_dim, 1))
        }

    def sigmoid(self, x):
        return jax.nn.sigmoid(x)

    def tanh(self, x):
        return jnp.tanh(x)

    def forward(self, x, h_prev, C_prev, params):
        x = x.reshape(-1,1)
        concat = jnp.concatenate((h_prev, x) , axis = 0)

        Ft = self.sigmoid(jnp.dot(params['Wf'], concat) + params['bf'])
        It = self.sigmoid(jnp.dot(params['Wi'], concat) + params['bi'])
        Ot = self.sigmoid(jnp.dot(params['Wo'], concat) + params['bo'])
        C_ht = self.tanh(jnp.dot(params['Wc'], concat) + params['bc'])

        Ct = Ft * C_prev + It * C_ht
        Ht = Ot * self.tanh(Ct)

        return Ht, Ct

def create_seq(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i:i + seq_len]
        target = data[i + seq_len, 3]  # Using 'Close' as the target
        sequences.append((seq, target))
    return sequences

seq_len = 50
sequences = create_seq(scaled_data, seq_len)

split = int(0.8 * len(sequences))
train_seq = sequences[:split]
test_seq = sequences[split:]

train_seq = [(jnp.array(x), jnp.array(y)) for x, y in train_seq]
test_seq = [(jnp.array(x), jnp.array(y)) for x, y in test_seq]

input_dim = 6  # Date is not considered
hidden_dim = 2
key = random.PRNGKey(0)

lstm = LSTM(input_dim, hidden_dim, key)

def mse_loss(params, model, batch):
    inputs, targets = batch
    preds = []
    h, c = jnp.zeros((model.hidden_dim, 1)), jnp.zeros((model.hidden_dim, 1))
    for t in range(inputs.shape[0]):
        h, c = model.forward(inputs[t], h, c, params)
    pred = h[0, 0]  # Output is the first element of the hidden state
    return jnp.mean((pred - targets) ** 2)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(lstm.params)

@jax.jit
def update(params, opt_state, batch):
    loss, grads = jax.value_and_grad(mse_loss)(params, lstm, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, opt_state, params

num_epochs = 10
params = lstm.params  # Initial parameters

for epoch in range(num_epochs):
    for batch in train_seq:
        loss, opt_state, params = update(params, opt_state, batch)
    print(f'Epoch {epoch + 1}, Loss: {loss}')

def predict(model, inputs, params):
    h, c = jnp.zeros((model.hidden_dim, 1)), jnp.zeros((model.hidden_dim, 1))
    for t in range(inputs.shape[0]):
        h, c = model.forward(inputs[t], h, c, params)
    return h[0, 0]  # Output is the first element of the hidden state

predictions = []
for seq, target in test_seq:
    pred = predict(lstm, seq, params)
    predictions.append(pred)

close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']])

predictions = close_scaler.inverse_transform(jnp.array(predictions).reshape(-1, 1))
actuals = close_scaler.inverse_transform(jnp.array([target for _, target in test_seq]).reshape(-1, 1))

print(predictions[:5])
print(actuals[:5])
