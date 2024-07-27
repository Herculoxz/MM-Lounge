import jax 
import jax.numpy as jnp 
from jax import random 

class LSTM:
    def __init__(self , input_dim , hidden_dim , key):
        self.hidden_dim = hidden_dim 


        self.Wf = random.normal(key , (hidden_dim , hidden_dim+input_dim)) # wf to be multiplied with output and input x
        self.bf = jnp.zeros ((hidden_dim,1)) # broadasting is used here 


        self.Wi = random.normal(key , (hidden_dim , hidden_dim+input_dim))
        self.bi = jnp.zeros((hidden_dim , 1))

        self.Wc = random.normal(key , (hidden_dim , hidden_dim+input_dim))
        self.bc = jnp.zeros((hidden_dim , 1))

        self.Wo = random.normal(key , (hidden_dim , hidden_dim+input_dim))
        self.bc = jnp.zeros((hidden_dim , 1))


        def sigmoid(self , x):
            return jax.nn.sigmoid(x)
        

        def tanh(self,x):
            return jnp.tanh(x)
        
        def forward (self , x , h_prev , C_prev):
            concat = jnp.vstaack((h_prev ,x))


            Ft = self.sigmoid(jnp.dot(self.Wf , concat)+ self.bf)

            It = self.sigmoid(jnp.dot(self.Wi , concat)+ self.bi)

            Ot = self.sigmoid(jnp.dot(self.Wo , concat)+ self.bo)

            C_ht = self.tanh(jnp.dot(self.Wc , concat)+ self.bc)

            Ct = Ft * C_prev + It * C_ht

            Ht = Ot * self.tanh(Ct)


            return Ht , Ct
        




# lstm = LSTM(input_dim , hidden_dim , key)

# key = random.PRNGKey(0)


        















