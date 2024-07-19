import torch 
import torch.nn as nn
import torch.optim as optim 
print(torch.rand(5))


tensor = torch.Tensor([[[3,2] ,[1,5]],[[2,4],[8,9]],[[0,9],[6,6]]])
#print(tensor)

tensor.device 
print(tensor.device)
 

#print(tensor.shape[2])



linear = nn.Linear(2,2)
input = torch.randn(50,2)
output = linear(input)

relu = nn.ReLU()

op = relu(output)
#print(op )

mlp_layer = nn.Sequential(
    nn.Linear(5,2),
    nn.BatchNorm1d(2),
    nn.ReLU()


)
input_1= torch.randn(5,5) +1
#mlp_layer = mlp_layer(input_1)


adam_opt = optim.Adam(mlp_layer.Parameters(), lr = 1e-1)
print (adam_opt)