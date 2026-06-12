import lala
from typing import List
from lala.functional import relu


input_features, neurons = 2, 2

class MLP:
    def __init__(self, in_dim: int, out_dim: int,  emb_dim: int, seq_len):
        self.wu = lala.rand(emb_dim, in_dim, requires_grad=True) #project up weights
        self.bu = lala.rand(emb_dim, seq_len, requires_grad=True) #project up bias
        
        self.wd = lala.rand(out_dim, emb_dim, requires_grad=True) #project down weights
        self.bd = lala.rand(out_dim, seq_len, requires_grad=True) #project down biases   
        

    def parametes(self):
        return [self.wu, self.wd, self.bu, self.bd]

    def forward(self, x: lala.Tensor):
        x = self.wu @ x + self.bu
        x = self.wd @ x + self.bd
        return x
        
    def fit(self, input_: lala.Tensor, target: lala.Tensor):
        logits = self.forward(input_)

        #mean squared error
        loss = (target - logits).mean().spow(2)

        loss_  = loss.item()
        loss.backward()
        self.step()
        return loss_
    
    def step(self):
        for param in self.parametes():
        #update weights and biases
            param -= param.grad
            param.detach()  #this is because we don't want the grad upate ops in the graph

            #remove grad for next run
            param.grad = None


mlp = MLP(3, 3, 3, 1)



epoch, batch = 250, 50

losses = []
for epoch in range(epoch):
    batch_loss = 0.0
    for i in range(batch):
        #input and target
        input_ = lala.rand(3, 1)
        #use the input as a target (teaching the nn to map input to itself)
        target = input_.smul(3)

        batch_loss += mlp.fit(input_, target)
        

    losses.append(batch_loss / batch)
    print(f"epoch {epoch} batch avg loss:", batch_loss/batch)


print("""
This code implements a Single Layer Fully Connected neural net 
The network is trained to map 

Assuming correct lalagrad installation you should see an output of decreasing numbers
which is the loss of the model going down as the net trains multiply an input_ by 2 
      
it also generates graph.html open it with a browser to see your computation graph
""")

#plot the loss over time
import matplotlib.pyplot as plt
plt.plot([x for x in range(epoch+1)], losses, label="lalagrad")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss over epoch")
plt.legend()

plt.show()
