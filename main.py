import lala
from typing import List
from lala.functional import relu


input_features, neurons = 2, 2

class MLP:
    def __init__(self, in_dim: int, out_dim,  layers: List[int]):
        self.layers = [(lala.rand(2, 3, requires_grad=True), lala.rand(2, 1, requires_grad=True))]

    def forward(self, x: lala.Tensor):
        for w, b in self.layers:
            x = w @ x + b
        return x
        
    def fit(self, input_: lala.Tensor, target: lala.Tensor):
        logits = self.forward(input_)

        loss = (target - logits).mean().spow(2)

        loss_  = loss.item()
        loss.backward()
        self.step()
        return loss_
    
    def step(self):
        for w, b in self.layers:
        #update weights and biases
            w -= w.grad
            b -= b.grad
            w.detach()
            b.detach()

            #remove grad for next run
            w.grad = None; b.grad = None


mlp = MLP((2, 3), (2, 3), [])



epoch, batch = 25, 5

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