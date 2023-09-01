import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features ,out_features):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()  # Adding sigmoid activation function

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)  # Apply sigmoid
        return out.round()  # Now this will be either 0 or 1


def train_mlp(model,train_loader,device="cpu",epochs=10) -> MLP:

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device).float()
            print(x)
            print(y)

            output = model(x)
            print(output)
            output = output.flatten()
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if batch_num % 40 == 0:
                print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))

    model.eval()
    return model



