import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        out = self.linear(x)
        return out.round()


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

            output = model(x)
            output = output.flatten()
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if batch_num % 40 == 0:
                print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))

    return model



