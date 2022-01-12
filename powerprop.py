#-*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

def train_models(nets, trainloader, criterion, optimizers, epoch, log_interval):
    #stats_all_loss = [[] for _ in range(len(nets))]
    #stats_all_acc = [[] for _ in range(len(nets))]
    stats_all_loss = [0. for _ in range(len(nets))]
    stats_all_acc = [0. for _ in range(len(nets))]
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.view(len(inputs), -1)

        for n_id, net in enumerate(nets):
            optimizers[n_id].zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizers[n_id].step()
            # loss
            #stats_all_loss[n_id].append(loss.item())
            stats_all_loss[n_id] += loss.item()
            # accuracy
            preds = outputs.detach().argmax(dim=1, keepdim=False)
            acc = (preds == labels).type(torch.float32).mean().item()
            #stats_all_acc[n_id].append(acc)
            stats_all_acc[n_id] += acc

        if (i+1) % log_interval == 0:
            for n_id, net in enumerate(nets):
                print('[epoch %d, batch %5d, alpha %f] loss: %.3f / acc: %.3f' % (epoch, i, net.alpha,
                    stats_all_loss[n_id] / log_interval, stats_all_acc[n_id] / log_interval))
            print('---')
            stats_all_loss = [0. for _ in range(len(nets))]
            stats_all_acc = [0. for _ in range(len(nets))]

def eval_model(net, masks, testloader, criterion):
    test_acc = 0.
    test_loss = 0.
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.view(len(inputs), -1)
            outputs = net(inputs, masks)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs.detach().data, 1)
            test_acc += (preds == labels).type(torch.float32).mean().item()
    test_acc = test_acc / len(testloader)
    test_loss = test_loss / len(testloader)
    net.train()
    return test_acc, test_loss

def _bottom_k_mask(percent_to_keep, condition):
    how_many = int(percent_to_keep * condition.size()[0])
    top_k = torch.topk(condition, k=how_many)
    mask = np.zeros(shape=condition.shape, dtype=np.float32)
    mask[top_k.indices.numpy()] = 1
    assert np.sum(mask) == how_many
    return mask

def prune_by_magnitude(percent_to_keep, weight):
    mask = _bottom_k_mask(percent_to_keep, np.abs(weight.view(-1, )))
    return mask.reshape(weight.shape)

class PowerPropVarianceScaling():
    def __init__(self, alpha, *args, **kwargs):
        super(PowerPropVarianceScaling, self).__init__(*args, **kwargs)
        self._alpha = alpha

    def __call__(self, shape, dtype):
        u = super(PowerPropVarianceScaling, self).__call__(shape, dtype)

        return torch.sign(u) * torch.pow(torch.abs(u), 1.0 / self._alpha)

def powerpropvariancescaling(module, alpha, init_fn):
    init_fn(module)
    with torch.no_grad():
        param_modified = torch.sign(param) * torch.pow(torch.abs(param), 1.0 / self._alpha)
        param.copy_(param_modified)
    return param
    

class PowerPropLinear(nn.Linear):
    """Powerpropagation Linear module."""
    def __init__(self, in_features, out_fetaures, alpha, bias=True, *args, **kwargs):
        self._alpha = alpha
        super(PowerPropLinear, self).__init__(in_features, out_fetaures, bias, *args, **kwargs)

    def reset_parameters(self):
        super(PowerPropLinear, self).reset_parameters()
        with torch.no_grad():
            weight = self.weight
            weight_modified = torch.sign(weight) * torch.pow(torch.abs(weight), 1.0 / self._alpha)
            self.weight.copy_(weight_modified)
        
    def get_weights(self):
        return torch.sign(self.weight) * torch.pow(torch.abs(self.weight), self._alpha)

    def forward(self, inputs, mask=None):
        params = self.weight * torch.pow(torch.abs(self.weight), self._alpha - 1)

        if mask is not None:
            params *= mask

        outputs = F.linear(inputs, params, self.bias)
        return outputs

class MLP(nn.Module):
    """A multi-layer perceptron module."""
    def __init__(self, alpha, output_sizes=[300, 100, 10], input_dim=784):
        super(MLP, self).__init__()
        self._alpha = alpha
        dims = [input_dim,] + output_sizes
        self._layers = []

        for i in range(1, len(dims)):
            self._layers.append(PowerPropLinear(dims[i-1], dims[i], alpha))
        self._layers = nn.ModuleList(self._layers)

    @property
    def alpha(self):
        return self._alpha

    def get_weights(self):
        return [l.get_weights().detach() for l in self._layers]

    def forward(self, inputs, masks=None):
        num_layers = len(self._layers)
        
        for i, layer in enumerate(self._layers):
            if masks is not None:
                inputs = layer(inputs, masks[i])
            else:
                inputs = layer(inputs)
            if i < (num_layers - 1):
                inputs = F.relu(inputs)
        return inputs

# hyper params
batch_size = 128
epochs = 20
lr = 0.01
log_interval = 100

# experiment configurations for different models
alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
model_types = []
nets = []
n_models = len(alphas)
for alpha in alphas:
    nets.append(MLP(alpha))
    if alpha > 1.0:
        model_types.append('Power. ($\\alpha={}$)'.format(alpha))
    else:
        model_types.append('Baseline')

initial_weights = [m.get_weights() for m in nets]
optimizers = [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9) for m in nets]

# loss fn
criterion = nn.CrossEntropyLoss()

# get mnist data
transform = transforms.Compose([transforms.ToTensor(),])
data_path = os.path.dirname(os.path.realpath(__file__)) + '/mnist'
trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# train model
for epoch in range(epochs):
    stats = train_models(nets, trainloader, criterion, optimizers, epoch, log_interval)

final_weights = [m.get_weights() for m in nets]
eval_at_sparsity_level = np.geomspace(0.01, 1.0, 20).tolist()
acc_at_sparsity = [[] for _ in range(len(nets))]

num_layers = len(final_weights[0])
for p_to_use in eval_at_sparsity_level:
    # half the sparsity at output layer
    percent = (num_layers - 1)*[p_to_use] + [min(1.0, p_to_use*2)]

    for n_id, net in enumerate(nets):
        masks = []
        for i, w in enumerate(final_weights[n_id]):
            masks.append(prune_by_magnitude(percent[i], w))
        test_acc, test_loss = eval_model(net, masks, testloader, criterion)
        acc_at_sparsity[n_id].append(test_acc)
        print('Performance @ {:1.0f}% of weights [Alpha {}]: Acc {:1.3f} Loss {:1.3f} '.format(\
            100*p_to_use, net.alpha, test_acc, test_loss))
    print('---')

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for acc, label in zip(acc_at_sparsity, model_types):
    ax.plot(eval_at_sparsity_level, acc, label=label, marker='o', lw=2)

ax.set_xscale('log')
ax.set_xlim([1.0, 0.01])
ax.set_ylim([0.0, 1.0])
ax.legend(frameon=False)
ax.set_xlabel('Weights Remaining (%)')
ax.set_ylabel('Test Accuracy (%)')

fig.savefig('results.pdf', format='pdf')
