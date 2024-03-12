from sys import path
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

apps = Path(__file__)
while apps.parent.name != 'apps':
    apps = apps.parent
path.append(str(apps.parent.parent))
del apps
from apps import APPS_FOLDER, DATA_FOLDER, MODELS_FOLDER

from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node

device = 'cuda'


def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    return torch.mean(0.5 * torch.sum(z**2, dim=(1, )) - jac) / z.shape[1]


def nf_head(n_feat, n_coupling_blocks, clamp_alpha, fc_internal, dropout):
    nodes = list()
    nodes.append(InputNode(n_feat, name='input'))
    print('input_size: {}'.format(n_feat))
    print('n_coupling_blocks: {}'.format(n_coupling_blocks))
    print('clamp_alpha: {}'.format(clamp_alpha))
    print('fc_internal: {}'.format(fc_internal))
    print('dropout: {}'.format(dropout))

    for k in range(n_coupling_blocks):
        nodes.append(
            Node([nodes[-1].out0],
                 permute_layer, {'seed': k},
                 name=F'permute_{k}'))
        nodes.append(
            Node(
                [nodes[-1].out0],
                glow_coupling_layer, {
                    'clamp': clamp_alpha,
                    'F_class': F_fully_connected,
                    'F_args': {
                        'internal_size': fc_internal,
                        'dropout': dropout
                    }
                },
                name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    # print number of parameters
    n_params = sum([p.numel() for p in coder.parameters()])
    print('n_params: {}'.format(n_params))

    return coder


def train(traindata, testdata):
    model = nf_head(720896, 2, 3, 32, 0.0)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0005,
                                 betas=(0.8, 0.8),
                                 eps=1e-04,
                                 weight_decay=1e-5)

    model.to(device)

    batch_size = 64

    # Create a DataLoader
    data_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(testdata, batch_size=4, shuffle=False)

    # train some epochs

    for epoch in range(1000):
        model.train()
        train_loss = list()
        test_loss = list()
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            inputs = batch.to(device)
            #print(inputs.size())
            z = model(inputs)
            loss = get_loss(z, model.jacobian(run_forward=False))
            train_loss.append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()

        mean_train_loss = np.mean(train_loss)
        if True:
            print('Epoch: {:d} \t train loss: {:.4f}'.format(
                epoch, mean_train_loss))

        if epoch % 10 == 0:
            model.eval()
            for i, batch in enumerate(data_loader):

                inputs = batch.to(device)

                z = model(inputs)
                loss = get_loss(z, model.jacobian(run_forward=False))
                test_loss.append(loss.cpu().data.numpy())
            mean_test_loss = np.mean(test_loss)
            print('Epoch: {:d} \t TEST LOSS: {:.4f}'.format(
                epoch, mean_test_loss))


class CustomDataset(Dataset):

    def __init__(self, npy_file):
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


if __name__ == '__main__':
    mvtec = DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/')
    train_good = mvtec.joinpath('./train-good-CNxV2_DepthAE_BnQn1d.npy')
    train_good2 = mvtec.joinpath('./train-good-CNxV2.npy')
    test_good = mvtec.joinpath('./test-good-CNxV2_DepthAE_BnQn1d.npy')
    test_defect = mvtec.joinpath('./test-defect-CNxV2_DepthAE_BnQn1d.npy')
    test_defect2 = mvtec.joinpath('./test-defect-CNxV2.npy')
    train_data = CustomDataset(train_good2)
    test_data = CustomDataset(test_defect2)

    print(train_data)
    #temp = np.load(train_good)
    train(train_data, test_data)
