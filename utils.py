import math, os

import numpy as np

import torch


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


class Writer(object):
    def __init__(self, env, seed, weight, epoch, prior, traj_size, fname='', folder='PU_log', pbound='0.0', noise=0.0):
        if weight: 
            label = '_weight'
        else: 
            label = ''
        if fname != '':
            fname = '_{}'.format(fname)
        if prior > 1e-6:
            plabel = '_{:.4f}'.format(prior)
        else:
            plabel = ''
        if pbound != '0.0':
            pblabel = '_{}'.format(pbound)
        else:
            pblabel = ''
        if noise < 1e-6:
            nlabel = ''
        else:
            nlabel = '_noise{:.2f}'.format(noise)

        self.fname = '{}_{}{}_{}{}_{}{}{}{}.csv'.format(env, seed, label, epoch, plabel, traj_size, pblabel, fname, nlabel)
        self.folder = folder
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        if os.path.exists('{}/{}'.format(self.folder, self.fname)):
            print('Overwrite {}/{}!'.format(self.folder, self.fname))
            os.remove('{}/{}'.format(self.folder, self.fname))

    def log(self, epoch, reward):
        with open(self.folder + '/' + self.fname, 'a') as f:
            f.write('{},{}\n'.format(epoch, reward))

def digitize(arr, unit):
    if unit < 1e-6:
        return arr
    return np.round(arr / unit) * unit

def save_model(model, name, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder + name) 
