import torch
import numpy as np
import matplotlib.pyplot as plt
# 5 mode guassian demo


# model setting
weight = torch.tensor([2/6, 3/6, 1/6])
mu = torch.tensor([-3, 0, 3])
std = 0.5
# init setting
init = torch.tensor([-3.2, -2.8, -0.2, 0.2, 2.8, 3.2])
iteration = 100
x_axis = torch.linspace(-6, 6, 121)
interval = 0.1
lr = 0.1
alpha = 1.0
size = 30
size_ = 150
bw = 0.25
linewidth = 2

def pdf_kernel(x, particles, mass, bandwidth):
    results = torch.exp( - (x - particles).pow(2) / bandwidth) / np.sqrt(np.pi * bandwidth)
    results = torch.sum(results * mass, dim = 1)
    return results

def pdf_calc(particles): # N * 1 array
    results = torch.exp(- (particles - mu).pow(2) / 2 / std**2) / np.sqrt(2 * np.pi * std**2)
    results = torch.sum(results * weight, dim = 1)
    return results
    

def potential_calc(particles):
    return -torch.log(pdf_calc(particles))
    

def grad_calc(particles):
    p1 =  - (particles - mu) / std**2
    p2 = torch.exp( - (particles - mu).pow(2) / 2 / std**2) * weight
    results = torch.sum(p1 * p2, dim = 1, keepdim = True) / pdf_calc(particles).view(-1, 1)
    return  - results

# ----------------------------------------------- BLOB -----------------------------------------------------------------------------
blob = init.clone().view(-1, 1)
mass_blob = torch.ones(len(blob)) / len(blob)
for i in range(iteration):
    grads = grad_calc(blob)
    sq_dist = torch.cdist(blob, blob, p = 2)**2
    bandwidth_h = sq_dist + torch.diag(torch.diag(sq_dist) + sq_dist.max())
    bandwidth_h = bandwidth_h.min(dim = 1)[0].mean()
    kernel = torch.exp( - sq_dist / bandwidth_h)
    blob = blob - lr * grads + lr * 2 * (
        (blob * kernel.sum(1, keepdim = True) - torch.matmul(kernel, blob)) / kernel.sum(1, keepdim = True) + \
        blob * (kernel / kernel.sum(1)).sum(1, keepdim = True) - torch.matmul(kernel / kernel.sum(1), blob)
    ) / bandwidth_h
# ---------------------------------------------- BLOBBD --------------------------------------------------------------------------
blobbd = init.clone().view(-1, 1)
mass_blobbd = torch.ones(len(blobbd)) / len(blobbd)
for i in range(iteration):
    grads = grad_calc(blobbd)
    sq_dist = torch.cdist(blobbd, blobbd, p = 2)**2
    bandwidth_h = sq_dist + torch.diag(torch.diag(sq_dist) + sq_dist.max())
    bandwidth_h = bandwidth_h.min(dim = 1)[0].mean()
    kernel = torch.exp( - sq_dist / bandwidth_h)
    blobbd = blobbd - lr * grads + lr * 2 * (
        (blobbd * torch.matmul(kernel,mass_blobbd.view(-1,1)) - torch.matmul(kernel * mass_blobbd, blobbd)) / torch.matmul(kernel, mass_blobbd.view(-1,1)) + \
        blobbd * (kernel * mass_blobbd / torch.matmul(kernel, mass_blobbd)).sum(1, keepdim = True) - torch.matmul(kernel * mass_blobbd / torch.matmul(kernel, mass_blobbd), blobbd)
    ) / bandwidth_h
    potential = potential_calc(blobbd) 
    beta = torch.log((mass_blobbd * kernel).sum(1) + 1e-6) + ((kernel * mass_blobbd) / torch.matmul(mass_blobbd,kernel)).sum(1) + potential
    beta_bar = beta - (beta * mass_blobbd).sum()
    mass_blobbd = mass_blobbd * (1 - beta_bar * lr * alpha)
    mass_blobbd = mass_blobbd / mass_blobbd.sum()
# ------------------------------------------------------- plot_figure --------------------------------------------------------------------
ax = plt.figure(figsize=(11.8, 2.8))
plt.subplot(1,2,1)
plt.plot(x_axis.numpy(), pdf_calc(x_axis.view(-1,1)).numpy(), c = 'black', linestyle = '--', label = 'target', zorder = 2)
plt.plot(x_axis.numpy(), pdf_kernel(x_axis.view(-1,1), blob.view(-1), mass_blob, bw).numpy(), c = 'red', linestyle = '-', label = 'kernel density estimation', zorder = 3)
# plt.scatter(blob.view(-1).numpy(), mass_blob.numpy(), s = size, c = 'purple', marker = 'o', edgecolor = 'purple', label = 'particles', zorder = 3)
plt.scatter(blob.view(-1).numpy(), np.ones(len(init)) * 0, s = size, c = 'purple', marker = 's', edgecolor = 'purple', label = 'particle', zorder = 4)
for i in range(len(init)):
    if i == 0:
        plt.vlines(blob.view(-1).numpy()[i], 0, mass_blob.numpy()[i], linewidth = linewidth, label = 'weight of particle', linestyle = '-', zorder = 1)
        plt.scatter(blob.view(-1).numpy()[i], mass_blob.numpy()[i], c = '#1f77b4', s = size_, marker = '_', zorder = 4)
    else:
        plt.vlines(blob.view(-1).numpy()[i], 0, mass_blob.numpy()[i], linewidth = linewidth, linestyle = '-', zorder = 1)
        plt.scatter(blob.view(-1).numpy()[i], mass_blob.numpy()[i], c = '#1f77b4', s = size_, marker = '_', zorder = 4)
plt.xticks([])
# plt.yticks([])
plt.tight_layout()
plt.tick_params(labelsize = 16)
plt.grid()
ax.legend(fontsize = 18, ncol = 4, bbox_to_anchor=(0.012, 0.96), loc=3, borderaxespad = 0)
plt.subplot(1,2,2)
plt.plot(x_axis.numpy(), pdf_calc(x_axis.view(-1,1)).numpy(), c = 'black', linestyle = '--', label = 'target', zorder = 2)
plt.plot(x_axis.numpy(), pdf_kernel(x_axis.view(-1,1), blobbd.view(-1), mass_blobbd, bw).numpy(), c = 'red', linestyle = '-', zorder = 3)
size_list = mass_blobbd.numpy() * len(init) * size
# plt.scatter(blobbd.view(-1).numpy(), mass_blobbd.numpy(), s = size_list, c = 'purple', marker = 'o', edgecolor = 'purple', label = 'particles', zorder = 3)
plt.scatter(blobbd.view(-1).numpy(), np.ones(len(init)) * 0, s = size, c = 'purple', marker = 's', edgecolor = 'purple', label = 'particle', zorder = 4)
for i in range(len(init)):
    plt.vlines(blobbd.view(-1).numpy()[i], 0, mass_blobbd.numpy()[i], linewidth = linewidth, linestyle = '-', zorder = 1)
    plt.scatter(blob.view(-1).numpy()[i], mass_blobbd.numpy()[i], c = '#1f77b4', s = size_, marker = '_', zorder = 4)
# plt.legend(fontsize = 12, loc = 2)
plt.xticks([])
# plt.yticks([])
plt.tight_layout()
plt.tick_params(labelsize = 16)
plt.grid()
plt.savefig('./figures/demo.pdf', bbox_inches = 'tight', dpi = 300)
plt.close()



    

