import argparse
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from torchvision.utils import save_image
import torch
from torch.autograd import Variable
from torch import autograd
import time as t
from collections import Counter
from stylegan import *
import itertools
import random
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda:0')
epochs=10000
lambda_=10
batch_size = 2
mseloss = nn.MSELoss()

torch.set_default_tensor_type(torch.FloatTensor)
manualSeed = 2
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("loading dat.....")
data= np.load("newdata_t/data.npy").reshape((-1,1,128,128,128))
data = torch.tensor(data[:120]).type(torch.FloatTensor)
pore = np.load("newdata_t/pore.npy").reshape((-1,128))

pore = torch.tensor(pore[:120]).type(torch.FloatTensor)
data_loader = torch.utils.data.DataLoader(data,batch_size=batch_size)
pore_loader = torch.utils.data.DataLoader(pore,batch_size=batch_size)
D=discriminator().to(device)
G=StyleGenerator().to(device)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.99))
g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.99))
optimizer_Q = torch.optim.Adam(itertools.chain(G.parameters(), D.parameters()), lr=0.0002, betas=(0.5, 0.99))
Loss_D_list = [0.0]
Loss_G_list = [0.0]
Loss_Q_list = [0.0]
print("starting training.....")
for ep in range(epochs):
    i = 0
    loss_D_list = []
    loss_G_list = []
    loss_Q_list = []
    D.train()
    G.train()
    d_loss_real = 0
    d_loss_fake = 0

    for real_img, pore_label in zip(data_loader,pore_loader):
        i += 1
        D.zero_grad()
        real_img = real_img.cuda()
        noise = torch.randn(batch_size,384).cuda()

        pore_label = pore_label.cuda()


        real_img = Variable(real_img, requires_grad=True)
        real_out, _ = D(real_img)

        real_scores = real_out

        fake_img = G(noise, pore_label)
        fake_out, _ = D(fake_img.detach())
        fake_scores = fake_out


        alpha = torch.rand((batch_size, 1, 1, 1, 1)).to(device)
        x_hat = alpha * real_img + (1 - alpha) * fake_img
        pred_hat, _ = D(x_hat)
        gradients = \
            torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        d_loss = torch.mean(fake_out) - torch.mean(real_out) + gradient_penalty
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        loss_D_list.append(d_loss.item())

        fake_img = G(noise, pore_label)
        output, _ = D(fake_img)
        output = output.squeeze(1)
        g_loss = torch.mean(-output)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        loss_G_list.append(g_loss.item())
        optimizer_Q.zero_grad()


        gen_imgs = G(noise, pore_label)
        _, pred_pore = D(gen_imgs)

        info_loss = mseloss(pred_pore, pore_label)
        info_loss.backward()
        optimizer_Q.step()
        loss_Q_list.append(info_loss.item())

        print(
            "[Epoch %d/%d] [Batch %d/%d]  [G loss: %f] [D loss: %f] [Q loss: %f]"
            % (ep, epochs, i, len(data_loader), g_loss.item(), d_loss.item(), info_loss.item())
        )


    Loss_G_list.append(np.mean(loss_G_list))
    Loss_D_list.append(np.mean(loss_D_list))
    Loss_Q_list.append(np.mean(loss_Q_list))

    if ep % 5 == 0:
        with torch.no_grad():
            fake_img = G(noise, pore_label)

            save_image(fake_img[0,0,0], 'test_fix_%d.png' % (ep))

    if (ep + 1) % 50 == 0:
        np.savetxt("Loss_D_%d.csv" % (ep), np.array(Loss_D_list))
        np.savetxt("Loss_G_%d.csv" % (ep), np.array(Loss_G_list))
        np.savetxt("Loss_Q_%d.csv" % (ep), np.array(Loss_Q_list))
        torch.save(D.state_dict(), "netD_%d.pth" % (ep))
        torch.save(G.state_dict(), "netG_%d.pth" % (ep))

Loss_D_list = Loss_D_list[1:]
Loss_G_list = Loss_G_list[1:]
Loss_Q_list = Loss_Q_list[1:]
np.savetxt("Loss_D.csv", np.array(Loss_D_list))
np.savetxt("Loss_G.csv", np.array(Loss_G_list))
np.savetxt("Loss_Q.csv", np.array(Loss_Q_list))
torch.save(D.state_dict(), "netD.pth")
torch.save(G.state_dict(), "netG.pth")
