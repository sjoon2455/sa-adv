#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import trange
from collections import Counter
from sa.model import MnistClassifier
from vae.model import VAE
from datetime import datetime
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np
import matplotlib.pyplot as plt
import ast
from keras import layers, models, datasets, backend
import keras
import foolbox
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from scipy.stats import gaussian_kde


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"


dataset = 'mnist'
if dataset == 'mnist':
    model = LeNet()
    model.load_state_dict(torch.load('LeNet'))
    model.eval()
else:

    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'resnet50', pretrained=True)
    model.eval()

fmodel = PyTorchModel(model, bounds=(0, 1))
images, labels = ep.astensors(
    *samples(fmodel, dataset=dataset, batchsize=20))

img_size = 28*28*1
torch.no_grad()  # since nothing is trained here
vae_model_path = './vae/models/MNIST_EnD.pth'
classifier_model_path = './sa/models/MNIST_conv_classifier.pth'

vae = VAE(img_size=28*28, h_dim=1600, z_dim=400)
vae.load_state_dict(torch.load(vae_model_path))
vae.eval()
vae.cuda()

classifier = MnistClassifier(img_size=img_size)
classifier.load_state_dict(torch.load(classifier_model_path))
classifier.eval()
classifier.cuda()

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=False)
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True)

sa_layer = 7
layer_dim = 512
all_ats = torch.zeros(0, layer_dim)
for i, (x, x_class) in enumerate(train_data_loader):
    x = x.cuda()
    ats = classifier.at_by_layer(x, sa_layer).detach()
    all_ats = torch.cat([all_ats, ats.cpu()], dim=0)
all_ats = all_ats.transpose(0, 1).numpy()
rem_cols = np.std(all_ats, axis=1) < 0.6
ref_all_ats = all_ats[~rem_cols]
ref_all_ats = ref_all_ats[:100]
our_kde = gaussian_kde(ref_all_ats)


def calc_lsa(at, kde):
    return -kde.logpdf(at)


def calc_img_lsa(img):
    pr_at = classifier.at_by_layer(img, sa_layer).detach()
#     pr_at = classifier.up2lyr2(img).detach()
    pr_at = pr_at.cpu().numpy().transpose()
    pr_at = pr_at[~rem_cols][:100]
    return calc_lsa(pr_at, our_kde)


def calc_z_lsa(z):
    vae_img = vae.decode(z).view(-1, 1, 28, 28)
    return calc_img_lsa(vae_img)


def z_to_img(z):
    return vae.decode(z).detach()


def main(batch, epsilons):

    attacks = [
        fa.L2FastGradientAttack(),
        fa.L2DeepFoolAttack(),
        # fa.L2CarliniWagnerAttack(),
        fa.DDNAttack(),

        fa.LinfBasicIterativeAttack(),
        fa.LinfFastGradientAttack(),
        fa.LinfDeepFoolAttack(),
        fa.LinfPGD(),
    ]
    total_drop, total_recovered = 0, 0
    attacks_result = [0 for _ in range(len(attacks))]
    drop_result = [0 for _ in range(len(attacks))]

    for n, attack_1 in enumerate(attacks):
        ori_predictions = fmodel(images).argmax(axis=-1)
        raw_advs, _, _ = attack_1(
            fmodel, images, labels, epsilons=epsilons)
        raw_advs = raw_advs[0]
        adv_predictions = fmodel(raw_advs).argmax(axis=-1)
        drop, recovered = 0, 0
        for i in range(20):
            if ori_predictions[i].raw.cpu().numpy() == adv_predictions[i].raw.cpu().numpy():
                drop += 1
                continue  # attack failed at all
            elif ori_predictions[i].raw.cpu().numpy() != labels[i].raw.cpu().numpy():
                drop += 1
                continue
            else:
                samp_img = raw_advs[i].raw.cpu().reshape(1, 1, 28, 28)
                samp_class = np.asscalar(
                    adv_predictions[i].raw.cpu().numpy())
                final_prediction = predict(
                    samp_img, samp_class, ori_predictions[i].raw.cpu().numpy())
                if final_prediction == ori_predictions[i]:
                    recovered += 1
        drop_result[n] = 20-drop
        attacks_result[n] = recovered
        total_recovered += recovered
        total_drop += drop
    print("-------------------------------------------------------")
    print("Dataset(epsilon = {0}): ".format(epsilons[0]), dataset)
    print("recovery matrix: ", attacks_result)
    print(" total  matrix:  ", drop_result)
    print("recovery rate: ", total_recovered /
          (batch*len(attacks)-total_drop))
    print("-------------------------------------------------------")

# given adversarial image and wrong prediction


def predict(samp_img, samp_class, ans):
    # filename = "./data2/4/" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + ".txt"
    # f = open(filename, 'w')
    ### GA Params ###
    gen_num = 7
    pop_size = 30
    best_left = 15
    mut_size = 0.1

    img_enc, _ = vae.encode(samp_img.view(-1, img_size).cuda())
    ### Initialize optimization ###
    init_pop = [img_enc + 0.7 * torch.randn(1, 400).cuda()
                for _ in range(pop_size)]
    now_pop = init_pop
    prev_best = -1
    binom_sampler = torch.distributions.binomial.Binomial(
        probs=0.5*torch.ones(img_enc.size()))

    ### gogo GA !!! ###
    for _ in range(gen_num):
        indivs = torch.cat(now_pop, dim=0)
        dec_imgs = vae.decode(indivs).view(-1, 1, 28, 28)
        all_logits = classifier(dec_imgs)

        indv_score = [999 if all_logits[(i_idx, samp_class)] != max(all_logits[i_idx])
                      else calc_z_lsa(indivs[i_idx])
                      for i_idx in range(pop_size)]
        all_logits = all_logits.detach().cpu().numpy()
        indv_predictions = np.argmax(all_logits, -1)
        # f.write(str(indv_predictions)+'\n')

        best_idxs = sorted(range(len(indv_score)),
                           key=lambda i: indv_score[i])[-best_left:]
        now_best = max(indv_score)
        if now_best == prev_best:
            mut_size *= 0.7
        else:
            mut_size = 0.1
        parent_pop = [now_pop[idx] for idx in best_idxs]

        k_pop = []
        for _ in range(pop_size-best_left):
            mom_idx, pop_idx = np.random.choice(
                best_left, size=2, replace=False)
            spl_idx = np.random.choice(400, size=1)[0]
            k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx],
                                parent_pop[pop_idx][:, spl_idx:]], dim=1)  # crossover

            # mutation
            diffs = (k_gene != img_enc).float()
            # random adding noise only to diff places
            print('before: ', k_gene)
            k_gene += mut_size * torch.randn(k_gene.size()).cuda() * diffs
            print(torch.randn(k_gene.size()))
            print('before: ', k_gene)
            # random matching to img_enc
            interp_mask = binom_sampler.sample().cuda()
            k_gene = interp_mask * img_enc + (1 - interp_mask) * k_gene

            k_pop.append(k_gene)
        now_pop = parent_pop + k_pop
        prev_best = now_best
        if mut_size < 1e-3:
            break  # that's enough and optim is slower than I expected
    indivs = torch.cat(now_pop, dim=0)
    dec_imgs = vae.decode(indivs).view(-1, 1, 28, 28)
    all_logits = classifier(dec_imgs)
    all_logits = all_logits.detach().cpu().numpy()
    indv_predictions = np.argmax(all_logits, -1)
    f.write(str(indv_predictions)+'\n')
    f.write(str(ans)+'\n')
    f.close()
    return Counter(indv_predictions).most_common()[0][0]
    # final_bound_img = vae.decode(parent_pop[-1])
    # final_bound_img = final_bound_img.reshape(1, 1, 28, 28)
    # final_bound_img = final_bound_img.detach()
    # predictions = fmodel(final_bound_img).argmax(axis=-1)

    # return np.asscalar(predictions[0].raw.cpu().numpy())
    # all_img_lst.append(final_bound_img)

    # all_imgs = np.vstack(all_img_lst)
    # np.save('bound_imgs_MNIST.npy', all_imgs)


# for epsilon in [0.01, 0.1, 0.8]:
for epsilon in [0.8]:
    main(20, [epsilon])
