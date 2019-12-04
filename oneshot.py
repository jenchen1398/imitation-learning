import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from bipedal_ds import BipedalDataset
#import mlb
import glob

examples = [201, 401, 601, 801, 1001, 1201]
data_paths = ["~/compbrain/imitation-learning/walker_new{}/".format(num) for num in examples]
#data_paths = glob.glob('~/compbrain/imitation-learning/walker_new*')

ds = BipedalDataset(data_paths)
dataloader = DataLoader(ds, batch_size=1)


im_height = 135
im_width = 150
n_actions = 4
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class VisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        state_len = 24
        bias_len = 10

        h = 40
        ker = 3
        self.stride = 2
        self.conv1 = nn.Conv2d(3,h,kernel_size=ker,stride=self.stride) # RGB -> h with kernel of 3
        self.conv2 = nn.Conv2d(h,h,kernel_size=ker,stride=self.stride)
        self.conv3 = nn.Conv2d(h,h,kernel_size=ker,stride=self.stride)


        def outshape(size, ker=ker, stride=self.stride):
            return (size - (ker - 1) - 1) // stride  + 1
        width = outshape(outshape(outshape(im_width)))
        height = outshape(outshape(outshape(im_height)))

        #self.spatial_softmax = SpatialSoftmax(height,width,h) # takes height,width,numchannels

        fc_inshape = width*height*h + state_len + bias_len
        #breakpoint()
        self.fc1 = nn.Linear(fc_inshape, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, n_actions)

        self.bias = nn.Parameter(torch.empty(10).uniform_(-1,1)) # arbitrarily decided on the initialization


    def forward(self, img, state_vec, weights=None):
        assert img.shape[1] == 3 # RGB. shape[0] should be batch dim i think
        if weights is None:
            img = F.relu(self.conv1(img))
            img = F.relu(self.conv2(img))
            img = F.relu(self.conv3(img))
            #img = self.spatial_softmax(img)
            flat_img = img.view(img.shape[0],-1) # flatten except batch dim
            bias = self.bias.unsqueeze(0).expand(state_vec.shape[0],self.bias.shape[0]) # expand to (B,10)
            img_and_state = torch.cat([flat_img, state_vec,bias],dim=1)
            img_and_state = F.relu(self.fc1(img_and_state))
            img_and_state = F.relu(self.fc2(img_and_state))
            img_and_state = F.relu(self.fc3(img_and_state))
            img_and_state = torch.tanh(self.fc4(img_and_state))
            return img_and_state
        else:
            img = F.relu(F.conv2d(img,weight=weights['conv1.weight'],bias=weights['conv1.bias'],stride=self.stride))
            img = F.relu(F.conv2d(img,weight=weights['conv2.weight'],bias=weights['conv2.bias'],stride=self.stride))
            img = F.relu(F.conv2d(img,weight=weights['conv3.weight'],bias=weights['conv3.bias'],stride=self.stride))
            flat_img = img.view(img.shape[0],-1)
            #img = self.spatial_softmax(img) # parameterless so it's ok
            bias = weights['bias']
            bias = bias.unsqueeze(0).expand(state_vec.shape[0],bias.shape[0])
            img_and_state = torch.cat([flat_img, state_vec, bias],dim=1)
            img_and_state = F.relu(F.linear(img_and_state,weight=weights['fc1.weight'],bias=weights['fc1.bias']))
            img_and_state = F.relu(F.linear(img_and_state,weight=weights['fc2.weight'],bias=weights['fc2.bias']))
            img_and_state = F.relu(F.linear(img_and_state,weight=weights['fc3.weight'],bias=weights['fc3.bias']))
            img_and_state = torch.tanh(F.linear(img_and_state,weight=weights['fc4.weight'],bias=weights['fc4.bias']))
            return img_and_state

#model = VisualModel()
#model = model.cuda()


#images, states, actions = ds[0]
#images = images.cuda()
#states = states.cuda()
#actions = actions.cuda()

#with mlb.debug():
#    x = model(images[0:16],states[:16])
#breakpoint()





from plotting import VisdomLinePlotter

class EpochLog:
    def __init__(self, port, frequency=10, name="Network Loss"):
        self.visdom = VisdomLinePlotter(port=port)
        self.frequency = frequency

        self.name = name

    def message(self, epoch, loss):
        if epoch % self.frequency == 0:
            self.visdom.plot("Loss", "Training Loss", self.name, epoch, loss, xlabel='epochs')
            print("Epoch {} loss: {}".format(epoch, loss))




def main_nomaml():
    model = VisualModel()
    loss_fn = nn.MSELoss()

    epochs = 1000
    #lr = 0.01
    lr = 0.001
    meta_lr = 0.001
    K = 64
    test_K = 32

    #optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = EpochLog(8097, frequency=100, name='MAML Network Loss')

    images, states, actions = ds[0]
    for epoch in range(epochs):
        #total_loss = 0

        # images: (T, H, W, 3)
        # states: (T, 24)
        # actions (T, 4)

        assert images.shape[0] >= K + test_K and states.shape[0] >= K + test_K and actions.shape[0] >= K + test_K

        indices = np.random.choice(range(images.shape[0]), K + test_K, replace=False)
        images, states, actions = images[indices], states[indices], actions[indices]
        #images, states, actions = images[indices].cuda(), states[indices].cuda(), actions[indices].cuda()

        train_images, train_states, train_actions = images[:K], states[:K], actions[:K]

        pred = model(train_images,train_states)
        loss = loss_fn(train_actions, pred)

        #grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        #weights = [param - lr * grad for param, grad in zip(model.parameters(), grads)]

        #test_images, test_states, test_labels = images[K:K+test_K], states[K:K+test_K], actions[K:K+test_K]

        #pred = model(test_images, weights)
        #loss = loss_fn(test_actions, pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.message(epoch, loss.item())


def main():
    model = VisualModel().cuda()
    loss_fn = nn.MSELoss()

    epochs = 1000
    lr = 0.01
    #lr = 0.001
    meta_lr = 0.001
    #K = 64
    #test_K = 32
    K = 30
    test_K = 32
    interval = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = EpochLog(1234, frequency=1, name='MAML Network Loss')

    if not os.path.exists('models'):
        os.mkdir('models')

    for epoch in range(epochs):
        total_loss = 0

        #for images, states, actions in dataloader:
        for images, states, actions in ds:

            # images: (T, H, W, 3)
            # states: (T, 24)
            # actions (T, 4)

            assert images.shape[0] >= K + test_K and states.shape[0] >= K + test_K and actions.shape[0] >= K + test_K

            indices = np.random.choice(range(images.shape[0]), K + test_K, replace=False)
            #images, states, actions = images[indices], states[indices], actions[indices]
            images, states, actions = images[indices].cuda(), states[indices].cuda(), actions[indices].cuda()

            train_images, train_states, train_actions = images[:K], states[:K], actions[:K]

            pred = model(train_images,train_states)
            loss = loss_fn(train_actions, pred)

            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            weights = {name:(param - lr * grad) for (name,param), grad in zip(model.named_parameters(), grads)}

            test_images, test_states, test_actions = images[K:K+test_K], states[K:K+test_K], actions[K:K+test_K]

            pred = model(test_images, test_states, weights=weights)
            loss = loss_fn(test_actions, pred)
            total_loss += loss

        total_loss = total_loss / len(ds)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        log.message(epoch, total_loss.item())
        
        if epoch % interval == 0:
            torch.save(model, "models/model-{}.pth".format(epoch))


#with mlb.debug():
main()