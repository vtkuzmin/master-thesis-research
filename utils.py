import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np
from scipy.stats.stats import pearsonr   

import matplotlib.pyplot as plt
from IPython import display


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


class Autoencoder(nn.Module):
    ''' 
        Simple linear autoencoder
    '''
    def __init__(self, hidden_size=64):
        super(Autoencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, self.hidden_size),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
def train_autoencoder(model, criterion, optimizer, train_loder, test_loader, num_epochs=100):
    train_loss = []
    test_loss = []
    
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            train_mse_per_epoch = []
            test_loss_per_epoch = []
            
            for data in train_loder:
                img, _ = data
                img = img.view(img.size(0), -1)
                img = Variable(img) #.cuda()
                # ===================forward=====================
                output = model(img)
                loss = criterion(output, img)
                MSE_loss = nn.MSELoss()(output, img)
                train_mse_per_epoch.append(MSE_loss.item())
                train_loss_per_epoch.append(loss.item())
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_loss.append(np.mean(train_loss_per_epoch))
            
            with torch.no_grad():
                for data in test_loader:
                    img, _ = data
                    img = img.view(img.size(0), -1)
                    img = Variable(img) #.cuda()
                
                    output = model(img)
                    loss = criterion(output, img)
                    test_loss_per_epoch.append(loss.item())
                
                test_loss.append(np.mean(test_loss_per_epoch))
                    
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()


            print('epoch [{}/{}], train loss:{:.4f}, train MSE_loss:{:.4f}, test loss:{:.4f}, best test loss: {:.4f}'
                  .format(epoch + 1, num_epochs, train_loss[-1], np.mean(train_mse_per_epoch), test_loss[-1],
                         best_test_loss))
            if epoch % 10 == 0:
                x = to_img(img.data)
                x_hat = to_img(output.data)
                save_image(x, './mlp_img/x_{}.png'.format(epoch))
                save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))

            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(model.state_dict(), f"./linear_autoencoder_{model.hidden_size}.pth")
            
    except KeyboardInterrupt:
        pass

    
class Binary_func_ideal(nn.Module):
    def __init__(self, unput_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(unput_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 2))

    def forward(self, x):
        x = self.net(x)
        return x
    
    
def train_ideal_binary_model(boolean_function, criterion, optimizer, 
                             train_loder, test_loader, save_name, num_epochs=60):
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            test_loss_per_epoch = []
            correct = 0
            total = 0
            for data in train_loder:
                img, label = data
                boolean_mask = label.numpy()%2==0
                boolean_mask = torch.Tensor(boolean_mask.astype(int)).long()

                img = img.view(img.size(0), -1)

                output = boolean_function(img)

                loss = criterion(output, boolean_mask)
                train_loss_per_epoch.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += boolean_mask.size(0)
                correct += (predicted == boolean_mask).sum().item()

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(train_loss_per_epoch))

            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for data in test_loader:
                    img, label = data
                    boolean_mask = label.numpy()%2==0
                    boolean_mask = torch.Tensor(boolean_mask.astype(int)).long()

                    img = img.view(img.size(0), -1)
                    output = boolean_function(img)

                    loss = criterion(output, boolean_mask)
                    test_loss_per_epoch.append(loss.item())

                    _, predicted = torch.max(output.data, 1)
                    test_total += boolean_mask.size(0)
                    test_correct += (predicted == boolean_mask).sum().item()

                test_loss.append(np.mean(test_loss_per_epoch))
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()

            print('epoch [{}/{}], train loss:{:.4f}, train acc:{:.4f}, test loss:{:.4f}, test acc:{:.4f}, best_test_loss :{:.4f}'
                  .format(epoch + 1, num_epochs, np.mean(train_loss_per_epoch), 100 * correct / total,
                          np.mean(test_loss_per_epoch), 100 * test_correct / test_total, best_test_loss))

            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(boolean_function.state_dict(), f"./{save_name}.pth")

    except KeyboardInterrupt:
        pass    


    
class Binary_func(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 2))

    def forward(self, x):
        x = self.net(x)
        return x
    
    
def train_binary_model(boolean_function, criterion, optimizer, train_loder, test_loader, model, save_name, num_epochs=60):
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            test_loss_per_epoch = []
            correct = 0
            total = 0
            for data in train_loder:
                img, label = data
                boolean_mask = label.numpy()%2==0
                boolean_mask = torch.Tensor(boolean_mask.astype(int)).long()

                img = img.view(img.size(0), -1)

                with torch.no_grad():
                    encodding = model.encoder(img)

                output = boolean_function(encodding)

                loss = criterion(output, boolean_mask)
                train_loss_per_epoch.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += boolean_mask.size(0)
                correct += (predicted == boolean_mask).sum().item()

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(train_loss_per_epoch))

            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for data in test_loader:
                    img, label = data
                    boolean_mask = label.numpy()%2==0
                    boolean_mask = torch.Tensor(boolean_mask.astype(int)).long()

                    img = img.view(img.size(0), -1)

                    encodding = model.encoder(img)
                    output = boolean_function(encodding)

                    loss = criterion(output, boolean_mask)
                    test_loss_per_epoch.append(loss.item())

                    _, predicted = torch.max(output.data, 1)
                    test_total += boolean_mask.size(0)
                    test_correct += (predicted == boolean_mask).sum().item()

                test_loss.append(np.mean(test_loss_per_epoch))
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()

            print('epoch [{}/{}], train loss:{:.4f}, train acc:{:.4f}, test loss:{:.4f}, test acc:{:.4f}, best_test_loss :{:.4f}'
                  .format(epoch + 1, num_epochs, np.mean(train_loss_per_epoch), 100 * correct / total,
                          np.mean(test_loss_per_epoch), 100 * test_correct / test_total, best_test_loss))

            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(boolean_function.state_dict(), f"./{save_name}.pth")

    except KeyboardInterrupt:
        pass    


class Cont_func(nn.Module):
    def __init__(self, hid_size=16):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(64, hid_size),
            nn.ReLU(True),
            nn.Linear(hid_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x
    
    
    
def train_cont_model(continuous_function, criterion, optimizer, train_loder, test_loader, model, save_name, num_epochs=60):
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            test_loss_per_epoch = []

            for data in train_loder:
                img, label = data
                img = img.view(img.size(0), -1)
                target = torch.mean(img, dim=1).unsqueeze(1)

                with torch.no_grad():
                    encodding = model.encoder(img)

                output = continuous_function(encodding)

                loss = criterion(output, target)
                train_loss_per_epoch.append(loss.item())

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(train_loss_per_epoch))

            with torch.no_grad():
                for data in test_loader:
                    img, label = data
                    img = img.view(img.size(0), -1)
                    target = torch.mean(img, dim=1).unsqueeze(1)

                    encodding = model.encoder(img)
                    output = continuous_function(encodding)

                    loss = criterion(output, target)
                    test_loss_per_epoch.append(loss.item())

                test_loss.append(np.mean(test_loss_per_epoch))
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()

            print('epoch [{}/{}], train loss:{:.6f}, test loss:{:.6f}, best_test_loss:{:.6f}'
                  .format(epoch + 1, num_epochs, np.mean(train_loss_per_epoch), np.mean(test_loss_per_epoch),
                         best_test_loss))

            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(continuous_function.state_dict(), f"./{save_name}.pth")

    except KeyboardInterrupt:
        pass
    
    
class Params_func(nn.Module):
    def __init__(self, param_dim, latent_dim=64):
        super().__init__()
        self.param_dim = param_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim+self.param_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.ReLU(True))

    def forward(self, x, p):
        input_x = torch.cat([x, p], dim=1)
        output = self.net(input_x)
        return output
    
    
class Params_func_hard(nn.Module):
    def __init__(self, input_param_dim, param_dim, latent_dim=64):
        super().__init__()
        self.param_dim = param_dim
        
        self.encode = nn.Linear(input_param_dim, self.param_dim)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim+self.param_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.ReLU(True))

    def forward(self, x, p):
        p = self.encode(p)
        input_x = torch.cat([x, p], dim=1)
        output = self.net(input_x)
        return output


def train_param_model(parametric_function, criterion, optimizer, train_loder, test_loader, model, save_name, num_epochs=100):
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            test_loss_per_epoch = []

            for data in train_loder:
                img, label = data
                ps = torch.zeros((img.shape[0], 28))
                target = []
                for i in range(img.shape[0]):
                    idx = np.random.randint(0, 28)
                    target.append(torch.sum(img[i, 0, idx]))
                    ps[i][idx] = 1.
                target = torch.stack(target).unsqueeze(1)

                img = img.view(img.size(0), -1)

                with torch.no_grad():
                    encodding = model.encoder(img)

                output = parametric_function(encodding, ps)

                loss = criterion(output, target)
                train_loss_per_epoch.append(loss.item())

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(train_loss_per_epoch))

            with torch.no_grad():
                for data in test_loader:
                    img, label = data
                    ps = torch.zeros((img.shape[0], 28))
                    target = []
                    for i in range(img.shape[0]):
                        idx = np.random.randint(0, 28)
                        target.append(torch.sum(img[i, 0, idx]))
                        ps[i][idx] = 1.
                    target = torch.stack(target).unsqueeze(1)

                    img = img.view(img.size(0), -1)

                    encodding = model.encoder(img)
                    output = parametric_function(encodding, ps)

                    loss = criterion(output, target)
                    test_loss_per_epoch.append(loss.item())

                test_loss.append(np.mean(test_loss_per_epoch))
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()

            print('epoch [{}/{}], train loss:{:.4f}, test loss:{:.4f}, best test loss:{:.4f}'
                  .format(epoch + 1, num_epochs, np.mean(train_loss_per_epoch), np.mean(test_loss_per_epoch),
                         best_test_loss))

            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(parametric_function.state_dict(), f"./{save_name}.pth")

    except KeyboardInterrupt:
        pass
    

def train_simple_param_model(parametric_function, criterion, optimizer, 
                             train_loder, test_loader, model, save_name, num_epochs=100):
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            test_loss_per_epoch = []

            for data in train_loder:
                img, label = data
                ps = torch.zeros((img.shape[0], 1))
                target = []
                for i in range(img.shape[0]):
                    idx = np.random.randint(0, 28)
                    target.append(torch.sum(img[i, 0, idx]))
                    ps[i] = idx
                target = torch.stack(target).unsqueeze(1)

                img = img.view(img.size(0), -1)

                with torch.no_grad():
                    encodding = model.encoder(img)

                output = parametric_function(encodding, ps)

                loss = criterion(output, target)
                train_loss_per_epoch.append(loss.item())

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(train_loss_per_epoch))

            with torch.no_grad():
                for data in test_loader:
                    img, label = data
                    ps = torch.zeros((img.shape[0], 1))
                    target = []
                    for i in range(img.shape[0]):
                        idx = np.random.randint(0, 28)
                        target.append(torch.sum(img[i, 0, idx]))
                        ps[i] = idx
                    target = torch.stack(target).unsqueeze(1)

                    img = img.view(img.size(0), -1)

                    encodding = model.encoder(img)
                    output = parametric_function(encodding, ps)

                    loss = criterion(output, target)
                    test_loss_per_epoch.append(loss.item())

                test_loss.append(np.mean(test_loss_per_epoch))
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()

            print('epoch [{}/{}], train loss:{:.4f}, test loss:{:.4f}, best test loss:{:.4f}'
                  .format(epoch + 1, num_epochs, np.mean(train_loss_per_epoch), np.mean(test_loss_per_epoch),
                         best_test_loss))

            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(parametric_function.state_dict(), f"./{save_name}.pth")

    except KeyboardInterrupt:
        pass
    
    

class Params_func2(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.net = nn.Sequential(
            nn.Linear(64+2*20, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x, p):
        input_x = torch.cat([x, p], dim=1)
        output = self.net(input_x)
        return output
    

def train_param_model2(parametric_function, criterion, optimizer, train_loder, test_loader, model, save_name, num_epochs=100):
    num_epochs = 100
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            test_loss_per_epoch = []

            for data in train_loder:
                img, label = data
                p_x = torch.zeros((img.shape[0], 20))
                p_y = torch.zeros((img.shape[0], 20))
                target = []
                for i in range(img.shape[0]):
                    id_x = np.random.randint(4, 24)
                    id_y = np.random.randint(4, 24)
                    target.append(img[i, 0, id_x, id_y])
                    p_x[i][id_x-4] = 1.
                    p_y[i][id_y-4] = 1.

                ps = torch.cat([p_x, p_y], dim=1)
                target = torch.stack(target).unsqueeze(1)

                img = img.view(img.size(0), -1)

                with torch.no_grad():
                    encodding = model.encoder(img)

                output = parametric_function(encodding, ps)

                loss = criterion(output, target)
                train_loss_per_epoch.append(loss.item())

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(train_loss_per_epoch))

            with torch.no_grad():
                for data in test_loader:
                    img, label = data
                    p_x = torch.zeros((img.shape[0], 20))
                    p_y = torch.zeros((img.shape[0], 20))
                    target = []
                    for i in range(img.shape[0]):
                        id_x = np.random.randint(4, 24)
                        id_y = np.random.randint(4, 24)
                        target.append(img[i, 0, id_x, id_y])
                        p_x[i][id_x-4] = 1.
                        p_y[i][id_y-4] = 1.

                    ps = torch.cat([p_x, p_y], dim=1)
                    target = torch.stack(target).unsqueeze(1)

                    img = img.view(img.size(0), -1)

                    encodding = model.encoder(img)
                    output = parametric_function(encodding, ps)

                    loss = criterion(output, target)
                    test_loss_per_epoch.append(loss.item())

                test_loss.append(np.mean(test_loss_per_epoch))
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()

            print('epoch [{}/{}], train loss:{:.4f}, test loss:{:.4f}, best test loss:{:.4f}'
                  .format(epoch + 1, num_epochs, np.mean(train_loss_per_epoch), np.mean(test_loss_per_epoch),
                         best_test_loss))

            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(parametric_function.state_dict(), f"./{save_name}.pth")

    except KeyboardInterrupt:
        pass
    

class Params_func3(nn.Module):
    def __init__(self, img_dim, lat_size=64):
        super().__init__()
        self.img_dim = img_dim
        self.net = nn.Sequential(
            nn.Linear(lat_size+self.img_dim*self.img_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1))

    def forward(self, x, p):
        p = p[:x.size(0)]
        input_x = torch.cat([x, p], dim=1)
        output = self.net(input_x)
        return output
    
    
def train_param_model3(parametric_function, criterion, optimizer, train_loder, test_loader, 
                       other_dataloader, other_test_loader, model, save_name, num_epochs=100):
    num_epochs = 100
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    try:
        for epoch in range(num_epochs):
            train_loss_per_epoch = [] 
            test_loss_per_epoch = []

            for data in train_loder:
                img, label = data
                img = img.view(img.size(0), -1)

                for other_data in other_dataloader:
                    other_img, _ = other_data
                    other_img = other_img[:img.size(0)]
                    other_img = other_img.view(other_img.size(0), -1)
                    break

                target = []
                for i in range(img.shape[0]):
                    target.append(torch.tensor(pearsonr(img[i].numpy(), other_img[i].numpy())[0]))
                target = torch.stack(target).unsqueeze(1)

                with torch.no_grad():
                    encodding = model.encoder(img)

                output = parametric_function(encodding, other_img)

                loss = criterion(output, target)
                train_loss_per_epoch.append(loss.item())

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(train_loss_per_epoch))

            with torch.no_grad():
                for data in test_loader:
                    img, label = data
                    img = img.view(img.size(0), -1)

                    for other_data in other_test_loader:
                        other_img, _ = other_data
                        other_img = other_img.view(other_img.size(0), -1)
                        break

                    target = []
                    for i in range(img.shape[0]):
                        target.append(torch.tensor(pearsonr(img[i].numpy(), other_img[i].numpy())[0]))
                    target = torch.stack(target).unsqueeze(1)

                    encodding = model.encoder(img)
                    output = parametric_function(encodding, other_img)

                    loss = criterion(output, target)
                    test_loss_per_epoch.append(loss.item())

                test_loss.append(np.mean(test_loss_per_epoch))
            # ===================log========================
            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))

            plt.title("loss")
            plt.xlabel("number of epoch")
            plt.ylabel("loss")
            plt.plot(train_loss, 'b', label = "Train data")
            plt.plot(test_loss, 'r', label = "Test data")
            plt.legend()
            plt.show()

            print('epoch [{}/{}], train loss:{:.4f}, test loss:{:.4f}, best test loss:{:.4f}'
                  .format(epoch + 1, num_epochs, np.mean(train_loss_per_epoch), np.mean(test_loss_per_epoch),
                         best_test_loss))


            if test_loss[-1] < best_test_loss:
                best_test_loss = test_loss[-1]
                torch.save(parametric_function.state_dict(), f"./{save_name}.pth")

    except KeyboardInterrupt:
        pass
