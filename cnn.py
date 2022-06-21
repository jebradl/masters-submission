import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
from sklearn.model_selection import KFold
from torchvision import transforms, datasets, utils
from copy import deepcopy


dataset_path = 'C:/Users/jeb1618/masters/models/data/fma/classified_medium' # path of images classified by genre

batch_size = 64  # number of images used per iteration
img_height = 235 # input image dimensions
img_width = 352

epochs =  50 # number of iterations the model is trained over
test_split = 0.8 # ratio of training to test data

# setting up normalisation of images
transform = transforms.Compose([
            transforms.Resize([img_height, img_width]),
            # transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=1), # grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]) # rgb
            # transforms.Normalize(mean=(0.5), std=(0.5))]) # grayscale

# using the above normalisation to load dataset images
data = datasets.ImageFolder(root=dataset_path, transform=transform)

# splitting the data into a train and test set
test_split_size = int(len(data)*test_split)
testset, testset = torch.utils.data.random_split(data, (test_split_size, len(data)-test_split_size))

train_ds = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
test_ds = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

# alternative k folds validation technique
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = data[train_index], data[test_index]

train_ds = torch.utils.data.DataLoader(dataset=[X_train, y_train], batch_size=batch_size, shuffle=True)
test_ds = torch.utils.data.DataLoader(dataset=[X_test, y_test], batch_size=batch_size, shuffle=True)


# classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock'] # classes for small dataset
classes = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz', 'Old-Time', 'Pop', 'Rock', 'Soul-RnB', 'Spoken'] # classes for medium dataset

def imshow(img):  # function to show data
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# dataiter = iter(train_ds) # get some images from the training set
# images, labels = dataiter.next() # iterates through data

# imshow(utils.make_grid(images)) # show images
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size))) # print labels



class Net(nn.Module):
    def __init__(self): # initialising methods
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=2) # convolutional layers
        self.conv2 = nn.Conv2d(32, 64, 3, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=2)
        self.pool = nn.MaxPool2d(2, 2) # max pooling before classification
        self.fc1 = nn.Linear(128 * 31 * 45, 128) # linear layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16) # finishing with 8 genres in small dataset, 16 in medium

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # actuating convolutional methods using relu activation function
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x)) # actuating linear layers using relu activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class MusicNet(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.convtemp = nn.Conv2d(3, 32, (1, 80), padding=0) # vertical kernel for time features
        self.convtimb = nn.Conv2d(3, 32, (32, 1), padding=0) # horizontal kernel for pitch features
        self.maxpooltemp = nn.MaxPool2d((235, 1)) # max pooling to image height x 1
        self.maxpooltimb = nn.MaxPool2d((1, 352)) # max pooling to image width x 1
        self.fc1 = nn.Linear(32*525, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 8) # finishing with 8 genres in small dataset, 16 in medium

        self.dropout = nn.Dropout2d(0.5) # drops proportion of neurons to prevent overfitting

    def forward(self, x):

        x_temp = deepcopy(x) # construct new compound object of x, then recursively copies in original x
        x_timb = deepcopy(x)

        x_temp = self.convtemp(x_temp) # actuate temporal and timbral convolution
        x_timb = self.convtimb(x_timb)

        x_temp = self.maxpooltemp(x_temp) # max pooling over temporal and timbral 
        x_timb = self.maxpooltimb(x_timb)

        x_temp = x_temp.squeeze(2)
        x_timb = x_timb.squeeze(3)

        x = torch.cat((x_temp, x_timb), dim=2)
        x = x.view(-1, 32*525)
        x = F.relu(self.fc1(x)) 
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) 
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# net = Net()
net = MusicNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(epochs):  # loop over the dataset multiple times

    print("epoch", epoch+1)

    running_loss = 0.0
    for i, data in enumerate(train_ds, 0):
        # separate labels and input images in data variable
        inputs, labels = data

        # zero the parameter gradients before gradient descent
        optimizer.zero_grad()

        # forwards train the network
        outputs = net(inputs)

        # actuate the loss function, then backpropogate weights and biases
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # sum loss
        running_loss += loss.item()

    print(f'loss: {running_loss / len(train_ds):.3f}')
            

print('finished training')

# sample some testing images
dataiter = iter(test_ds)
images, labels = dataiter.next()

# print images
imshow(utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# use the newly trained model to make predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


# testing loop

# initialise variables to track the accuracy of the model
correct = 0
total = 0

with torch.no_grad(): # disable gradient calculation because we're not training or updating weights
    for data in test_ds:
        images, labels = data
        # calculate outputs
        outputs = net(images)
        # this will be one hot encoded, with a probability of each class, so get the most probably class
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        # check whether the predictions are correct
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')