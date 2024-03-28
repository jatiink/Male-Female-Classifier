import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    
    def __init__(self, txt_path, transform = None):
        
        txt = open(txt_path, 'r')
        
        data, names, paths, labels = [], [], [], []
        
        for line in txt.readlines():
            txt = line.replace('\n', '')
            txt = txt.replace(' ', '')
            txt = txt.split(",")
            data.append(txt)
            
        for img_names, img_paths, img_labels in data:
            names.append(img_names)
            paths.append(img_paths)
            labels.append(img_labels)
        
        self.paths = paths
        self.labels = labels
        self.img_names = names
        self.transform = transform
                
    def __getitem__(self, indx):
        
        image = cv2.imread(self.paths[indx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.array(self.labels[indx], dtype=np.uint8)
        name = self.img_names[indx]
        
        if self.transform:
            image = self.transform(image)
        
        inputs = {"image" : image, 
                  "labels" : label, 
                  "name": name}
        
        return inputs
    
    def __len__(self):
        return len(self.labels)

train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.Resize((500, 500)),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((500, 500)),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])])

board = SummaryWriter("board")

train_txt = "train_data.txt"
test_txt = "test_data.txt"

train_data = MyDataset(train_txt, train_transform)
test_data = MyDataset(test_txt, test_transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10 , shuffle=False)

classes = ["female", "male"]

def tensor_to_image(tensor):
    """
    Convert torch tensor into numpy array
    """
    image = tensor.detach().cpu().numpy()
    if len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))
    else:
        image = np.transpose(image, (0, 2, 3, 1))
    if image.max() <= 1 and image.min() >= 0:
        image = image * 255
    elif image.max() <= 1 and image.min() >= -1 and image.min() < 0:
        image = (image + 1) / 2
        image = image * 255
    image = image.astype("uint8").copy()
    return image


def get_visual(img_batch, predictions, labels):
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    img_list = []
    for i in range(img_batch.shape[0]):
        img = tensor_to_image(img_batch[i])
        img = cv2.putText(img, ("Prediction: " + classes[predictions[i]]), (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 0), 2)
        img = cv2.putText(img, ("Target: " + classes[labels[i]]), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 0), 2)
        img_list.append(img)
    return img_list


def board_add_images(board, tag_name, img_list, step_count, names=None):
    if names is None:
        for i, img in enumerate(img_list):
            board.add_image(f'{tag_name}/{i}', img, step_count, dataformats='HWC')
            board.flush()
    else:
        for name, img in zip(names, img_list):
            board.add_image(f'{tag_name}/{name}', img, step_count, dataformats='HWC')
            board.flush()

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 1)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 5, 2, 1)
        self.conv5 = nn.Conv2d(256, 256, 5, 1, 1)
        self.conv6 = nn.Conv2d(256, 512, 5, 2, 1)
        self.conv7 = nn.Conv2d(512, 512, 5, 1, 1)
        self.fc1 = nn.Linear(11*11*512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = F.relu(self.conv5(X))
        X = F.relu(self.conv6(X))
        X = F.relu(self.conv7(X))
        X = X.view(-1, 11*11*512)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.log_softmax(self.fc3(X), dim=1)
        return X

model = Model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def training():
    
    epochs = 20
    i = 0
    b = 0
    
    for epoch in range(epochs):

        training_corr = 0
        
        for step, data in enumerate(train_loader, 1):
            inputs, labels, names = data["image"], data["labels"], data["name"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = torch.max(outputs.data, 1)[1]
            training_corr += (predictions == labels).sum()
            i += 1
            print(f'Epoch: {epoch} batch: {step} loss: {loss.item()} Accuracy: {training_corr*100/(10*step)}')
            if step % 50 == 0:
                img_list = get_visual(inputs, predictions, labels)
                board_add_images(board, "Train", img_list, i, names = names)
                board.add_scalars("Train_losses", {"net_loss": loss.item()}, i)

        model.train(False)
        for vstep, vdata in enumerate(test_loader, 1):
            vinputs, vlabels, vnames = vdata["image"], vdata["labels"], vdata["name"]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            vpredictions = torch.max(voutputs.data, 1)[1]
            b += 1
            if vstep % 50 == 0:
                img_list = get_visual(vinputs, vpredictions, vlabels)
                board_add_images(board, "Test", img_list, b, names=vnames)
                board.add_scalars("Losses/test", {"Net_loss": vloss.item()}, b)
        model.train(True)
        
    torch.save(model.state_dict(), 'male_female_classifier.pt')

if __name__ == '__main__':
    training()