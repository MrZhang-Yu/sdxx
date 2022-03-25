import os
import sys
import json
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import GoogLeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    transform = transforms.Compose(
        [transforms.Resize((96, 96), 2),  # 2插值
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024,
                                               shuffle=True, num_workers=0)
    train_steps = len(train_loader)
    print(train_steps)

    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=0)
    val_num = len(val_set)

    net = GoogLeNet(num_classes=10, aux_logits=False, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = 5
    best_acc = 0.0
    save_path = './googleNet.pth'
    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            #         logits, aux_logits2, aux_logits1 = net(images.to(device))
            #         loss0 = loss_function(logits, labels.to(device))
            #         loss1 = loss_function(aux_logits1, labels.to(device))
            #         loss2 = loss_function(aux_logits2, labels.to(device))
            #         loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            # 验证
            net.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))  # eval model only have last output layer
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (
            epoch + 1, running_loss / train_steps, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
    print('Finished Training')

if __name__ == '__main__':
    main()