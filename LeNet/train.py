import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    # transfrom对图像预处理，normalize标准化
    # 训练集
    train_set = torchvision.datasets.MNIST(root='./mnist', train=True, download=False,transform=transform)
    # 第一次运行时要download=Ture，下载完数据再换成False
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)
    # train_loader导入train_set下载的数据集，一次处理36张图片

    # 测试集
    val_set = torchvision.datasets.MNIST(root='./mnist', train=False, download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000, shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)  # 迭代器
    val_image, val_label = val_data_iter.next()  # 获取到图片及其对应的标签

    net = LeNet()  # 实例化模型
    loss_function = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器，第一个参数为net的参数，第二个参数学习率

    for epoch in range(5):  # 训练集迭代5次
        running_loss = 0.0  # 累加训练中的损失
        for step, data in enumerate(train_loader, start=0):  # enumerate 返回训练集中的步数，和数据索引
            inputs, labels = data  # 输入的图像、标签
            optimizer.zero_grad()  # 清除历史梯度，可以实现大的batch
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()  # 优化器参数更新

            running_loss += loss.item()  # 累加到
            if step % 500 == 499:  # 每500步打印
                with torch.no_grad():  # 测试的时候冻结参数
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]  # 查找最大的输出索引，1只需索引，不需要具体值，返回预测类别
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    # 真值和测试值比较求和别上测试样本数目
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))  # 500因为每500打印一次
                    running_loss = 0.0  # 清零准备下次迭代

    print('Finished Training')
    save_path = './JLenet.pth'
    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()