import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet
import matplotlib.pyplot as plt
def main():
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
    net = LeNet()
    net.load_state_dict(torch.load('JLenet.pth'))
    im = Image.open('./h2.png')
    im = im.convert('L')
    plt.imshow(im)
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
        print(predict)
    #    predict = torch.softmax(outputs,dim=1)
    print(classes[int(predict)])
    #     print(predict)