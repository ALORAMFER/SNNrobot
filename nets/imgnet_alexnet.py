import torch
import torch.nn as nn
import torch.nn.init as init

from icecream import ic


#from utils_.quant_dorefa import activation_quantize_fn, conv2d_Q_fn, linear_Q_fn

from nets.utils_.quant_dorefa import activation_quantize_fn, conv2d_Q_fn, linear_Q_fn

#from utils.quant_dorefa import activation_quantize_fn
#from utils.quant_dorefa import conv2d_Q_fn, linear_Q_fn



class AlexNet_Q(nn.Module):
  def __init__(self, wbit, abit, num_classes=10):
    super(AlexNet_Q, self).__init__()
    Conv2d = conv2d_Q_fn(w_bit=wbit)
    Linear = linear_Q_fn(w_bit=wbit)

    self.features = nn.Sequential(
      Conv2d(3, 16, kernel_size=7, stride=2),
      nn.ReLU(inplace=True),
      activation_quantize_fn(a_bit=abit),

      Conv2d(16, 32, kernel_size=5, stride=2),
      nn.ReLU(inplace=True),
      activation_quantize_fn(a_bit=abit),

      Conv2d(32, 64, kernel_size=5, stride=1),
      nn.ReLU(inplace=True),
      activation_quantize_fn(a_bit=abit),

      Conv2d(64, 64, kernel_size=3),
      nn.ReLU(inplace=True),
      activation_quantize_fn(a_bit=abit),


    )
    self.classifier = nn.Sequential(
      Linear(64 * 12* 12, 512),
      nn.ReLU(inplace=True),
      activation_quantize_fn(a_bit=abit),

      Linear(512, 512),
      nn.ReLU(inplace=True),
      activation_quantize_fn(a_bit=abit),

      nn.Linear(512, num_classes),
    )

    for m in self.modules():
      if isinstance(m, Conv2d) or isinstance(m, Linear):
        init.xavier_normal_(m.weight.data)

    # self.features = nn.Sequential(
    #   nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
    #   nn.BatchNorm2d(96),
    #   nn.MaxPool2d(kernel_size=3, stride=2),
    #   nn.ReLU(inplace=True),

    #   Conv2d(96, 256, kernel_size=5, padding=2),
    #   nn.BatchNorm2d(256),
    #   nn.MaxPool2d(kernel_size=3, stride=2),
    #   nn.ReLU(inplace=True),
    #   activation_quantize_fn(a_bit=abit),

    #   Conv2d(256, 384, kernel_size=3, padding=1),
    #   nn.ReLU(inplace=True),
    #   activation_quantize_fn(a_bit=abit),

    #   Conv2d(384, 384, kernel_size=3, padding=1),
    #   nn.ReLU(inplace=True),
    #   activation_quantize_fn(a_bit=abit),

    #   Conv2d(384, 256, kernel_size=3, padding=1),
    #   nn.MaxPool2d(kernel_size=3, stride=2),
    #   nn.ReLU(inplace=True),
    #   activation_quantize_fn(a_bit=abit),
    # )
    # self.classifier = nn.Sequential(
    #   Linear(256 * 6 * 6, 4096),
    #   nn.ReLU(inplace=True),
    #   activation_quantize_fn(a_bit=abit),

    #   Linear(4096, 4096),
    #   nn.ReLU(inplace=True),
    #   activation_quantize_fn(a_bit=abit),
    #   nn.Linear(4096, num_classes),
    # )

    # for m in self.modules():
    #   if isinstance(m, Conv2d) or isinstance(m, Linear):
    #     init.xavier_normal_(m.weight.data)

  def forward(self, x):
    x = self.features(x)
    #ic(x.shape)
    x = x.view(x.size(0), 64 *12 *12)
    x = self.classifier(x)
    return x


if __name__ == '__main__':
  from torch.autograd import Variable

  features = []

  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = AlexNet_Q(wbit=1, abit=2)
  net.train()

  for w in net.named_parameters():
    print(w[0])

  for m in net.modules():
    m.register_forward_hook(hook)

  y = net(Variable(torch.randn(1, 3, 224, 224)))
  print(y.size())
