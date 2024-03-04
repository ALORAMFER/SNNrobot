import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from nets.utils_.original_quant_dorefa import activation_quantize_fn, conv2d_Q_fn, linear_Q_fn, weight_quantize_fn
from cnn2snn import QuantizedReLU, QuantizedDense, QuantizedConv2D
from cnn2snn.quantization_ops import LinearWeightQuantizer, MaxPerAxisQuantizer

from keras.layers import Dense, ReLU, Conv2D, Flatten
from keras.models import Sequential
from cnn2snn import convert
from cnn2snn import quantize
from cnn2snn import check_model_compatibility

from Env.environment import DiscreteMountainCar

nn_dense_param = 256


class DoReFa_Q(nn.Module):
    def __init__(self, wbit, abit, num_classes=21):
        env = DiscreteMountainCar()
        super(DoReFa_Q, self).__init__()
        Linear = linear_Q_fn(w_bit=wbit)
        self.relu = activation_quantize_fn(a_bit=abit)
        self.classifier = nn.Sequential(
            Linear(env.observation_space, nn_dense_param),
            activation_quantize_fn(a_bit=abit),
            Linear(nn_dense_param, 256),
            activation_quantize_fn(a_bit=abit),
            Linear(256, num_classes),
        )
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.classifier(x)
        return x


class float_Q(nn.Module):
    def __init__(self, num_classes=21):
        env = DiscreteMountainCar()
        super(float_Q, self).__init__()
        self.fc1 = nn.Linear(env.observation_space, 256)  # 64枚の22×16の画像を、256次元のoutputへ
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
        # act=F.relu()

    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.float16)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

    def half(self):
        for module in self.children():
            module.half()
        return self


class keras_NN(nn.Module):
    def __init__(self):
        return

    def get_akida_model(self, wbit=4, abit=4, num_classes=32):
        env = DiscreteMountainCar()
        quantize = {'class_name': 'MaxPerAxisQuantizer',
                    'config': {'bitwidth': wbit,
                               'dtype': 'float32',
                               'name': 'max_per_axis_quantizer',
                               'trainable': True}}

        model = Sequential([
            QuantizedDense(nn_dense_param, quantizer=quantize, input_shape=(env.observation_space,)),
            QuantizedReLU(bitwidth=abit, max_value=1),
            QuantizedDense(256, quantizer=quantize),
            QuantizedReLU(bitwidth=abit, max_value=1),
            QuantizedDense(num_classes, quantizer=quantize)
        ])

        return model


class DoReFa_STDP(nn.Module):
    def __init__(self, wbit, abit, num_classes=21):
        env = DiscreteMountainCar()
        super(DoReFa_STDP, self).__init__()
        Linear = linear_Q_fn(w_bit=wbit)
        self.relu = activation_quantize_fn(a_bit=abit)
        self.classifier = nn.Sequential(
            # Linear(env.observation_space, 5),
            # activation_quantize_fn(a_bit=abit),
            Linear(env.observation_space, num_classes),
        )
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.classifier(x)
        return x


class keras_NN_STDP(nn.Module):
    def __init__(self):
        return

    def get_akida_model(self, wbit=8, abit=4, num_classes=32):
        env = DiscreteMountainCar()
        quantize = {'class_name': 'MaxPerAxisQuantizer',
                    'config': {'bitwidth': wbit,
                               'dtype': 'float32',
                               'name': 'max_per_axis_quantizer',
                               'trainable': True}}

        model = Sequential([
            # QuantizedDense(5, quantizer=quantize, input_shape = (env.observation_space,)),
            # QuantizedReLU(bitwidth=abit, max_value=1),
            QuantizedDense(num_classes, quantizer=quantize, input_shape=(env.observation_space,))
        ])
        return model
