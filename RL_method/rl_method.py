
from icecream import ic
from Env.environment import DiscreteMountainCar
from Network.network import DoReFa_Q, float_Q, keras_NN, DoReFa_STDP, keras_NN_STDP
from Utility.utility import Utility
from nets.utils_.original_quant_dorefa import activation_quantize_fn, conv2d_Q_fn, linear_Q_fn, weight_quantize_fn

import torch.nn as nn
import numpy as np
import torch
import copy
import math
import time
from cnn2snn import quantize
import torch.optim as optim
from torch.autograd import Variable

class RL_method():
    def __init__(self, wbit, abit):
        self.env = DiscreteMountainCar()
        self.action_length = self.env.action_space
        self.obs_dim = self.env.observation_space

        self.EPOCH_NUM = 3000  # エポック数 1000
        self.STEP_MAX = 200  # 最高ステップ数 100
        self.MEMORY_SIZE = 5 * self.STEP_MAX  # メモリサイズいくつで学習を開始するか
        self.BATCH_SIZE = 32  # バッチサイズ
        self.TRAIN_FREQ = 1  # Q関数の学習間隔
        self.UPDATE_TARGET_Q_FREQ = 1 * self.STEP_MAX  # Q関数の更新間隔
        self.GAMMA = 0.97  # 割引率
        self.lr = 0.000025  # 学習率 original: 0.00015 QPD: 0.00005
        self.LOG_FREQ = 10  # ログ出力の間隔
        self.wbit = wbit
        self.abit = abit
        self.alpha = 0.99      
        self.beta = 0.6 # original: 1.0

        return

    def main(self, ModelSelect, model_name):
        if ModelSelect == "QPN":
            ic("QPN")
            self.QPN(model_name)
        elif ModelSelect == "DQN":
            ic("DQN")
            self.DQN(model_name)
        elif ModelSelect == "CVI":
            ic("CVI")
            self.CVI(model_name)
        elif ModelSelect == "QQN":
            ic("QQN")
            self.QQN(model_name)
        elif ModelSelect == "STDP":
            ic("STDP")
            self.STDP(model_name)
        elif ModelSelect == "random":
            ic("start random")
            self.random(model_name)
        elif ModelSelect == "QQN_Distillation":
            ic("start QQN Distillation")
            self.QQN_Distillation(model_name)
        elif ModelSelect == "QPN_Distillation":
            ic("start QPN Distillation")
            self.QPN_Distillation(model_name)
        else:
            print("Model does not exist.")
            exit()
        return

    def QPN(self, model_name):
        """ update quantized network by CVI """
        self.model = DoReFa_Q(wbit=self.wbit, abit=self.abit, num_classes=self.action_length).cuda()
        model_keras_class = keras_NN()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        update_type = "CVI"
        network_type = "DoReFa"
        action_type = "softmax_selection"
        self.RL_main(model_name, update_type, network_type, action_type)
        return

    def QQN(self, model_name):
        """ update network by DQN """
        self.model = DoReFa_Q(wbit=self.wbit, abit=self.abit, num_classes=self.action_length).cuda()
        model_keras_class = keras_NN()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        self.lr = 0.00005
        update_type = "DQN"
        network_type = "DoReFa"
        action_type = "softmax_selection"
        self.RL_main(model_name, update_type, network_type, action_type)
        return

    def CVI(self, model_name):
        """ update float network by CVI"""
        self.model = float_Q(num_classes=self.action_length).cuda()
        # self.model = self.model.half()
        model_keras_class = keras_NN()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        update_type = "CVI"
        network_type = "float"
        action_type = "softmax_selection"
        self.RL_main(model_name, update_type, network_type, action_type)
        return

    def DQN(self, model_name):
        """ update network by DQN """
        self.model = float_Q(num_classes=self.action_length).cuda()
        # self.model = self.model.half()
        model_keras_class = keras_NN()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        update_type = "DQN"
        network_type = "float"
        action_type = "softmax_selection"
        self.RL_main(model_name, update_type, network_type, action_type)
        return

    def QQN_Distillation(self, model_name):
        """ update network by QQN Distillation """
        self.model = float_Q(num_classes=self.action_length).cuda()
        model_keras_class = keras_NN()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        self.model2 = DoReFa_Q(wbit=self.wbit, abit=self.abit, num_classes=self.action_length).cuda()
        model_keras_class2 = keras_NN()        
        self.model_keras2 = model_keras_class2.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        update_type = "DQN"
        network_type = "float"
        action_type = "softmax_selection"
        self.RL_main2(model_name, update_type, network_type, action_type)
        return

    def QPN_Distillation(self, model_name):
        """ update network by QPN Distillation """
        self.model = float_Q(num_classes=self.action_length).cuda()
        model_keras_class = keras_NN()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        self.model2 = DoReFa_Q(wbit=self.wbit, abit=self.abit, num_classes=self.action_length).cuda()
        model_keras_class2 = keras_NN()        
        self.model_keras2 = model_keras_class2.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        update_type = "CVI"
        network_type = "float"
        action_type = "softmax_selection"
        self.RL_main2(model_name, update_type, network_type, action_type)
        return


    def random(self, model_name):
        """ random """
        self.model = DoReFa_STDP(wbit=8, abit=4, num_classes=self.action_length).cuda()
        model_keras_class = keras_NN_STDP()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        update_type = "random"
        network_type = "random"
        action_type = "random"
        self.RL_main(model_name, update_type, network_type, action_type)

    # ----------------------------------------------------------------------------------------------
    def RL_main(self, model_name, update_type, network_type, action_type):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ic(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.95, eps=0.01)
        self.criterion = nn.CrossEntropyLoss().cuda
        Q_ast = copy.deepcopy(self.model)
        self.utility = Utility(self.beta, self.wbit, self.abit, self.LOG_FREQ, self.STEP_MAX, self.model, self.model_keras)

        total_step = 0  # 総ステップ（行動）数
        memory = []  # メモリ
        total_rewards = []  # 累積報酬記録用リスト

        # 学習開始
        print("Train")
        print("\t".join(["epoch", "reward", "total_step", "elapsed_time"]))
        start = time.time()
        return


    def update_network_DQN(self, epoch, memory, Q_ast):
        # RLによって学習モデルを更新する関数

        # 経験リプレイ
        if epoch % self.TRAIN_FREQ == 0:

            memory_ = np.random.permutation(memory)
            #memory_idx = range(len(memory_))

            for i in range(1):  # memory_idx[::self.BATCH_SIZE]:
                batch = np.array(memory_[i:i + self.BATCH_SIZE])  # 経験ミニバッチ
                pobss = np.array(batch[:, 0].tolist(), dtype="float32")
                pacts = np.array(batch[:, 1].tolist(), dtype="int32")
                rewards = np.array(batch[:, 2].tolist(), dtype="float32")
                obss = np.array(batch[:, 3].tolist(), dtype="float32")
                dones = np.array(batch[:, 4].tolist(), dtype="bool")
                # set y
                pobss_ = Variable(torch.from_numpy(pobss)).to(self.device)
                q = self.model.forward(pobss_)
                obss_ = Variable(torch.from_numpy(obss)).to(self.device)
                maxs, indices = torch.max(Q_ast(obss_).data, 1)
                maxq = maxs.cpu().numpy()  # maxQ
                q_target = Q_ast.forward(pobss_)
                target = copy.deepcopy(q_target.cpu().data.numpy())
                #target = copy.deepcopy(q.cpu().data.numpy())
                for j in range(self.BATCH_SIZE):
                    target[j][pacts[j]] = rewards[j] + self.GAMMA * maxq[j] * (not dones[j])  # 教師信号
                # Perform a gradient descent step
                self.optimizer.zero_grad()
                loss = nn.MSELoss()(q, Variable(torch.from_numpy(target).to(self.device)))
                loss.backward()
                self.utility.save_loss(loss, epoch)
                self.optimizer.step()
        return 

    def softmax_selection(self, values, tau):
        """
            softmax 行動選択
            @input : values : numpy.array型
            @return: action : int
        """
        # values = values[0]
        max_value = np.max(values)
        sum_exp_values = sum([np.exp((v - max_value) * tau) for v in values])   # softmax選択の分母の計算
        p = [np.exp((v - max_value) * tau) / sum_exp_values for v in values]  # 確率分布の生成
        max_p = max(p)
        max_action=p.index(max_p)
        action = np.random.choice(np.arange(len(values)), p=p)                # 確率分布pに従ってランダムで選択
        return action, max_p, max_action

    def mellowmax_operator(self, Q_values, alpha=0.99, beta=1.0):
        """
            mellowmax operatorのための関数
            (m_\beta Q) \coloneqq 1/\beta log((1/|A|)*(\sum exp(\beta Q (s,a))))
            だけどlog sum expを回避するために
            \log \left( \sum_{i=1}^n \exp(x_i) \right) = x_{\max} + \log \left( \sum_{i=1}^n \exp(x_i-x_{\max}) \right)
            を導入

            @input : Q_values  : numpy.array型[][] : [batch_size][action_size]のQ関数の値
            @input : alpha     : final double      : 係数
            @input : beta      : final double      : 係数
            @return: mellowmax : double[]          : batch分のmellowmaxの値
        """
        mellowmax = []
        for i, Q_value in enumerate(Q_values):
            max_Q_value = max(Q_value)
            sum = 0
            for j in range(self.action_length):
                sum += math.exp(beta * (Q_value[j] - max_Q_value))
            mellowmax.append(max_Q_value + ((1 / beta) * math.log((1 / abs(self.action_length)) * sum)))
        return mellowmax

    def update_beta(self,beta):
        new_beta = beta + self.beta_growth_rate
        if new_beta > self.final_beta:
            return self.final_beta
        else:

            return new_beta

    def psi_value(self, reward, psi_value, mellowmax_s0, mellowmax_s1, done, alpha=0.99):
        """
            psi learning
            @input : reward       : double  : 報酬の値
            @input : psi_value    : double  : psiの値
            @input : mellowmax_s0 : double  : mellowmaxの値
            @input : mellowmax_s1 : double  : 行動後のmellowmaxの値
            @input : done         : boolean : 環境による終了フラグ
            @input : alpha        : double  : 係数
            @return:              : double  :
        """
        return alpha * (psi_value - mellowmax_s0) + reward + (self.GAMMA * mellowmax_s1) * (not done)
