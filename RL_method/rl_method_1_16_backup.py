
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

        self.EPSILON = 1.0  # ε-greedy法
        self.EPSILON_DECREASE = 0.0001  # εの減少値
        self.EPSILON_MIN = 0.1  # εの下限
        self.START_REDUCE_EPSILON = 100  # εを減少させるステップ数

        self.EPOCH_NUM = 1000  # エポック数 1000
        self.STEP_MAX = 200  # 最高ステップ数 100
        self.MEMORY_SIZE = 5 * self.STEP_MAX  # メモリサイズいくつで学習を開始するか
        self.BATCH_SIZE = 32  # バッチサイズ
        self.TRAIN_FREQ = 1  # Q関数の学習間隔
        self.UPDATE_TARGET_Q_FREQ = 1 * self.STEP_MAX  # Q関数の更新間隔
        self.GAMMA = 0.97  # 割引率
        self.lr = 0.000025  # 学習率 original: 0.00015 QPD: 0.00005
        self.wd = 5e-4  # weight_decay
        self.LOG_FREQ = 10  # ログ出力の間隔
        self.wbit = wbit
        self.abit = abit
        self.reward_factor=1

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
        elif ModelSelect == "QQN-epsilon":
            ic("QQN-epsilon")
            self.QQN_epsilon(model_name)
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

    def QQN_epsilon(self, model_name):
        """ update network by DQN """
        self.model = DoReFa_Q(wbit=self.wbit, abit=self.abit, num_classes=self.action_length).cuda()
        model_keras_class = keras_NN()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)
        update_type = "DQN"
        network_type = "DoReFa"
        action_type = "epsilon_greedy"
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

    def STDP(self, model_name):
        """ update network by STDP """
        self.model = DoReFa_STDP(wbit=8, abit=4, num_classes=self.action_length).cuda()
        model_keras_class = keras_NN_STDP()
        self.model_keras = model_keras_class.get_akida_model(wbit=self.wbit, abit=self.abit, num_classes=self.action_length)

        self.wbit = 8
        self.abit = 4
        update_type = "STDP"
        network_type = "DoReFa"
        action_type = "epsilon_greedy"
        self.RL_main(model_name, update_type, network_type, action_type)

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
        self.utility = Utility(self.beta, self.wbit, self.abit, self.LOG_FREQ, self.EPSILON, self.STEP_MAX, self.model, self.model_keras,self.reward_factor)

        total_step = 0  # 総ステップ（行動）数
        memory = []  # メモリ
        total_rewards = []  # 累積報酬記録用リスト

        # 学習開始
        print("Train")
        print("\t".join(["epoch", "EPSILON", "reward", "total_step", "elapsed_time"]))
        start = time.time()

        # 保存用ディレクトリ作成
        log_time = time.strftime("%d_0%b_%Y_%H_%M", time.gmtime())
        model_PATH = self.utility.make_directory(str(log_time), model_name)
        self.utility.make_tensorboard(str(log_time), model_name)

        total_reward_torch = []
        total_reward_akida = []
        total_reward_snn = []
        total_reward_random = []
        total_mse_actions = []
        total_index_miss_actions = []
        for epoch in range(self.EPOCH_NUM):
            pobs = self.env.reset()  # 環境初期化
            step = 0  # ステップ数
            done = False  # ゲーム終了フラグ
            total_reward = 0  # 累積報酬
            total_maxq = 0
            loss = 0
            total_mse_action = 0
            total_index_miss_action = 0

            if network_type == "DoReFa":
                model_keras = self.utility.torch_to_keras(self.model, self.model_keras)
                self.model_akida = quantize(model_keras,
                                            input_weight_quantization=self.wbit,
                                            weight_quantization=self.wbit,
                                            activ_quantization=self.abit)
                self.model_snn = self.utility.convert_from_ANN_to_SNN(self.model_akida)

            while not done and step < self.STEP_MAX:
                # 行動選択
                action_index = np.random.randint(0, self.action_length - 1)
                pact = self.env.action_from_discrete_to_consecutive(action_index)
                pact = np.int64(pact[0])

                if action_type == "softmax_selection":
                    # softmax行動選択
                    if network_type == "DoReFa":
                        pobs_ = np.array(pobs, dtype="uint8").reshape((1, self.obs_dim))
                        pact_ = self.model_snn.predict(pobs_)
                        self.utility.save_qSNN(pact_, total_step)
                        pact, max_p_SNN, max_action_SNN = self.softmax_selection(pact_[0][0][0], tau=self.beta)
                        self.utility.save_max_p_SNN(max_p_SNN, total_step)
                        self.utility.save_max_action_SNN(max_action_SNN, total_step)

                    elif network_type == "float":
                        pobs_ = np.array(pobs, dtype="float32").reshape((1, self.obs_dim))
                        pobs_ = Variable(torch.from_numpy(pobs_)).to(self.device)
                        pact = self.model(pobs_)
                        pact = pact.cpu().detach().numpy()
                        self.utility.save_qFPNN(pact, total_step)
                        q_target_tmp = Q_ast.forward(pobs_)
                        target_tmp = copy.deepcopy(q_target_tmp.cpu().data.numpy())
                        self.utility.save_targetFPNN(target_tmp, total_step)
                        pact, max_p_FPNN, max_action_FPNN = self.softmax_selection(pact[0], tau=self.beta)
                        self.utility.save_max_p_FPNN(max_p_FPNN, total_step)
                        self.utility.save_max_action_FPNN(max_action_FPNN, total_step)

                    else:
                        ic("行動選択エラー")
                        exit()
                elif action_type == "random":
                    pass
                else:
                    ic("行動選択エラー")
                    exit()
                # 行動
                obs, reward, done, _ = self.env.step(pact)
                reward=reward*self.reward_factor
                # self.env._render()
                # メモリに蓄積
                memory.append((pobs, pact, reward, obs, done))  # 状態、行動、報酬、行動後の状態、ゲーム終了フラグ
                if len(memory) > self.MEMORY_SIZE:  # メモリサイズを超えていれば消していく
                    memory.pop(0)

                # 次の行動へ
                total_reward += reward
                step += 1
                total_step += 1
                pobs = obs

            for _ in range(self.STEP_MAX):
                # 学習
                if len(memory) == self.MEMORY_SIZE:  # メモリサイズ分溜まっていれば学習
                    if update_type == "CVI":
                        total_maxq = self.update_network_CVI(epoch, total_maxq, memory, Q_ast)
                    elif update_type == "DQN":
                        total_maxq = self.update_network_DQN(epoch, total_maxq, memory, Q_ast)
                    elif update_type == "random":
                        pass
                    else:
                        ic("更新されていません")
                        exit()
                    # Q関数の更新
                    # if total_step % self.UPDATE_TARGET_Q_FREQ == 0:
                    if epoch % 15 == 0:
                        Q_ast = copy.deepcopy(self.model)

                # εの減少
                if self.EPSILON > self.EPSILON_MIN and total_step > self.START_REDUCE_EPSILON:
                    self.EPSILON -= self.EPSILON_DECREASE

            total_rewards.append(total_reward)  # 累積報酬を記録
            total_reward_torch, total_reward_akida, total_reward_snn, total_reward_random,\
                total_mse_actions, total_index_miss_actions = self.utility.save_log(network_type,
                                                                                    action_type,
                                                                                    total_mse_action,
                                                                                    total_index_miss_action,
                                                                                    epoch, start,
                                                                                    total_rewards,
                                                                                    total_step, model_PATH,
                                                                                    total_reward,
                                                                                    total_reward_torch,
                                                                                    total_reward_akida,
                                                                                    total_reward_snn,
                                                                                    total_reward_random,
                                                                                    total_mse_actions,
                                                                                    total_index_miss_actions,
                                                                                    self.beta)

        self.utility.save_neural_network(model_PATH, epoch)
        if network_type == "DoReFa":
            self.utility.save_snn_neural_network(model_PATH, epoch)
        return

    # ----------------------------------------------------------------------------------------------
    def RL_main2(self, model_name, update_type, network_type, action_type):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ic(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.95, eps=0.01)
        self.optimizer2 = optim.RMSprop(self.model2.parameters(), lr=self.lr, alpha=0.95, eps=0.01)
        self.criterion = nn.CrossEntropyLoss().cuda
        Q_ast = copy.deepcopy(self.model)
        Q_ast2 = copy.deepcopy(self.model2)
        self.utility = Utility(self.beta, self.wbit, self.abit, self.LOG_FREQ, self.EPSILON, self.STEP_MAX, self.model, self.model_keras,self.reward_factor)
        self.utility2 = Utility(self.beta, self.wbit, self.abit, self.LOG_FREQ, self.EPSILON, self.STEP_MAX, self.model2, self.model_keras2,self.reward_factor)

        total_step = 0  # 総ステップ（行動）数
        memory = []  # メモリ
        total_rewards = []  # 累積報酬記録用リスト

        # 学習開始
        print("Train")
        print("\t".join(["epoch", "EPSILON", "reward", "total_step", "elapsed_time"]))
        start = time.time()

        # 保存用ディレクトリ作成
        log_time = time.strftime("%d_%b_%Y_%H_%M", time.gmtime())
        model_PATH = self.utility.make_directory(str(log_time), model_name)
        model_PATH2 = self.utility2.make_directory(str(log_time), model_name+"_2")
        self.utility.make_tensorboard(str(log_time), model_name)
        self.utility2.make_tensorboard(str(log_time), model_name+"_2")

        total_reward_torch = []
        total_reward_akida = []
        total_reward_snn = []
        total_reward_random = []
        total_mse_actions = []
        total_index_miss_actions = []
        for epoch in range(self.EPOCH_NUM):#iteration: because it includes the updates
            pobs = self.env.reset()  # 環境初期化
            step = 0  # ステップ数
            done = False  # ゲーム終了フラグ
            total_reward = 0  # 累積報酬
            total_maxq = 0
            total_maxq2 = 0
            loss = 0
            total_mse_action = 0
            total_index_miss_action = 0

            #if network_type == "DoReFa":
            model_keras2 = self.utility2.torch_to_keras(self.model2, self.model_keras2)
            self.model_akida2 = quantize(model_keras2,
                                            input_weight_quantization=self.wbit,
                                            weight_quantization=self.wbit,
                                            activ_quantization=self.abit)
            self.model_snn2 = self.utility2.convert_from_ANN_to_SNN(self.model_akida2)
            
            #Aqui agregar un for mas para cambiar numero de episodios por iteracion. Tambien se tiene que cambiar el tamano de la memoria 
            # para que pueda contener al menos 5 ieraciones (cada iteracion es E episodios de exploracion). Voy a enfrentar entonces C 
            # policy updates con una memoria mas grande. y hara el update tantas veces como 
            #tan grande sea mi memoria, de acuerdo con el algoritmo debo utilizar todos los datos de mi memoria en cada update
            while not done and step < self.STEP_MAX:
                # 行動選択
                action_index = np.random.randint(0, self.action_length - 1)
                pact = self.env.action_from_discrete_to_consecutive(action_index)
                pact = np.int64(pact[0])

                if action_type == "softmax_selection":
                    # softmax行動選択
                    
                    #if network_type == "DoReFa":
                    pobs_ = np.array(pobs, dtype="uint8").reshape((1, self.obs_dim))
                    pact_ = self.model_snn2.predict(pobs_)
                    self.utility2.save_qSNN(pact_, total_step)#############################################333
                    pact, max_p_SNN,max_action_SNN = self.softmax_selection(pact_[0][0][0], tau=self.beta)
                    self.utility2.save_max_p_SNN(max_p_SNN, total_step)
                    self.utility2.save_max_action_SNN(max_action_SNN, total_step)

                    #just to graph Q value, target value, max prob, max action FPNN
                    pobs_tmp = np.array(pobs, dtype="float32").reshape((1, self.obs_dim))
                    pobs_tmp = Variable(torch.from_numpy(pobs_tmp)).to(self.device)
                    pact_tmp = self.model(pobs_tmp)
                    pact_tmp = pact_tmp.cpu().detach().numpy()
                    self.utility.save_qFPNN(pact_tmp, total_step)
                    q_target_tmp = Q_ast.forward(pobs_tmp)
                    target_tmp = copy.deepcopy(q_target_tmp.cpu().data.numpy())
                    self.utility.save_targetFPNN(target_tmp, total_step)
                    pact_tmp,max_p_FPNN, max_action_FPNN = self.softmax_selection(pact_tmp[0], tau=self.beta)
                    self.utility.save_max_p_FPNN(max_p_FPNN, total_step)
                    self.utility.save_max_action_FPNN(max_action_FPNN, total_step)

                    # q = self.model.forward(pobss_)
                    # self.utility.save_qFPNN(q, epoch)
                    # q_target = Q_ast.forward(pobss_)
                    # target = copy.deepcopy(q_target.cpu().data.numpy())
                    # self.utility.save_targetFPNN(target, epoch)

                elif action_type == "random":
                    pass
                else:
                    ic("行動選択エラー")
                    exit()
                # 行動
                obs, reward, done, _ = self.env.step(pact)
                # self.env._render()
                # メモリに蓄積
                reward=reward*self.reward_factor
                memory.append((pobs, pact, reward, obs, done))  # 状態、行動、報酬、行動後の状態、ゲーム終了フラグ
                if len(memory) > self.MEMORY_SIZE:  # メモリサイズを超えていれば消していく
                    memory.pop(0)

                # 次の行動へ
                total_reward += reward
                step += 1
                total_step += 1
                pobs = obs

            for _ in range(self.STEP_MAX):
                # 学習
                if len(memory) == self.MEMORY_SIZE:  # メモリサイズ分溜まっていれば学習
                    if update_type == "CVI":
                        total_maxq = self.update_network_CVI(epoch, total_maxq, memory, Q_ast) #update qith one minibatch
                    elif update_type == "DQN":
                        total_maxq = self.update_network_DQN(epoch, total_maxq, memory, Q_ast) #update qith one minibatch
                    else:
                        ic("更新されていません")
                        exit()
                    # Q関数の更新
                    # if total_step % self.UPDATE_TARGET_Q_FREQ == 0:
                    if epoch % 15 == 0:
                        Q_ast = copy.deepcopy(self.model)

            for idx in range(self.STEP_MAX):#step_max*epoch: 50*100=5000
                # 学習
                if len(memory) == self.MEMORY_SIZE: # メモリサイズ分溜まっていれば学習

                    total_maxq2 = self.update_network_Distill(epoch,total_maxq2, memory, Q_ast2) #update with one minibatch

                    if epoch % 15 == 0:
                        Q_ast2 = copy.deepcopy(self.model2)

                # εの減少
                if self.EPSILON > self.EPSILON_MIN and total_step > self.START_REDUCE_EPSILON:
                    self.EPSILON -= self.EPSILON_DECREASE

            #total_mse_actions, total_index_miss_actions = self.utility2.save_log(network_type,
            total_rewards.append(total_reward)  # 累積報酬を記録
            total_reward_torch, total_reward_akida, total_reward_snn, total_reward_random,\
                 total_mse_actions, total_index_miss_actions = self.utility2.save_log("DoReFa",
                                                                                    action_type,
                                                                                    total_mse_action,
                                                                                    total_index_miss_action,
                                                                                    epoch, start,
                                                                                    total_rewards,
                                                                                    total_step, model_PATH,
                                                                                    total_reward,
                                                                                    total_reward_torch,
                                                                                    total_reward_akida,
                                                                                    total_reward_snn,
                                                                                    total_reward_random,
                                                                                    total_mse_actions,
                                                                                    total_index_miss_actions,
                                                                                    self.beta)


        self.utility.save_neural_network(model_PATH, epoch)
        #if network_type == "DoReFa":
        self.utility2.save_snn_neural_network(model_PATH2, epoch)
        return

    #total_maxq2 = self.update_network_DQN_2(total_step, total_maxq2, memory, Q_ast2)
    def update_network_Distill(self, epoch, total_maxq, memory, Q_ast2):
        # RLによって学習モデルを更新する関数

        # 経験リプレイ
        if epoch % self.TRAIN_FREQ == 0:

            memory_ = np.random.permutation(memory)
            memory_idx = range(len(memory_))

            for i in range(1):  # memory_idx[::self.BATCH_SIZE]:
                batch = np.array(memory_[i:i + self.BATCH_SIZE])  # 経験ミニバッチ
                pobss = np.array(batch[:, 0].tolist(), dtype="float32")
                pacts = np.array(batch[:, 1].tolist(), dtype="int32")
                rewards = np.array(batch[:, 2].tolist(), dtype="float32")
                obss = np.array(batch[:, 3].tolist(), dtype="float32")
                dones = np.array(batch[:, 4].tolist(), dtype="bool")
                # set y
                pobss_ = Variable(torch.from_numpy(pobss)).to(self.device)
                q = self.model2.forward(pobss_)
                # set target
                obss_ = Variable(torch.from_numpy(obss)).to(self.device)
                #maxs, indices = torch.max(Q_ast(obss_).data, 1)
                maxs, indices = torch.max(Q_ast2(obss_).data, 1)
                maxq = maxs.cpu().numpy()  # maxQ
                target=self.model.forward(pobss_)
                # Perform a gradient descent step
                self.optimizer2.zero_grad()
                #loss = nn.MSELoss()(q, Variable(torch.from_numpy(target).to(self.device)))
                loss = nn.MSELoss()(q, target)
                loss.backward()
                self.utility2.save_loss(loss, epoch)
                self.optimizer2.step()
                total_maxq += max(maxq)
        self.utility2.save_total_maxq(total_maxq, epoch)
        return total_maxq

    def update_network_DQN(self, epoch, total_maxq, memory, Q_ast):
        # RLによって学習モデルを更新する関数

        # 経験リプレイ
        if epoch % self.TRAIN_FREQ == 0:

            memory_ = np.random.permutation(memory)
            memory_idx = range(len(memory_))

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
                total_maxq += max(maxq)
        self.utility.save_total_maxq(total_maxq, epoch)
        return total_maxq

    def update_network_CVI(self, epoch, total_maxq, memory, Q_ast):
        # RLによって学習モデルを更新する関数

        # 経験リプレイ
        if epoch % self.TRAIN_FREQ == 0:

            memory_ = np.random.permutation(memory)
            memory_idx = range(len(memory_))

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
                #q_ = self.model.forward(obss_)
                q_target = Q_ast.forward(pobss_)
                q_target_ = Q_ast.forward(obss_)
                maxs, indices = torch.max(Q_ast(obss_).data, 1)
                maxq = maxs.cpu().numpy()  # maxQ
                #target = copy.deepcopy(q.cpu().data.numpy())
                target = copy.deepcopy(q_target.cpu().data.numpy())
                mellowmax_s0 = self.mellowmax_operator(target, alpha=self.alpha, beta=self.beta)
                #target_ = copy.deepcopy(q_.cpu().data.numpy())
                target_ = copy.deepcopy(q_target_.cpu().data.numpy())
                mellowmax_s1 = self.mellowmax_operator(target_, alpha=self.alpha, beta=self.beta)

                for j in range(self.BATCH_SIZE):
                    target[j][pacts[j]] = self.psi_value(reward=rewards[j],
                                                         psi_value=target[j][pacts[j]],
                                                         mellowmax_s0=mellowmax_s0[j],
                                                         mellowmax_s1=mellowmax_s1[j],
                                                         done=dones[j], alpha=self.alpha)
                # Perform a gradient descent step
                self.optimizer.zero_grad()
                loss = nn.MSELoss()(q, Variable(torch.from_numpy(target).to(self.device)))
                loss.backward()
                self.utility.save_loss(loss, epoch)
                self.optimizer.step()
                total_maxq += max(maxq)
        self.utility.save_total_maxq(total_maxq, epoch)
        return total_maxq

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
