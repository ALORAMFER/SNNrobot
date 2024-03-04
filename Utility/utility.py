#coding: utf-8
import math
import time
import os
import torch
import numpy as np
from Env.environment import DiscreteMountainCar
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from cnn2snn import quantize
from cnn2snn import convert
from icecream import ic

class Utility():
    def __init__(self, beta, wbit, abit, log_freq, step_max, model, model_keras):
        self.env = DiscreteMountainCar()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.wbit = wbit
        self.abit = abit
        self.LOG_FREQ = log_freq
        self.action_length = self.env.action_space
        self.obs_dim = self.env.observation_space
        self.STEP_MAX = step_max
        self.model = model
        self.model_keras = model_keras
        return

    def make_directory(self, str_time, model_name):
        # 学習結果を保存するためのディレクトリを作成する関数
        mkdir_PATH = "./RL/model/" + model_name
        if not os.path.exists("RL"):
            os.makedirs("RL")
        if not os.path.exists("RL/runs"):
            os.makedirs("RL/runs")
        if not os.path.exists("RL/model"):
            os.makedirs("RL/model")
        if not os.path.exists(mkdir_PATH):
            os.mkdir(mkdir_PATH)

        model_PATH = './RL/model/' + model_name + '/' + str_time
        if not os.path.exists(model_PATH):
            os.mkdir(model_PATH)
            os.mkdir(model_PATH + "/snn")
            os.mkdir(model_PATH + "/pytorch")
        return model_PATH

    def make_tensorboard(self, str_time, model_name):
        log_PATH = "./RL/runs/" + model_name + '/' + str_time
        self.writer = SummaryWriter(log_dir=log_PATH)
        return

    def model_convert_softmax_selection(self,beta):

        # ---------------------------- DoReFaネットワーク -------------------------------------
        # softmax行動選択
        obs_torch = self.env.reset()
        step = 0
        total_reward_torch = 0
        done = False  # ゲーム終了フラグ
        while not done and step < self.STEP_MAX:
            pobs_torch = np.array(obs_torch, dtype="float32")
            pobs_torch = Variable(torch.from_numpy(pobs_torch)).to(self.device)
            pact_torch = self.model(pobs_torch)
            pact_torch = pact_torch.to('cpu').detach().numpy().copy()
            pact_torch = self.softmax_selection(pact_torch, tau=beta)
            obs_torch, reward_torch, done, _ = self.env.step(pact_torch)
            total_reward_torch += reward_torch
            step += 1
        # -----------------------------量子化ネットワーク--------------------------------------
        model_keras = self.torch_to_keras(self.model, self.model_keras)
        self.model_akida = quantize(model_keras,
                                    input_weight_quantization=self.wbit,
                                    weight_quantization=self.wbit,
                                    activ_quantization=self.abit)
        obs_akida = self.env.reset()
        step = 0
        total_reward_akida = 0
        done = False  # ゲーム終了フラグ
        while not done and step < self.STEP_MAX:
            pobs_akida = np.array(obs_akida, dtype="float32").reshape([1, self.obs_dim])
            pact_akida = self.model_akida.predict(pobs_akida, verbose=0)
            pact_akida = self.softmax_selection(pact_akida[0], tau=beta)
            obs_akida, reward_akida, done, _ = self.env.step(pact_akida)
            total_reward_akida += reward_akida
            step += 1
        # -----------------------------SNNネットワーク--------------------------------------
        self.model_snn = self.convert_from_ANN_to_SNN(self.model_akida)
        obs_snn = self.env.reset()
        step = 0
        total_reward_snn = 0
        done = False
        while not done and step < self.STEP_MAX:
            pobs_snn = np.array(obs_snn, dtype="uint8").reshape([1, self.obs_dim])
            pact_snn = self.model_snn.predict(pobs_snn)
            pact_snn = self.softmax_selection(pact_snn[0][0][0], tau=beta)
            obs_snn, reward_snn, done, _ = self.env.step(pact_snn)
            total_reward_snn += reward_snn
            step += 1
        return total_reward_torch, total_reward_akida, total_reward_snn

    def model_convert_floatNN_to_SNN(self, beta):
        # ---------------------------- DoReFaネットワーク -------------------------------------
        # softmax行動選択
        obs_torch = self.env.reset()
        step = 0
        total_reward_torch = 0
        done = False  # ゲーム終了フラグ
        while not done and step < self.STEP_MAX:
            pobs_torch = np.array(obs_torch, dtype="float32")
            pobs_torch = Variable(torch.from_numpy(pobs_torch)).to(self.device)
            pact_torch = self.model(pobs_torch)
            pact_torch = pact_torch.to('cpu').detach().numpy().copy()
            pact_torch = self.softmax_selection(pact_torch, tau=beta)
            obs_torch, reward_torch, done, _ = self.env.step(pact_torch)
            total_reward_torch += reward_torch
            step += 1

        model_keras = self.torch_to_keras(self.model, self.model_keras)
        self.model_akida = quantize(model_keras,
                                    input_weight_quantization=self.wbit,
                                    weight_quantization=self.wbit,
                                    activ_quantization=self.abit)
        # -----------------------------量子化ネットワーク--------------------------------------
        obs_akida = self.env.reset()
        step = 0
        total_reward_akida = 0
        done = False  # ゲーム終了フラグ
        while not done and step < self.STEP_MAX:
            pobs_akida = np.array(obs_akida, dtype="float32").reshape([1, self.obs_dim])
            pact_akida = self.model_akida.predict(pobs_akida, verbose=0)
            pact_akida = np.argmax(pact_akida[0])
            obs_akida, reward_akida, done, _ = self.env.step(pact_akida)
            total_reward_akida += reward_akida
            step += 1

        # -----------------------------SNNネットワーク--------------------------------------
        self.model_snn = self.convert_from_ANN_to_SNN(self.model_akida)
        obs_snn = self.env.reset()
        step = 0
        total_reward_snn = 0
        done = False
        while not done and step < self.STEP_MAX:
            pobs_snn = np.array(obs_snn, dtype="uint8").reshape([1, self.obs_dim])
            pact_snn = self.model_snn.predict(pobs_snn)
            pact_snn = np.argmax(pact_snn[0][0][0])
            obs_snn, reward_snn, done, _ = self.env.step(pact_snn)
            total_reward_snn += reward_snn
            step += 1
        
        # -----------------------------SNNネットワーク through Distillation--------------------------------------
        
        
        
        
        return total_reward_torch, total_reward_akida, total_reward_snn

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
        action = np.random.choice(np.arange(len(values)), p=p)                # 確率分布pに従ってランダムで選択
        return action

    def save_neural_network(self, model_path_name, epoch):
        model_name = model_path_name + '/pytorch/' + str(epoch) + "_" + time.strftime("%H_%M_%S", time.gmtime()) + '.pth'
        torch.save(self.model.state_dict(), model_name)
        return

    def save_snn_neural_network(self, model_path_name, epoch):
        model_name = model_path_name + '/snn/' + str(epoch) + "_" + time.strftime("%H_%M_%S", time.gmtime()) + '.fbz'
        self.model_snn.save(model_name)
        return

    def convert_from_ANN_to_SNN(self, model_akida):
        model_snn = convert(model_akida, input_is_image=False)
        return model_snn

    def torch_to_keras(self, model_torch, model_keras):
        model_torch = model_torch.cpu()
        torch_params = model_torch.state_dict()
        torch_keys = list(torch_params.keys())
        flatten_pass_flag = False
        for layer in model_keras.layers:
            for var in (layer.weights):
                torch_key = torch_keys.pop(0)
                torch_param = torch_params[torch_key].numpy()
                if len(torch_param.shape) == 4:  # Convolution層
                    var.assign(torch_param.transpose(2, 3, 1, 0))
                elif len(torch_param.shape) == 2:  # layer層
                    var.assign(torch_param.transpose(1, 0))
                else:
                    var.assign(torch_param)
        model_torch.cuda()
        return model_keras

    def save_log(self, network_type, action_type, epoch, start, total_rewards,
                 total_step, model_PATH, total_reward_torch, total_reward_akida, total_reward_snn):
        """
        モデル変換による出力結果
        """
        if network_type == "DoReFa":
            if action_type == "softmax_selection":
                reward_torch, reward_akida, reward_snn = self.model_convert_softmax_selection(self.beta)
            total_reward_torch.append(reward_torch)
            total_reward_akida.append(reward_akida)
            total_reward_snn.append(reward_snn)
            self.writer.add_scalar('reward_torch', reward_torch, epoch)
            self.writer.add_scalar('reward_akida', reward_akida, epoch)
            self.writer.add_scalar('reward_snn', reward_snn, epoch)
            if (epoch + 1) % (self.LOG_FREQ) == 0:
                r = sum(total_rewards[((epoch + 1) - self.LOG_FREQ):(epoch + 1)]) / self.LOG_FREQ  # ログ出力間隔での平均累積報酬
                elapsed_time = time.time() - start
                print("\t".join(map(str, [epoch + 1, r, total_step, str(elapsed_time) + "[sec]"])))  # ログ出力
                self.writer.add_scalar('total_rewards_torch', sum(total_reward_torch) / self.LOG_FREQ, epoch)
                self.writer.add_scalar('total_rewards_akida', sum(total_reward_akida) / self.LOG_FREQ, epoch)
                self.writer.add_scalar('total_rewards_snn', sum(total_reward_snn) / self.LOG_FREQ, epoch)
                
                total_reward_torch = []
                total_reward_akida = []
                total_reward_snn = []
            if (epoch + 1) % (self.LOG_FREQ * 10) == 0:  # モデルを保存する間隔
                self.save_neural_network(model_PATH, epoch)
                self.save_snn_neural_network(model_PATH, epoch)
        if network_type == "float":
            reward_torch, reward_akida, reward_snn = self.model_convert_floatNN_to_SNN(self.beta)
            total_reward_torch.append(reward_torch)
            total_reward_akida.append(reward_akida)
            total_reward_snn.append(reward_snn)
            self.writer.add_scalar('reward_torch', reward_torch, epoch)
            self.writer.add_scalar('reward_akida', reward_akida, epoch)
            self.writer.add_scalar('reward_snn', reward_snn, epoch)
            if (epoch + 1) % (self.LOG_FREQ) == 0:
                r = sum(total_rewards[((epoch + 1) - self.LOG_FREQ):(epoch + 1)]) / self.LOG_FREQ  # ログ出力間隔での平均累積報酬
                elapsed_time = time.time() - start
                print("\t".join(map(str, [epoch + 1, r, total_step, str(elapsed_time) + "[sec]"])))  # ログ出力
                self.writer.add_scalar('total_rewards_torch', sum(total_reward_torch) / self.LOG_FREQ, epoch)
                self.writer.add_scalar('total_rewards_akida', sum(total_reward_akida) / self.LOG_FREQ, epoch)
                self.writer.add_scalar('total_rewards_snn', sum(total_reward_snn) / self.LOG_FREQ, epoch)
                total_reward_torch = []
                total_reward_akida = []
                total_reward_snn = []
            if (epoch + 1) % (self.LOG_FREQ * 10) == 0:  # モデルを保存する間隔
                self.save_neural_network(model_PATH, epoch)
                self.save_snn_neural_network(model_PATH, epoch)

        return total_reward_torch, total_reward_akida, total_reward_snn 

    def save_loss(self, loss, total_step):
        self.writer.add_scalar('loss', loss, total_step)
        return

    def save_total_maxq(self, total_maxq, total_step):
        self.writer.add_scalar('total_maxq', total_maxq, total_step)
        return

    def save_qFPNN(self, q, total_step):
        for i in range(3):
            self.writer.add_scalar('q_FPNN'+"{0}".format(i), q[0,i].tolist(), total_step)
        return
    
    def save_qSNN(self, q, total_step):
        for i in range(3):
            self.writer.add_scalar('q_SNN'+"{0}".format(i), q[0, 0, 0, i], total_step)
        return
    
    def save_targetFPNN(self, target, total_step):
        for i in range(3):
            self.writer.add_scalar('target'+"{0}".format(i), target[0,i].tolist(), total_step)
        return
    
    def save_max_action_SNN(self, max_action_SNN, total_step):
        self.writer.add_scalar('max_action_SNN', max_action_SNN, total_step)
        return

    def save_max_p_SNN(self, max_p_SNN, total_step):
        self.writer.add_scalar('max_p_SNN', max_p_SNN, total_step)
        return

    def save_max_action_FPNN(self, max_action_FPNN, total_step):
        self.writer.add_scalar('max_action_FPNN', max_action_FPNN, total_step)
        return

    def save_max_p_FPNN(self, max_p_FPNN, total_step):
        self.writer.add_scalar('max_p_FPNN', max_p_FPNN, total_step)
        return