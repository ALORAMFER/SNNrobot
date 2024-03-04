#coding: utf-8

from RL_method.rl_method import RL_method

from icecream import ic
from torch.utils.tensorboard import SummaryWriter

def main(ModelSelect, model_name, wbit, abit):
    rl = RL_method(wbit, abit)
    rl.main(ModelSelect, model_name)
    return


if __name__ == '__main__':
    
    model_mode = "QPN" # random | QPN | CVI| QQN | DQN | STDP | QQN-epsilon | QQN_Distillation | QPN_Distillation
    wbit = 8
    abit = 4
    model_name = "QPN-0p97-lr0p000025-beta0p6cte-1rewardF-GPm0p3"
    for _ in range(20):
        main(model_mode, model_name, wbit, abit)