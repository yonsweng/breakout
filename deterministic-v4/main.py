import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

gpu = torch.device('cuda:1')

env = gym.make('BreakoutDeterministic-v4')

# hyperparameters
MINIBATCH_SIZE = 32
HISTORY_SIZE = 4
TRAIN_START = 50000
FINAL_EXPLORATION = 0.05
TARGET_UPDATE = 10000
MEMORY_SIZE = 200000
EXPLORATION = 500000
FRAMES_PER_EPOCH = 10000
START_EXPLORATION = 1.
INPUT = env.observation_space.shape  # (210, 160, 3)
OUTPUT = env.action_space.n  # 4
HEIGHT = 84
WIDTH = 84
LEARNING_RATE = 0.00005
DISCOUNT = 0.99
EPSILON = 0.01
MOMENTUM = 0.95


def pre_proc(X):
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x


def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[:, :, i] = pre_proc(s)


def get_game_type(count, info, no_life_game, start_live):
    if count == 1:
        start_live = info['ale.lives']
        # 시작 라이프가 0일 경우, 라이프 없는 게임
        if start_live == 0:
            no_life_game = True
        else:
            no_life_game = False

    return [no_life_game, start_live]


def get_terminal(start_live, info, reward, no_life_game, ter):
    if no_life_game:
        # 목숨이 없는 게임일 경우 Terminal 처리
        if reward < 0:
            ter = True
    else:
        # 목숨 있는 게임일 경우 Terminal 처리
        if start_live > info['ale.lives']:
            ter = True
            start_live = info['ale.lives']

    return [ter, start_live]


def train_minibatch(mainDQN, targetDQN, minibatch, optimizer):
    s_stack = []
    a_stack = []
    r_stack = []
    s1_stack = []
    d_stack = []

    for s_r, a_r, r_r, d_r in minibatch:
        s_stack.append(s_r[:, :, :4])
        a_stack.append(a_r)
        r_stack.append(r_r)
        s1_stack.append(s_r[:, :, 1:])
        d_stack.append(d_r)

    r_stack = np.array(r_stack)
    d_stack = np.array(d_stack) + 0  # terminal이면 1, 아니면 0
    # (minibatch, 84, 84, 4) -> (minibatch, 4, 84, 84)
    s_stack = np.array(s_stack).transpose((0, 3, 1, 2)) / 255.
    s_stack = torch.tensor(s_stack, device=gpu).float()
    s1_stack = np.array(s1_stack).transpose((0, 3, 1, 2)) / 255.
    s1_stack = torch.tensor(s1_stack, device=gpu).float()

    Q1 = targetDQN(s1_stack).cpu().detach().numpy().max(axis=1)
    y = r_stack + (1 - d_stack) * DISCOUNT * Q1
    y = torch.tensor(y, device=gpu).float()

    Q = mainDQN(s_stack)
    one_hot = torch.tensor(np.eye(OUTPUT)[a_stack], device=gpu).float()
    q_val = torch.sum(torch.mul(Q, one_hot), 1)

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_val, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_data(epoch, epoch_score, average_reward, epoch_Q, average_Q):
    plt.clf()
    epoch_score.append(np.mean(average_reward))
    epoch_Q.append(np.mean(average_Q))

    plt.subplot(211)
    plt.axis([0, epoch, 0, np.max(epoch_Q) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Action Value(Q)')
    plt.plot(epoch_Q)

    plt.subplot(212)
    plt.axis([0, epoch, 0, np.max(epoch_score) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Reward per Episode')
    plt.plot(epoch_score, "r")

    plt.savefig("graph.png")


class DQN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 2)
        self.fc1 = nn.Linear(64*9*9, 256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.reshape(-1, 64*9*9)))
        x = self.fc2(x)
        return x


def get_action(q, e):
    if e > np.random.rand(1):
        action = np.random.randint(OUTPUT)
    else:
        action = np.argmax(q)
    return action


def main():
    mainDQN = DQN(HISTORY_SIZE, OUTPUT)
    targetDQN = copy.deepcopy(mainDQN)
    mainDQN.to(gpu)
    targetDQN.to(gpu)

    optimizer = torch.optim.RMSprop(mainDQN.parameters(), lr=LEARNING_RATE,
                                    eps=EPSILON, momentum=MOMENTUM)

    e = START_EXPLORATION
    episode, epoch, frame = 0, 0, 0

    recent_rlist = deque(maxlen=100)
    epoch_score, epoch_Q = deque(), deque()
    average_Q, average_reward = deque(), deque()

    epoch_on = False
    no_life_game = False
    replay_memory = deque(maxlen=MEMORY_SIZE)

    max_score = 0

    # Train agent during 200 epoch
    while epoch < 1000:
        episode += 1

        history = np.zeros([84, 84, 5], dtype=np.uint8)
        rall, count = 0, 0  # rall: 한 episode의 reward 합
        done = False
        start_lives = 0

        s = env.reset()
        prev_s = pre_proc(s)
        history[:, :, :HISTORY_SIZE] = np.zeros((84, 84, HISTORY_SIZE))

        while not done:  # until episode ends
            frame += 1  # frame counts in all episodes
            count += 1  # frame counts in an episode

            # e-greedy
            if e > FINAL_EXPLORATION and frame > TRAIN_START:
                e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

            # 히스토리의 0~3 부분으로 Q값 예측
            X = history[:, :, :4].transpose(2, 0, 1).reshape(-1, 4, 84, 84)/255
            X = torch.tensor(X, device=gpu).float()
            Q = mainDQN(X)
            Q = Q.reshape(-1).cpu().detach().numpy()
            average_Q.append(np.max(Q))

            # 1만 frame마다 Q 출력
            if frame % FRAMES_PER_EPOCH == 0:
                print('Q:', Q)

            # 액션 선택
            action = get_action(Q, e)

            # s1 : next frame / r : reward
            s1, r, done, info = env.step(action)

            # 라이프가 있는 게임이면 no_life_game=True
            no_life_game, start_lives = \
                get_game_type(count, info, no_life_game, start_lives)

            # 라이프가 줄어들거나 negative 리워드를 받았을 때 terminal 처리를 해줌
            ter, start_lives = \
                get_terminal(start_lives, info, r, no_life_game, done)

            reward = -1 if ter else r

            # 앞 frame들에 discounted reward 전파
            if reward != 0:
                running_reward = reward
                idx = replay_memory.__len__() - 1
                while idx >= 0 and replay_memory[idx][2] == 0:
                    running_reward *= DISCOUNT
                    replay_memory[idx][2] = running_reward
                    idx -= 1

            # 새로운 프레임을 히스토리 마지막에 넣어줌
            pre_proc_s1 = pre_proc(s1)
            history[:, :, 4] = pre_proc_s1 - prev_s
            prev_s = pre_proc_s1

            # 메모리 저장 효율을 높이기 위해 5개의 프레임을 가진 히스토리를 저장
            # state와 next_state는 3개의 데이터가 겹침을 이용.
            replay_memory.append([np.copy(history[:, :, :]),
                                  action, reward, ter])
            history[:, :, :4] = history[:, :, 1:]

            rall += r

            if frame > TRAIN_START:
                if frame % 4 == 0:
                    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
                    train_minibatch(mainDQN, targetDQN, minibatch, optimizer)

                # 1만 프레임일때마다 target_net 업데이트
                if frame % TARGET_UPDATE == 0:
                    targetDQN = copy.deepcopy(mainDQN)
                    targetDQN.to(gpu)
                    print('targetDQN updated')

            # 1 epoch마다 plot
            if (frame - TRAIN_START) % FRAMES_PER_EPOCH == 0:
                epoch_on = True

        recent_rlist.append(rall)  # 최근 100개 episode의 reward
        average_reward.append(rall)  # 한 epoch의 모든 reward

        if episode % 100 == 0:
            print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | "
                  "Reward:{3:3.0f} | e-greedy:{4:.5f} | "
                  "Avg_Max_Q:{5:2.5f} | Recent reward:{6:.5f}"
                  .format(episode, frame, count, rall, e,
                          np.mean(average_Q), np.mean(recent_rlist)))

            # save the best model
            score = np.mean(recent_rlist)
            if score > max_score:
                max_score = score
                torch.save(mainDQN.state_dict(), 'model.pt')

        if epoch_on:
            epoch += 1
            plot_data(epoch, epoch_score, average_reward, epoch_Q, average_Q)
            epoch_on = False
            average_reward = deque()
            average_Q = deque()


main()
