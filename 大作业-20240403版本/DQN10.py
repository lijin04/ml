import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

# 定义地图中的字符表示
EMPTY = '.'
BLUE_BASE = '*'
RED_BASE = '#'

# 定义动作空间
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3

# 定义战斗机类
class Fighter:
    def __init__(self, id, row, col, fuel_capacity, missile_capacity):
        self.id = id
        self.row = row
        self.col = col
        self.fuel_capacity = fuel_capacity
        self.missile_capacity = missile_capacity
        self.fuel = 0
        self.missile = 0

# 定义基地类
class Base:
    def __init__(self, row, col, fuel_reserve, missile_reserve, defense_value, military_value):
        self.row = row
        self.col = col
        self.fuel_reserve = fuel_reserve
        self.missile_reserve = missile_reserve
        self.defense_value = defense_value
        self.military_value = military_value

# 定义智能体类
class DQNAgent:
    def __init__(self, state_size, action_size, agent_id):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # 折扣因子
        self.epsilon = 1   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    # 构建模型
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    # 更新目标网络
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 存储记忆
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 选择动作
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float)
        act_values = self.model(state_tensor)
        return np.argmax(act_values.detach().numpy())

    # 学习并更新模型
    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float)
                # 使用行为网络选择动作，使用目标网络计算Q值
                target = reward + self.gamma * np.amax(self.model(next_state_tensor).detach().numpy())
            state_tensor = torch.tensor(state, dtype=torch.float)
            target_f = self.model(state_tensor).detach().numpy()
            target_f[action] = target
            target_f = torch.tensor(target_f, dtype=torch.float)
            self.model.train()
            self.model.zero_grad()
            criterion = nn.MSELoss()
            output = self.model(state_tensor)
            loss = criterion(output, target_f)
            
            loss.backward()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 解析地图
def parse_map(lines):
    n, m = map(int, lines.pop(0).split())
    grid = [list(lines.pop(0).strip()) for _ in range(n)]
    return grid, lines

# 解析基地信息
def parse_bases(lines):
    blue_bases = []
    red_bases = []
    blue_count = int(lines.pop(0))
    for _ in range(blue_count):
        row, col= map(int, lines.pop(0).split())
        fuel, missile, defense, military = map(int, lines.pop(0).split())
        # row, col, fuel, missile, defense, military = map(int, lines.pop(0).split())
        blue_bases.append(Base(row, col, fuel, missile, defense, military))
    red_count = int(lines.pop(0))
    for _ in range(red_count):
        row, col= map(int, lines.pop(0).split())
        fuel, missile, defense, military = map(int, lines.pop(0).split())
        # row, col, fuel, missile, defense, military = map(int, lines.pop(0).split())
        red_bases.append(Base(row, col, fuel, missile, defense, military))
    return blue_bases, red_bases, lines

# 解析战斗机信息
def parse_fighters(lines):
    fighter_count = int(lines.pop(0))
    fighters = []
    for i in range(fighter_count):
        id = i
        row, col, fuel_capacity, missile_capacity = map(int, lines.pop(0).split())
        fighters.append(Fighter(id, row, col, fuel_capacity, missile_capacity))
    return fighters

# 构建环境状态
def build_state(grid, fighters, blue_bases, red_bases):
    state = []
    # 添加地图信息
    for row in grid:
        for cell in row:
            # 将地图中的字符转换为整数或独热编码向量
            if cell == EMPTY:
                state.append(0)  # 例如，空单元格可以表示为0
            elif cell == BLUE_BASE:
                state.append(1)  # 例如，蓝色基地可以表示为1
            elif cell == RED_BASE:
                state.append(2)  # 例如，红色基地可以表示为2
    # 添加战斗机信息
    for fighter in fighters:
        state.append(fighter.row)
        state.append(fighter.col)
        state.append(fighter.fuel)
        state.append(fighter.missile)
    # 添加基地信息
    for base in blue_bases:
        state.append(base.row)
        state.append(base.col)
        state.append(base.fuel_reserve)
        state.append(base.missile_reserve)
        state.append(base.defense_value)
        state.append(base.military_value)
    for base in red_bases:
        state.append(base.row)
        state.append(base.col)
        state.append(base.fuel_reserve)
        state.append(base.missile_reserve)
        state.append(base.defense_value)
        state.append(base.military_value)
    for fighter in fighters:
        for base in red_bases:
            distance = abs(fighter.row - base.row) + abs(fighter.col - base.col)
            state.append(distance)
        for base in blue_bases:
            distance = abs(fighter.row - base.row) + abs(fighter.col - base.col)
            state.append(distance)
    return np.array(state, dtype=np.float32)

def main():
    episodes = 100
    max_score = 100
    for e in range(episodes):
        with open("./data/testcase10.in", "r") as file:
            lines = file.readlines()        
    
        grid, lines = parse_map(lines)
        blue_bases, red_bases, lines = parse_bases(lines)
        fighters = parse_fighters(lines)
        
        state_size = len(grid) * len(grid[0]) + len(fighters) * 4 + len(blue_bases) * 6 + len(red_bases) * 6 + len(fighters) * len(blue_bases) + len(fighters) * len(red_bases)
        action_size = 4  # 修改动作空间大小为4
        agents = [DQNAgent(state_size, action_size, agent_id) for agent_id in range(len(fighters))]

        # 加载预训练的模型参数
        for agent_id, agent in enumerate(agents):
            agent.model.load_state_dict(torch.load(f"./model/model10_agent_{agent_id}.pt"))
            agent.update_target_model()

        batch_size = 8
        step = 0
        
        score = 0
        done = False
        state = build_state(grid, fighters, blue_bases, red_bases)
        while step < 15000:
            for agent in agents:
                agent.epsilon *= agent.epsilon_decay
                # action = agent.act(state)
                # fighter = fighters[agent.agent_id]
                
            actions = [agent.act(state) for agent in agents]
            with open("./data/testcase101.out", "a") as file:
                # 更新战斗机状态
                for fighter, action, agent in zip(fighters, actions, agents):
                    reward = 0
                    #fuel
                    for base in blue_bases:
                        if base.row == fighter.row and base.col == fighter.col and base.fuel_reserve > 0 and fighter.fuel_capacity - fighter.fuel > 0:
                            fuel = min(base.fuel_reserve, fighter.fuel_capacity - fighter.fuel)
                            fighter.fuel += fuel
                            base.fuel_reserve -= fuel
                            file.write(f"fuel {fighter.id} {fuel}\n")
                            reward += 100

                    #missile
                    for base in blue_bases:
                        if base.row == fighter.row and base.col == fighter.col and base.missile_reserve > 0 and fighter.missile_capacity - fighter.missile>0:
                            missile = min(base.missile_reserve, fighter.missile_capacity - fighter.missile)
                            fighter.missile += missile
                            base.missile_reserve -= missile
                            file.write(f"missile {fighter.id} {missile}\n")  
                            reward += 100 

                    #attack
                    # 检查是否有红色基地在攻击范围内
                    for base in red_bases:
                        attack_uprow = fighter.row - 1
                        attack_upcol = fighter.col
                        if base.row == attack_uprow and base.col == attack_upcol:
                            if fighter.missile > 0 and base.defense_value > 0:
                                missile = min(fighter.missile, base.defense_value)
                                base.defense_value -= missile
                                fighter.missile -= missile
                                reward += 10
                                file.write(f"attack {fighter.id} 0 {missile}\n")
                                if base.defense_value <= 0:
                                    reward += base.military_value 
                                    score += base.military_value
                                    base.military_value = 0  # 标记为摧毁
                        
                        attack_downrow = fighter.row + 1
                        attack_downcol = fighter.col
                        if base.row == attack_downrow and base.col == attack_downcol:
                            if fighter.missile > 0 and base.defense_value > 0:
                                missile = min(fighter.missile, base.defense_value)
                                base.defense_value -= missile
                                fighter.missile -= missile
                                reward += 10
                                file.write(f"attack {fighter.id} 1 {missile}\n")
                                if base.defense_value <= 0:
                                    reward += base.military_value 
                                    score += base.military_value
                                    base.military_value = 0  # 标记为摧毁
                                    
                        attack_leftrow = fighter.row 
                        attack_leftcol = fighter.col - 1
                        if base.row == attack_leftrow and base.col == attack_leftcol:
                            if fighter.missile > 0 and base.defense_value > 0:
                                missile = min(fighter.missile, base.defense_value)
                                base.defense_value -= missile
                                fighter.missile -= missile
                                reward += 10
                                file.write(f"attack {fighter.id} 2 {missile}\n")
                                if base.defense_value <= 0:
                                    reward += base.military_value
                                    score += base.military_value 
                                    base.military_value = 0  # 标记为摧毁

                        attack_rightrow = fighter.row - 1
                        attack_rightcol = fighter.col
                        if base.row == attack_rightrow and base.col == attack_rightcol:
                            if fighter.missile > 0 and base.defense_value > 0:
                                missile = min(fighter.missile, base.defense_value)
                                base.defense_value -= missile
                                fighter.missile -= missile
                                reward += 10
                                file.write(f"attack {fighter.id} 3 {missile}\n")
                                if base.defense_value <= 0:
                                    reward += base.military_value 
                                    score += base.military_value
                                    base.military_value = 0  # 标记为摧毁
                    #移动
                    if not done:
                        if action == MOVE_UP:
                            # 执行向上移动操作
                            if fighter.row > 0 and fighter.fuel > 0:
                                bool_move = 1
                                for base in red_bases:
                                    if base.row == fighter.row - 1 and base.col == fighter.col:
                                        if base.defense_value > 0:
                                            bool_move = 0
                                            reward -= 5
                                            break
                                if bool_move:   
                                    fighter.row -= 1
                                    fighter.fuel -= 1
                                    reward += 1
                                    file.write(f"move {fighter.id} {action}\n")
                            else:
                                reward -= 10
                                            
                        elif action == MOVE_DOWN:
                            # 执行向下移动操作
                            if fighter.row < len(grid)-1 and  fighter.fuel > 0:
                                bool_move = 1
                                for base in red_bases:
                                    if base.row == fighter.row + 1 and base.col == fighter.col:
                                        if base.defense_value > 0:
                                            bool_move = 0
                                            reward -= 5
                                            break
                                if bool_move:   
                                    fighter.row += 1
                                    fighter.fuel -= 1
                                    reward += 1   
                                    file.write(f"move {fighter.id} {action}\n")
                            else:
                                reward -= 10

                        elif action == MOVE_LEFT:
                            # 执行向左移动操作
                            if fighter.col > 0 and fighter.fuel > 0:
                                bool_move = 1
                                for base in red_bases:
                                    if base.row == fighter.row  and base.col == fighter.col - 1:
                                        if base.defense_value > 0:
                                            bool_move = 0
                                            reward -= 5
                                            break
                                if bool_move:   
                                    fighter.col -= 1
                                    fighter.fuel -= 1
                                    reward += 1
                                    file.write(f"move {fighter.id} {action}\n")
                            else:
                                reward -= 10
                        elif action == MOVE_RIGHT:
                            # 执行向右移动操作
                            if fighter.col < len(grid[0])-1 and fighter.fuel > 0:
                                bool_move = 1
                                for base in red_bases:
                                    if base.row == fighter.row  and base.col == fighter.col + 1:
                                        if base.defense_value > 0:
                                            bool_move = 0
                                            reward -= 5
                                            break
                                if bool_move:   
                                    fighter.col += 1
                                    fighter.fuel -= 1
                                    reward += 1
                                    file.write(f"move {fighter.id} {action}\n")
                            else:
                                reward -= 10  
                    next_state = build_state(grid, fighters, blue_bases, red_bases)              
                    agent.remember(state, action, reward, next_state, done)
                    agent.learn(batch_size)
                    state = next_state 
                    print("episodes", e," step ", step, "OK, ","score: ",score," fighter id: ",fighter.id," reward: ",reward," action: ",action," fuel: ",fighter.fuel, " misile: ", fighter.missile, "max_score: ", max_score)  # 表示该帧指令输出结束，并刷新输出                       
                file.write("OK\n")

            step += 1    
            
            # 检查是否所有的红方基地都被摧毁，如果是，则设置done为True
            if all(base.military_value == 0 for base in red_bases):
                done = True
                break

            if all(fighter.fuel == 0 for fighter in fighters):
                break
            # for agent in agents:
            #     agent.remember(state, action, reward, next_state, done)
            #     agent.learn(batch_size)
            #     state = next_state
            # print("episodes", e," step ", step, "OK, ","score: ",score," reward: ",reward," action: ",action," fuel: ",fighter.fuel)  # 表示该帧指令输出结束，并刷新输出
            # 保存模型参数
            if max_score < score:
                max_score = score
                for agent_id, agent in enumerate(agents):
                    torch.save(agent.model.state_dict(), f"./model/model10_agent_{agent_id}.pt")
                    print("max_score: ", max_score)
                    agent.update_target_model()


if __name__ == "__main__":
    main()