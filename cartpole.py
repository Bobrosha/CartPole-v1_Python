import gym
import numpy as np
from gym import envs
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import sys
import random

# Агент с глубоким Q-обучением
class Agent():  
    # Инициализация
    def __init__(self, state_space, action_space):
        self.state_size = state_space # Количество параметров входящих в нейронную сеть
        self.action_size = action_space # Количество вариантов действий агента
        self.memory = deque(maxlen = 2000) # Хранилище совершенных агентом действий и их последствий
        self.gamma = 0.95 # Фактор дисконтирования. Коэффициент уменьшения вознаграждения
       
        self.exploration_rate = 1.0 # Процент рандомных действий агента
        self.exploration_min = 0.01 # Минимальный процент
        self.exploration_decay = 0.995 # Уменьшение рандомных решений после каждого обучения
        self.learning_rate = 0.001 # Изменение скорости обучения оптимизатора с течением времени
        self.model = self.build_model()
    
    # Создание скомпилированной модели
    def build_model(self):
        model = Sequential()
        
        model.add(Dense(24, activation='relu', 
                        input_dim=self.state_size))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', 
                      optimizer=Adam(lr = self.learning_rate))
        return model
    
    # Сохранение модели
    def save_model(self, name):
        self.model.save(name)
        
    # Загрузка модели
    def load_model(self, name):
        model.load_weights(name)
        self.exploration_rate = self.exploration_min
        
    # Сохранение истории в очередь
    def remember(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))
    
    # Определяет и возвращает действие
    def action(self, state):
        # Случайный выбор действия
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        
        # Использование предсказания агента для выбора действия
        q_values = self.model.predict(state)
        return np.argmax(q_values[0]) # Возвращает действие
    
    # Обучение
    def training(self, batch_size):
        if len(self.memory) < batch_size: return
        # Случайная выборка batch_size элементов для обучения агента
        sample_batch = random.sample(self.memory, batch_size)
        for state, action, reward, state_next, done in sample_batch:
            q_values = self.model.predict(state)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(state_next)[0])
            # Формируем цель обучения сети
            q_values[0][action] = target
            # Обучение сети
            self.model.fit(state, q_values, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min: 
            self.exploration_rate *= self.exploration_decay
            
def play():
    env = gym.make('CartPole-v1') # Создаем среду
    
    observ_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    # DQN - глубокая Q-нейронная сеть
    agent = Agent(observ_space, action_space) # Создаем агента
    
    episodes = 500 # Число игровых эпизодов
    
    scores = deque(maxlen = 100)
    # scores - хранит длительность последних 100 игр
    
    #
    # Цикл игры и обучения
    #

    for e in range(episodes + 1):
        # Получаем начальное состояние объекта перед началом каждой игры (каждого эпизода)
        state = env.reset()
        # state[0] - позиция тележки
        # state[1] - скорость тележки
        # state[2] - угол отклонения шеста от вертикали в радианах
        # state[3] - скорость изменения угла наклона шеста
        
        state = np.reshape(state, [1, observ_space])
        
        frames = 0
        done = False
        
        while not done:
            env.render() # Графическое отображение симуляции
            frames += 1
            action = agent.action(state) # Определяем очередное действие
            
            state_next, reward, done, info = env.step(action)
            # Получаем от среды обновленные состояние объекта, награду, значение флага завершения игры
            # В каждый момент игры, пока не наступило одно из условий ее прекращения, награда равна 1
            
            state_next = np.reshape(state_next, [1, observ_space])
            
            agent.remember(state, action, reward, state_next, done)
            # Запоминаем предыдущее состояние объекта, действие, 
            # награду за это действие, текущее состояние, значение done
            
            state = state_next # Обновляем текущее состояние
            
        print("Эпизод: {:>3}/{}, продолжительность игры в кадрах: {:>3}".format(e, episodes, frames))
        
        scores.append(frames)
        score_mean = np.mean(scores)
            
        print('Средняя продолжительность: ', score_mean, '\n')
        
        # Продолжаем обучать агента
        agent.training(32)
        
    return 'Simulation complited'
    
play()
