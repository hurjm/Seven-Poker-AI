from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
import tensorflow as tf
import numpy as np
from Calc import *

episode = 1
  
class A3CAgent:
    def __init__(self):
        self.action_size = 2
        self.actor_lr = 0.00001
        self.critic_lr = 0.00001
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        K.set_learning_phase(1)
        self.sess.run(tf.global_variables_initializer())
        
        self.ai1_training_states, self.training_actions, self.training_rewards = [], [[], [], [], []], []
        self.ai2_training_states, self.training_actions, self.training_rewards = [], [[], [], [], []], []
        self.training_states = [[], []]
        
        self.actor_model = [self.build_flop_actor_model(), self.build_turn_actor_model()]
        self.critic_model = [self.build_flop_critic_model(), self.build_turn_critic_model()]
        
        self.optimizer_actor = [self.bulid_actor_optimizer(0), self.bulid_actor_optimizer(1)]
        self.optimizer_critic = [self.bulid_critic_optimizer(0), self.bulid_critic_optimizer(1)]
        
        #self.load_model()
 

    def build_flop_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(200, input_dim=98, activation='relu', kernel_initializer='he_normal'))
        
        for _ in range(10):
            actor.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
        
        actor.add(Dense(2, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor

    def build_flop_critic_model(self):
        critic = Sequential()
        
        critic.add(Dense(200, input_dim=98, activation='relu', kernel_initializer='he_normal'))
        
        for _ in range(10):
            critic.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
    
        critic.add(Dense(1, activation='linear', kernel_initializer='he_normal'))
        
        critic.summary()

        return critic
        
    def build_turn_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(300, input_dim=144, activation='relu', kernel_initializer='he_normal'))

        for _ in range(10):
            actor.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
        
        actor.add(Dense(2, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor
    
    def build_turn_critic_model(self):
        critic = Sequential()
        
        critic.add(Dense(144, input_dim=144, activation='relu', kernel_initializer='he_normal'))

        for _ in range(7):
            critic.add(Dense(144, activation='relu', kernel_initializer='he_normal'))
    
        critic.add(Dense(1, activation='linear', kernel_initializer='he_normal'))
        
        critic.summary()

        return critic
        
    
    def bulid_actor_optimizer(self, type):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])
        
        policy = self.actor_model[type].output
        
        action_prob = K.sum(action * policy, axis=1)
        
        cross_entropy = K.log(action_prob) * advantages
        loss = -K.sum(cross_entropy)
        '''
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)
        
        loss = cross_entropy + 0.01 * entropy
        '''

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor_model[type].trainable_weights, [],loss)
        train = K.function([self.actor_model[type].input, action, advantages], [loss], updates=updates)
        return train
    
    def bulid_critic_optimizer(self, type):
        discounted_prediction = K.placeholder(shape=(None,))
        
        value = self.critic_model[type].output
        
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = Adam(lr=self.critic_lr)
        
        updates = optimizer.get_updates(self.critic_model[type].trainable_weights, [], loss)
        train = K.function([self.critic_model[type].input, discounted_prediction], [loss], updates=updates)
            
        return train
    
    def load_model(self):
        for type in range(2):
            self.actor_model[type].load_weights('actor_' + str(type) + '.h5')
            self.critic_model[type].load_weights('critic_' + str(type) + '.h5')

    def save_model(self):
        for type in range(2):
            self.actor_model[type].save_weights('actor_' + str(type) + '.h5')
            self.critic_model[type].save_weights('critic_' + str(type) + '.h5')
        
    def table(self):
        global episode
        time.sleep(5)
        while True:
            print(' ')
            print(episode)
            
            deck = []
            
            generate_deck(deck)
            
            random.shuffle(deck)
                
            ai1_card = np.array([])
            
            ai2_card = np.array([])
            
            for _ in range(5):
                ai1_card = np.append(ai1_card, deck.pop())
                ai2_card = np.append(ai2_card, deck.pop())
            
            winner1, range1 = judge_end(ai1_card, ai2_card)
            winner2, range2 = judge_end(ai2_card, ai1_card)
                
            ai1_state = self.preprocess(winner1, range1, ai1_card, ai2_card, 1)
            ai2_state = self.preprocess(winner2, range2, ai2_card, ai1_card, 1)
        
            self.training_states[0].append(ai1_state)
            self.training_states[1].append(ai2_state)
            
            ai1_card = np.append(ai1_card, deck.pop())
            ai2_card = np.append(ai2_card, deck.pop())
            
            winner1, range1 = judge_end(ai1_card, ai2_card)
            winner2, range2 = judge_end(ai2_card, ai1_card)
                
            ai1_state = self.preprocess(winner1, range1, ai1_card, ai2_card, 2)
            ai2_state = self.preprocess(winner2, range2, ai2_card, ai1_card, 2)
        
            self.training_states[0].append(ai1_state)
            self.training_states[1].append(ai2_state)
            
            ai1_card = np.append(ai1_card, deck.pop())
            ai2_card = np.append(ai2_card, deck.pop())
            
            winner, ai_range = judge_end(ai1_card, ai2_card)
            
            if ai_range[winner][0] <= 2:
                self.train(winner)
                episode += 1
            if episode%1000 == 0:
                self.save_model()
            
    def preprocess(self, winner, range, ai1_card, ai2_card, turn):
        count = 0
        count2 = 0
        
        if range[winner][0] > 2:
            return False
        
        if turn == 1:
            ai_state = np.zeros(98)
            
            if range[0][0] == 0 and range[1][0] == 0:
                for i in ai1_card:
                    for j in ai2_card:
                        if i.value > j.value:
                            count += 1
                ai_state[count] = 1
                
            elif range[0][0] == 0 and range[1][0] == 1:
                for i in ai1_card:
                    if i.value > range[1][1]:
                        count += 1
                ai_state[count + 31] = 1
            
            elif range[0][0] == 0 and range[1][0] == 2:
                for i in ai1_card:
                    if i.value > range[1][1]:
                        count += 1
                ai_state[count + 37] = 1   
            
            elif range[0][0] == 1 and range[1][0] == 0:
                for i in ai2_card:
                    if i.value > range[0][1]:
                        count += 1
                ai_state[count + 43] = 1
                
            elif range[0][0] == 1 and range[1][0] == 1:
                if winner == 0:
                    for i in ai2_card:
                        if i.value > range[0][1]:
                            count2 += 1
                            for j in ai1_card:
                                if j.value > range[0][1]:
                                    if i.value < j.value:
                                        count += 1
                    if count2 == 0:
                        ai_state[49] = 1
                    elif count2 == 1:
                        ai_state[count + 50] = 1
                    elif count2 == 2:
                        ai_state[count + 54] = 1
                    elif count2 ==  3:
                        ai_state[count + 61] = 1
                        
                elif winner == -1:
                    for i in ai1_card:
                        if i.value > range[1][1]:
                            count2 += 1
                            for j in ai2_card:
                                if j.value > range[1][1]:
                                    if i.value < j.value:
                                        count += 1
                    if count2 == 0:
                        ai_state[71] = 1
                    elif count2 == 1:
                        ai_state[count + 72] = 1
                    elif count2 == 2:
                        ai_state[count + 76] = 1
                    elif count2 ==  3:
                        ai_state[count + 83] = 1
                
            elif range[0][0] == 1 and range[1][0] == 2:
                if range[0][1] > range[1][1]:
                    ai_state[93] = 1
                elif range[0][1] <= range[1][1]:
                    for i in ai1_card:
                        if i.value > range[0][1]:
                            if i.value > range[1][1]:
                                count += 1
                    ai_state[count + 94] = 1
            else:
                return False
            
            return np.reshape(ai_state, (1, 98))
                        
                        
        elif turn == 2:
            ai_state = np.zeros(144)
            
            if range[0][0] == 0 and range[1][0] == 0:
                for i in ai1_card:
                    for j in ai2_card:
                        if i.value > j.value:
                            count += 1
                ai_state[count] = 1
                
            elif range[0][0] == 0 and range[1][0] == 1:
                for i in ai1_card:
                    if i.value > range[1][1]:
                        count += 1
                if count == 0:
                    return False
                ai_state[count + 37 - 1] = 1
            
            elif range[0][0] == 1 and range[1][0] == 0:
                for i in ai2_card:
                    if i.value > range[0][1]:
                        count += 1
                if count == 0:
                    return False
                ai_state[count + 43 - 1] = 1
            
            elif range[0][0] == 1 and range[1][0] == 1:
                if winner == 0:
                    for i in ai2_card:
                        if i.value > range[0][1]:
                            count2 += 1
                            for j in ai1_card:
                                if j.value > range[0][1]:
                                    if i.value < j.value:
                                        count += 1
                    if count2 == 0:
                        ai_state[49] = 1
                    elif count2 == 1:
                        ai_state[count + 50] = 1
                    elif count2 == 2:
                        ai_state[count + 55] = 1
                    elif count2 ==  3:
                        ai_state[count + 64] = 1
                    elif count2 ==  4:
                        ai_state[count + 77] = 1
                        
                elif winner == -1:
                    for i in ai1_card:
                        if i.value > range[1][1]:
                            count2 += 1
                            for j in ai2_card:
                                if j.value > range[1][1]:
                                    if i.value < j.value:
                                        count += 1
                    if count2 == 0:
                        ai_state[94] = 1
                    elif count2 == 1:
                        ai_state[count + 95] = 1
                    elif count2 == 2:
                        ai_state[count + 100] = 1
                    elif count2 ==  3:
                        ai_state[count + 109] = 1
                    elif count2 ==  4:
                        ai_state[count + 122] = 1
                    
            elif range[0][0] == 1 and range[1][0] == 2:
                if range[0][1] > range[1][1]:
                    ai_state[139] = 1
                elif range[0][1] <= range[1][1]:
                    for i in ai1_card:
                        if i.value != range[0][1]:
                            if i.value > range[1][1]:
                                count += 1
                    if count == 0:
                        return False
                    ai_state[count + 140 - 1] = 1
            else:
                return False
                        
            return np.reshape(ai_state, (1, 144))

    def train(self, winner):
        action = np.zeros(2).reshape((1,2))
        action[0][0] = 1
        reward = np.ones(1)
        
        for i in range(2):
            if type(self.training_states[winner][i]) == np.ndarray:
                value1 = self.critic_model[i].predict(self.training_states[winner][i])[0]
                if self.actor_model[i].predict(self.training_states[winner][i])[0][0] < 0.99:
                    self.optimizer_actor[i]([self.training_states[winner][i], action, reward - value1])
                    self.optimizer_critic[i]([self.training_states[winner][i], reward])
                print(self.actor_model[i].predict(self.training_states[winner][i])[0])
                print(reward - value1)
            if type(self.training_states[winner + 1][i]) == np.ndarray:
                value2 = self.critic_model[i].predict(self.training_states[winner + 1][i])[0]
                if self.actor_model[i].predict(self.training_states[winner + 1][i])[0][1] < 0.99:
                    self.optimizer_actor[i]([self.training_states[winner + 1][i], action, -reward - value2])
                    self.optimizer_critic[i]([self.training_states[winner + 1][i], -reward])
            
if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.table()
    
    
    
