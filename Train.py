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
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        K.set_learning_phase(1)
        self.sess.run(tf.global_variables_initializer())
        
        self.ai1_training_states, self.training_actions, self.training_rewards = [], [[], [], [], []], []
        self.ai2_training_states, self.training_actions, self.training_rewards = [], [[], [], [], []], []
        self.training_states = [[], []]
        
        self.actor_model = [self.build_flop_turn_actor_model(), self.build_flop_turn_actor_model()]#, self.build_river_actor_model()]
        self.critic_model = [self.build_flop_turn_critic_model(), self.build_flop_turn_critic_model()]#, self.build_river_critic_model()]
        
        self.optimizer_actor = [self.bulid_actor_optimizer(0), self.bulid_actor_optimizer(1)]#, self.bulid_actor_optimizer(2)]
        self.optimizer_critic = [self.bulid_critic_optimizer(0), self.bulid_critic_optimizer(1)]#, self.bulid_critic_optimizer(2)]
        
        #self.load_model()
 

    def build_flop_turn_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(2024, input_dim=2024, activation='relu', kernel_initializer='he_normal'))
        #actor.add(Dropout(0.5))
        for _ in range(3):
            actor.add(Dense(1350, activation='relu', kernel_initializer='he_normal'))
            #actor.add(Dropout(0.5))
        
        actor.add(Dense(1350, activation='relu', kernel_initializer='he_normal'))
        actor.add(Dense(2, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor

    def build_flop_turn_critic_model(self):
        critic = Sequential()
        
        critic.add(Dense(2024, input_dim=2024, activation='relu', kernel_initializer='he_normal'))
        #critic.add(Dropout(0.5))
        for _ in range(3):
            critic.add(Dense(1350, activation='relu', kernel_initializer='he_normal'))
            #critic.add(Dropout(0.5))
    
        critic.add(Dense(1350, activation='relu', kernel_initializer='he_normal'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_normal'))
        
        critic.summary()

        return critic
        
    def build_river_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(1369, input_dim=1369, activation='relu', kernel_initializer='he_normal'))
        #actor.add(Dropout(0.5))
        for _ in range(3):
            actor.add(Dense(912, activation='relu', kernel_initializer='he_normal'))
            #actor.add(Dropout(0.5))
        
        actor.add(Dense(912, activation='relu', kernel_initializer='he_normal'))
        actor.add(Dense(2, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor
    
    def build_river_critic_model(self):
        critic = Sequential()
        
        critic.add(Dense(1296, input_dim=1296, activation='relu', kernel_initializer='he_normal'))
        #critic.add(Dropout(0.5))
        for _ in range(3):
            critic.add(Dense(864, activation='relu', kernel_initializer='he_normal'))
            #critic.add(Dropout(0.5))
    
        critic.add(Dense(864, activation='relu', kernel_initializer='he_normal'))
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
                
            ai1_hands = np.array([])
            ai1_board = np.array([])
            
            ai2_hands = np.array([])
            ai2_board = np.array([])
            
            for _ in range(3):
                ai1_hands = np.append(ai1_hands, deck.pop())
                ai2_hands = np.append(ai2_hands, deck.pop())
            
            for _ in range(2):
                ai1_board = np.append(ai1_board, deck.pop())
                ai2_board = np.append(ai2_board, deck.pop())
                
            ai1_state, ai2_state = self.preprocess(np.append(ai1_hands, ai1_board), np.append(ai2_hands, ai2_board))
            
            self.training_states[0].append(ai1_state)
            self.training_states[1].append(ai2_state)
            
            ai1_board = np.append(ai1_board, deck.pop())
            ai2_board = np.append(ai2_board, deck.pop())
            
            ai1_state, ai2_state = self.preprocess(np.append(ai1_hands, ai1_board), np.append(ai2_hands, ai2_board))
            
            self.training_states[0].append(ai1_state)
            self.training_states[1].append(ai2_state)
            
            ai1_hands = np.append(ai1_hands, deck.pop())
            ai2_hands = np.append(ai2_hands, deck.pop())
            
            #ai1_state, ai2_state, 
            ai_range = self.preprocess_final(np.append(ai1_hands, ai1_board), np.append(ai2_hands, ai2_board))
            
            #self.training_states[0].append(ai1_state)
            #self.training_states[1].append(ai2_state)
            
            winner = judge_end(np.append(ai1_hands, ai1_board), np.append(ai2_hands, ai2_board))
            self.train(winner, ai_range)
            episode += 1
            if episode%100 == 0:
                self.save_model()
            
    def preprocess(self, ai1_card, ai2_card):
        temp = np.array([ai1_card, ai2_card])
        result_list = []
        for card in temp:
            val_list = np.zeros(45, dtype=int).reshape(15, 3)
            suit_list = np.zeros(4, dtype=int)
            straight_list = np.array([])
            count = 0
            for i in range(len(card)):
                for j in reversed(range(3)):
                    if val_list[card[i].value][j] == 0 and j == 0:
                        val_list[card[i].value][j] = 1
                    elif val_list[card[i].value][j] == 1:
                        val_list[card[i].value][j] = 0
                        if j == 2:
                            val_list[13][2] = 1
                        else:
                            val_list[card[i].value][j + 1] = 1
                        break
                    
                suit_list[card[i].suit_index] += 1
                straight_list = np.append(straight_list, card[i].value)
            
            straight_list = np.sort(straight_list)
            
            for i in range(len(straight_list) - 1):
                if straight_list[i] == straight_list[i + 1] - 1:
                    count += 1
                else:
                    count = 0 
            #straight
            if count == 3:
                val_list[13][0] = 1
            elif count >= 4:
                val_list[13][1] = 1
            
            #flush
            if np.amax(suit_list) == 4:
                val_list[14][0] = 1
            elif np.amax(suit_list) >= 5:
                val_list[14][1] = 1
            
            result_list.append(val_list)
            
        return result_list[0], result_list[1]
                     
    def preprocess_final(self, ai1_card, ai2_card):
        #ai1_state = np.zeros(1369, dtype=int).reshape(37, 37)
        #ai2_state = np.zeros(1369, dtype=int).reshape(37, 37)
        
        ai1_range = np.array([], dtype=int)
        ai2_range = np.array([], dtype=int)
        
        ai1_range = np.append(ai1_range, judge_state(ai1_card))
        ai2_range = np.append(ai2_range, judge_state(ai2_card))
        
        #ai1_state[ai2_range[0]][ai1_range[0]] = 1
        #ai1_state[ai1_range[0]][ai2_range[0]] = 1
        
        return [ai1_range, ai2_range]

    def train(self, winner, ai_range):
        ai_state = np.array([[np.zeros(2025, dtype=int).reshape(15, 3, 15, 3), np.zeros(2025, dtype=int).reshape(15, 3, 15, 3)], [np.zeros(2025, dtype=int).reshape(15, 3, 15, 3), np.zeros(2025, dtype=int).reshape(15, 3, 15, 3)]])
        
        if ai_range[winner][0] >= 0 and ai_range[winner][0] <= 5:
            for i in range(2):
                if self.training_states[winner][i][ai_range[winner][1]][0] == 1:
                    for j in range(15):
                        if np.amax(self.training_states[winner + 1][i][j]) == 1:
                            ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][0] = 1
                            ai_state[winner + 1][i][ai_range[winner][1]][0][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                            self.optimize(ai_state, winner)
                            ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][0] = 0
                            ai_state[winner + 1][i][ai_range[winner][1]][0][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
        
        elif ai_range[winner][0] >= 6 and ai_range[winner][0] <= 18:
            for i in range(2):
                for k in range(2):
                    if self.training_states[winner][i][ai_range[winner][1]][k] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 1
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 0
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
                            
        elif ai_range[winner][0] >= 19 and ai_range[winner][0] <= 30:
            for i in range(2):
                for h in [ai_range[winner][1], ai_range[winner][2]]:
                    for k in range(2):
                        if self.training_states[winner][i][h][k] == 1:
                            for j in range(15):
                                if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                    ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][h][k] = 1
                                    ai_state[winner + 1][i][h][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                    self.optimize(ai_state, winner)
                                    ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][h][k] = 0
                                    ai_state[winner + 1][i][h][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
                  
        elif ai_range[winner][0] == 31:
            for i in range(2):
                for k in range(3):
                    if self.training_states[winner][i][ai_range[winner][1]][k] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 1
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 0
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
            
        elif ai_range[winner][0] == 32:
            for i in range(2):
                for k in range(2):
                    if self.training_states[winner][i][13][k] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][13][k] = 1
                                ai_state[winner + 1][i][13][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][13][k] = 0
                                ai_state[winner + 1][i][13][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
                         
                        
        elif ai_range[winner][0] == 33:
            for i in range(2):
                for k in range(2):
                    if self.training_states[winner][i][14][k] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][14][k] = 1
                                ai_state[winner + 1][i][14][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][14][k] = 0
                                ai_state[winner + 1][i][14][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
            
        elif ai_range[winner][0] == 34:
            for i in range(2):
                for h in [ai_range[winner][1], ai_range[winner][2]]:
                    for k in range(3):
                        if self.training_states[winner][i][h][k] == 1:
                            for j in range(15):
                                if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                    ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][h][k] = 1
                                    ai_state[winner + 1][i][h][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                    self.optimize(ai_state, winner)
                                    ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][h][k] = 0
                                    ai_state[winner + 1][i][h][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0

        elif ai_range[winner][0] == 35:
            for i in range(2):
                for k in range(3):
                    if self.training_states[winner][i][ai_range[winner][1]][k] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 1
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 0
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
                                
                    elif self.training_states[winner][i][13][2] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 1
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][ai_range[winner][1]][k] = 0
                                ai_state[winner + 1][i][ai_range[winner][1]][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
        
        elif ai_range[winner][0] == 36:
            for i in range(2):
                for k in range(2):
                    if self.training_states[winner][i][13][k] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][13][k] = 1
                                ai_state[winner + 1][i][13][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][13][k] = 0
                                ai_state[winner + 1][i][13][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
                                
                    if self.training_states[winner][i][14][k] == 1:
                        for j in range(15):
                            if np.amax(self.training_states[winner + 1][i][j]) == 1:
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][14][k] = 1
                                ai_state[winner + 1][i][14][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 1
                                self.optimize(ai_state, winner)
                                ai_state[winner][i][j][np.argmax(self.training_states[winner + 1][i][j])][14][k] = 0
                                ai_state[winner + 1][i][14][k][j][np.argmax(self.training_states[winner + 1][i][j])] = 0
                                
        else:
            print('erorrrrrrrrrrrr')
            print(winner)
            print(ai_range)
    def optimize(self, ai_state, winner):
        temp_state = np.array([[np.zeros(2024), np.zeros(2024)], [np.zeros(2024), np.zeros(2024)]])
        training_state = np.array([[np.zeros(2024).reshape(1, 2024), np.zeros(2024).reshape(1, 2024)], [np.zeros(2024).reshape(1, 2024), np.zeros(2024).reshape(1, 2024)]])

        for i in range(2):
            temp_state[winner][i] = np.copy((ai_state[winner][i]).flatten()[:-1])
            temp_state[winner + 1][i] = np.copy((ai_state[winner + 1][i]).flatten()[:-1])
            training_state[winner][i] = np.copy(temp_state[winner][i]).reshape((1, 2024))
            training_state[winner + 1][i] = np.copy(temp_state[winner + 1][i]).reshape((1, 2024))
        action = np.zeros(2).reshape((1,2))
        action[0][0] = 1
        reward = np.ones(1)

        for i in range(2):
            value1 = self.critic_model[i].predict(training_state[winner][i])[0]
            value2 = self.critic_model[i].predict(training_state[winner + 1][i])[0]
            if self.actor_model[i].predict(training_state[winner][i])[0][0] < 0.99:
                self.optimizer_actor[i]([training_state[winner][i], action, reward - value1])
                self.optimizer_critic[i]([training_state[winner][i], reward])
            if self.actor_model[i].predict(training_state[winner + 1][i])[0][1] < 0.99:
                self.optimizer_actor[i]([training_state[winner + 1][i], action, -reward - value2])
                self.optimizer_critic[i]([training_state[winner + 1][i], -reward])
            print(self.actor_model[i].predict(training_state[winner][i])[0])
            print(reward - value1, -reward - value2)
            
if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.table()
    
    
    
