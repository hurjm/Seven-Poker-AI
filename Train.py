from keras.layers import Dense, Input
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import threading
import random
import time
from collections import deque
import Calc
from Calc import generate_deck, judge
import os
import copy
import h5py

clear = lambda: os.system('cls')

suit_index_dict = {"s": 0, "c": 1, "h": 2, "d": 3}
reverse_suit_index = ("s", "c", "h", "d")
val_string = "AKQJT98765433"
hand_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                 "Straight", "Back Straight", "Mountain", "Flush", "Full House", "Four of a Kind",
                 "Straight Flush", "Back Straight Flush", "Royal Flush")
suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
for num in range(2, 10):
    suit_value_dict[str(num)] = num
    
def check_turn(ai1_board, ai2_board):
    ai1_temp, ai2_temp= [0] * 13, [0] * 13
    ai1_histogram, ai2_histogram, ai1_full, ai2_full = [], [], [], []
    result, result_full = [], []
    j = 0
    
    for card in ai1_board:
        ai1_temp[card.value - 2] += 1
        ai1_full.append((card.value, card.suit_index))
    
    for card in ai2_board:
        ai2_temp[card.value - 2] += 1
        ai2_full.append((card.value, card.suit_index))
        
    for i in ai1_temp:
        if i > 0:
            ai1_histogram.append([i, j])
        j += 1
    
    j = 0
    
    for i in ai2_temp:
        if i > 0:
            ai2_histogram.append([i, j])
        j += 1
        
    ai1_histogram.sort(reverse=True)
    ai2_histogram.sort(reverse=True)
    
    result.append(ai1_histogram)
    result.append(ai2_histogram)
    
    if ai1_histogram == ai2_histogram:
        if ai1_histogram[0][0] == 2:
            if [ai1_histogram[0][1], 3] in ai1_full:
                return 1
            else:
                return 2
        else:
            result_full.append(ai1_full)
            result_full.append(ai2_full)
            return result_full.index(max(result_full)) + 1
        
    return result.index(max(result)) + 1

def reward(table_stake):
    reward = table_stake
    if reward <= 4000:
        return 1
    elif reward <= 8000:
        return 1.1
    elif reward <= 16000:
        return 1.2
    elif reward <= 33000:
        return 1.3
    elif reward <= 64000:
        return 1.4
    elif reward <= 128000:
        return 1.5
    elif reward <= 256000:
        return 1.6
    elif reward <= 512800:
        return 1.7
    elif reward <= 1024000:
        return 1.8
    else:
        return 1.9
            
            
global episode
episode = 1

class A3CAgent:
    def __init__(self):

        self.state_size = 34
        self.action_size = 3

        self.discount_factor = 0.9
        self.actor_lr = 0.000002
        self.critic_lr = 0.000002

        self.threads = 8

        self.actor, self.critic = self.build_actor_model(), self.build_critic_model()
        self.graph = tf.get_default_graph()
        
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)
        
        self.load_model("./save_model/Poker")
    
    def train(self):
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.threads)]
        
        for agent in agents:
            time.sleep(1)
            agent.start()

        while True:
            if episode%2:
                self.save_model("./save_model/Poker")

    def build_actor_model(self):
        actor = Sequential()
        actor.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        
        actor._make_predict_function()
        actor.summary()

        return actor
    
    def build_critic_model(self):
        critic = Sequential()
        critic.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        
        critic._make_predict_function()
        critic.summary()

        return critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)
        return train

    
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction], [loss], updates=updates)
        return train
        

    def load_model(self, name):
        self.actor.load_weights(name + "_actor")
        self.critic.load_weights(name + "_critic")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor")
        self.critic.save_weights(name + "_critic")

    
    def setup_summary(self):
        episode_avg_max_q = tf.Variable(0.)
        total_ai1_reward = tf.Variable(0.)
        total_ai2_reward = tf.Variable(0.)
        episode_ai1_reward = tf.Variable(0.)
        episode_ai2_reward = tf.Variable(0.)

        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Total ai1 Rward/Episode', total_ai1_reward)
        tf.summary.scalar('Total ai2 Rward/Episode', total_ai2_reward)
        tf.summary.scalar('ai1 Rward/Episode', episode_ai1_reward)
        tf.summary.scalar('ai2 Rward/Episode', episode_ai2_reward)

        summary_vars = [episode_avg_max_q, total_ai1_reward, total_ai2_reward, episode_ai1_reward, episode_ai2_reward]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op



class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)
        
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.graph = tf.get_default_graph()
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,self.update_ops, self.summary_writer] = summary_ops
        
        self.ai1_states, self.ai1_actions, self.ai1_rewards = [], [], []
        self.ai2_states, self.ai2_actions, self.ai2_rewards = [], [], []
        
        self.local_actor, self.local_critic = self.build_local_actor_model(), self.build_local_critic_model()

        self.avg_p_max = 0
        self.total_ai1_reward = 0
        self.total_ai2_reward = 0
        self.ai1_reward = 0
        self.ai2_reward = 0
        
        self.ai2_sel = False

    def run(self):
        global episode

        ai1_fund = 10000000
        ai2_fund = 10000000
        time = 0
        ai1_reward = 0
        ai2_reward = 0
        win1 = 0
        win2 = 0

        while True:
            deck = []
            drawn_cards = []
            dict = {}
            j = 0
            h = 0
            l = 1
            
            generate_deck(deck)
            
            for i in deck:
                dict[str(i)] = l
                l += 1
                
            random.shuffle(deck)
            
            for i in range(14):
                drawn_cards.append(deck.pop())
                
            ai1_hands = []
            ai1_board = []
            
            ai2_hands = []
            ai2_board = []
            
            #seed money
            table_stake = 2000
            ai1_fund -= 1000
            ai2_fund -= 1000
            
            stake = 0
            winner = 0
                    
            ai1_state = np.zeros(34)
            ai2_state = np.zeros(34)
            
            pre_ai1_state = np.zeros(34)
            
            ai1_state = np.reshape(ai1_state, [1, 34])
            ai2_state = np.reshape(ai2_state, [1, 34])
            
            pre_ai1_state = np.reshape(ai1_state, [1, 34])
        
            while winner == 0 :
                #turn1
                for i in range(3):
                    ai1_hands.append(drawn_cards.pop())
                    ai2_hands.append(drawn_cards.pop())
                
                for i in ai1_hands:
                    ai1_state[0, j] = dict[str(i)]
                    ai2_state[0, j + 17] = dict[str(i)]
                    j += 1
                
                for i in ai2_hands:
                    ai2_state[0, h] = dict[str(i)]
                    ai1_state[0, h + 17] = dict[str(i)]
                    h += 1
                    
                ai1_action = self.ai1_get_action(ai1_state)
                ai2_action = self.ai2_get_action(ai2_state)
                self.ai1_append_sample(ai1_state, ai1_action, 0)
                
                
                ai1_board.append(ai1_hands.pop(ai1_action))
                ai2_board.append(ai2_hands.pop(ai2_action))
                
                pre_ai1_state = copy.deepcopy(ai1_state)
                pre_ai1_action = ai1_action
                
                #turn2
                for i in range(2):
                    ai1_board.append(drawn_cards.pop())
                    ai2_board.append(drawn_cards.pop())
                    
                for i in ai1_board:
                    ai1_state[0, j] = dict[str(i)]
                    ai2_state[0, j + 17] = dict[str(i)]
                    j += 1
                
                for i in ai2_board:
                    ai2_state[0, h] = dict[str(i)]
                    ai1_state[0, h + 17] = dict[str(i)]
                    h += 1
                    
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.ai1_get_action(ai1_state)
                    self.ai1_append_sample(ai1_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet1(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    ai2_action = self.ai2_get_action(ai2_state)
                    if(ai2_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet1(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    if(ai2_action == 1):
                        ai1_action = self.ai1_get_action(ai1_state)
                        self.ai1_append_sample(ai1_state, ai1_action, 0)
                        if(ai1_action == 2):
                            self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                            winner = 2
                            break
                        ai1_action = 0
                        stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet1(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                        
                        
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = self.ai2_get_action(ai2_state)
                    if(ai2_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet1(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    ai1_action = self.ai1_get_action(ai1_state)
                    self.ai1_append_sample(ai1_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet1(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    if(ai1_action == 1):
                        ai2_action = self.ai2_get_action(ai2_state)
                        if(ai2_action == 2):
                            self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                            winner = 1
                            break
                        ai2_action = 0
                        stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet1(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror1")
                    print(check_turn(ai1_board, ai2_board))
                    break
                       
                #turn3
                stake = 0
                
                ai1_board.append(drawn_cards.pop())
                ai2_board.append(drawn_cards.pop())    
                
                ai1_state[0, j] = dict[str(ai1_board[3])]
                ai2_state[0, j + 17] = dict[str(ai1_board[3])]
                j += 1 
                
                ai2_state[0, h] = dict[str(ai2_board[3])]
                ai1_state[0, h + 17] = dict[str(ai2_board[3])]
                h += 1 
                
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.ai1_get_action(ai1_state)
                    self.ai1_append_sample(ai1_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    ai2_action = self.ai2_get_action(ai2_state)
                    if(ai2_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    if(ai2_action == 1):
                        while(True):
                            ai1_action = self.ai1_get_action(ai1_state)
                            self.ai1_append_sample(ai1_state, ai1_action, 0)
                            if(ai1_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai1_action == 0):
                                break
                            
                            ai2_action = self.ai2_get_action(ai2_state)
                            if(ai2_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai2_action == 0):
                                break
                        
                
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = self.ai2_get_action(ai2_state)
                    if(ai2_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    ai1_action = self.ai1_get_action(ai1_state)
                    self.ai1_append_sample(ai1_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    if(ai1_action == 1):
                        while(True):
                            ai2_action = self.ai2_get_action(ai2_state)
                            if(ai2_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai2_action == 0):
                                break
                            
                            ai1_action = self.ai1_get_action(ai1_state)
                            self.ai1_append_sample(ai1_state, ai1_action, 0)
                            if(ai1_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai1_action == 0):
                                break
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror2")
                    print(check_turn(ai1_board, ai2_board))
                    break
                    
                if(winner != 0):
                    break
                       
                #final
                stake = 0
            
                ai1_hands.append(drawn_cards.pop())
                ai2_hands.append(drawn_cards.pop())    
                
                ai1_state[0, j] = dict[str(ai1_hands[2])]
                ai2_state[0, j + 17] = dict[str(ai1_hands[2])]
                j += 1 
            
                ai2_state[0, h] = dict[str(ai2_hands[2])]
                ai1_state[0, h + 17] = dict[str(ai2_hands[2])]
                h += 1
                
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.ai1_get_action(ai1_state)
                    self.ai1_append_sample(ai1_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet3(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    ai2_action = self.ai2_get_action(ai2_state)
                    if(ai2_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet3(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    if(ai2_action == 1):
                        while(True):
                            ai1_action = self.ai1_get_action(ai1_state)
                            self.ai1_append_sample(ai1_state, ai1_action, 0)
                            if(ai1_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai1_action == 0):
                                break
                            
                            ai2_action = self.ai2_get_action(ai2_state)
                            if(ai2_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai2_action == 0):
                                break
                        
                
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = self.ai2_get_action(ai2_state)
                    if(ai2_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet3(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    ai1_action = self.ai1_get_action(ai1_state)
                    self.ai1_append_sample(ai1_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet3(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                    
                    
                    if(ai1_action == 1):
                        while(True):
                            ai2_action = self.ai2_get_action(ai2_state)
                            if(ai2_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet3(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai2_action == 0):
                                break
                            
                            ai1_action = self.ai1_get_action(ai1_state)
                            self.ai1_append_sample(ai1_state, ai1_action, 0)
                            if(ai1_action == 2):
                                self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund = bet3(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, ai1_fund, ai2_fund)
                            if(ai1_action == 0):
                                break
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror3")
                    print(check_turn(ai1_board, ai2_board))
                    break
                
                if(winner != 0):
                    break
                
                winner = judge(ai1_hands + ai1_board, ai2_hands + ai2_board)
         
                if winner == 1:
                    ai1_fund += table_stake
                    self.ai1_rewards[len(self.ai1_rewards) - 1] = reward(table_stake)           
                    break
                elif winner == 2:
                    ai2_fund += table_stake
                    self.ai1_rewards[len(self.ai1_rewards) - 1] = -reward(table_stake)
                    break
                
            episode += 1
            
            
            test = np.ones(34)
            
            test = np.reshape(test, [1, 34])
            
            with self.graph.as_default():
                static = self.local_actor.predict(test)
                
            print(static)
            
            self.ai1_train_model()
            self.update_local_model()
            
            if episode>=1000000:
                break
            
            
    def discounted_prediction(self, rewards):
        discounted_prediction = np.zeros_like(rewards, dtype=float)
        
        running_add = 0
            
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
            
        return discounted_prediction

    
    def ai1_train_model(self):
        discounted_prediction = self.discounted_prediction(self.ai1_rewards)

        ai1_states = np.zeros((len(self.ai1_states), 34))
        for i in range(len(self.ai1_states)):
            ai1_states[i] = self.ai1_states[i]
        
        with self.graph.as_default():
            values = self.critic.predict(ai1_states)
        
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values
        
        #print(ai1_states)
        #print(self.ai1_actions)
        #print(self.ai1_rewards)
        
        self.optimizer[0]([ai1_states, self.ai1_actions, advantages])
        self.optimizer[1]([ai1_states, discounted_prediction])
        self.ai1_states, self.ai1_actions, self.ai1_rewards = [], [], []
    
    def build_local_actor_model(self):
        '''
        input = Input(shape=self.state_size)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(input)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)
        dnn = Dense(64, activation='relu', kernel_initializer='zero')(dnn)

        policy = Dense(self.action_size, activation='softmax', kernel_initializer='zero')(dnn)
        value = Dense(1, activation='linear', kernel_initializer='zero')(dnn)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)
        
        local_actor = Sequential()
        local_actor.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='zero'))
        
        local_critic = Sequential()
        local_critic.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='zero'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='zero'))
        local_critic.add(Dense(1, activation='linear', kernel_initializer='zero'))

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic
        '''
        local_actor = Sequential()
        local_actor.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))

        local_actor._make_predict_function()
        local_actor.set_weights(self.actor.get_weights())
        local_actor.summary()

        return local_actor
    
    def build_local_critic_model(self):
        local_critic = Sequential()
        local_critic.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        local_critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))

        local_critic._make_predict_function()
        local_critic.set_weights(self.critic.get_weights())
        local_critic.summary()

        return local_critic

    
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def ai1_get_action(self, history):
        with self.graph.as_default():
            policy = self.local_actor.predict(history)[0]
            
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    def ai2_get_action(self, history):
        with self.graph.as_default():
            policy = self.local_actor.predict(history)[0]
            
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    def ai1_append_sample(self, history, action, reward):
        state = copy.deepcopy(history)
        self.ai1_states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.ai1_actions.append(act)
        self.ai1_rewards.append(reward)
        
def bet1(action, stake, table_stake, id, ai1_state, ai2_state, ai1_fund, ai2_fund):
    if action == 0:
        #Check
        if stake == 0:
            if id == 1:
                ai1_state[0, 8] += 1
                ai2_state[0, 25] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 8] += 1
                ai1_state[0, 25] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        #Call    
        else:
            if id == 1:
                ai1_state[0, 9] += 1
                ai2_state[0, 26] += 1
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 9] += 1
                ai1_state[0, 26] += 1
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
    #Raise
    elif action == 1:
        if stake == 0:
            if id == 1:
                ai1_state[0, 10] += 1
                ai2_state[0, 27] += 1
                stake = table_stake/2
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 10] += 1
                ai1_state[0, 27] += 1
                stake = table_stake/2
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        else:
            if id == 1:
                ai1_state[0, 10] += 1
                ai2_state[0, 27] += 1
                table_stake += stake
                ai1_fund -= stake
                stake = table_stake/2
                ai1_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 10] += 1
                ai1_state[0, 27] += 1
                table_stake += stake
                ai2_fund -= stake
                stake = table_stake/2
                ai2_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund

def bet2(action, stake, table_stake, id, ai1_state, ai2_state, ai1_fund, ai2_fund):
    if action == 0:
        #Check
        if stake == 0:
            if id == 1:
                ai1_state[0, 11] += 1
                ai2_state[0, 28] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 11] += 1
                ai1_state[0, 28] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        #Call    
        else:
            if id == 1:
                ai1_state[0, 12] += 1
                ai2_state[0, 29] += 1
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 12] += 1
                ai1_state[0, 29] += 1
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
    #Raise
    elif action == 1:
        if stake == 0:
            if id == 1:
                ai1_state[0, 13] += 1
                ai2_state[0, 30] += 1
                stake = table_stake/2
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 13] += 1
                ai1_state[0, 30] += 1
                stake = table_stake/2
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        else:
            if id == 1:
                ai1_state[0, 13] += 1
                ai2_state[0, 30] += 1
                table_stake += stake
                ai1_fund -= stake
                stake = table_stake/2
                ai1_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 13] += 1
                ai1_state[0, 30] += 1
                table_stake += stake
                ai2_fund -= stake
                stake = table_stake/2
                ai2_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        
        
def bet3(action, stake, table_stake, id, ai1_state, ai2_state, ai1_fund, ai2_fund):
    if action == 0:
        #Check
        if stake == 0:
            if id == 1:
                ai1_state[0, 14] += 1
                ai2_state[0, 31] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 14] += 1
                ai1_state[0, 31] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        #Call    
        else:
            if id == 1:
                ai1_state[0, 15] += 1
                ai2_state[0, 32] += 1
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 15] += 1
                ai1_state[0, 32] += 1
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
    #Raise
    elif action == 1:
        if stake == 0:
            if id == 1:
                ai1_state[0, 16] += 1
                ai2_state[0, 33] += 1
                stake = table_stake/2
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 16] += 1
                ai1_state[0, 33] += 1
                stake = table_stake/2
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        else:
            if id == 1:
                ai1_state[0, 16] += 1
                ai2_state[0, 33] += 1
                table_stake += stake
                ai1_fund -= stake
                stake = table_stake/2
                ai1_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 16] += 1
                ai1_state[0, 33] += 1
                table_stake += stake
                ai2_fund -= stake
                stake = table_stake/2
                ai2_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        
if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()
    
    
    
    
    '''

            
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob) * advantages
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=[None,])

        value = self.critic.output

        
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction], [], updates=updates)
        return train
    '''
    