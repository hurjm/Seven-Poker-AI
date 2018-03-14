from keras.layers import Dense
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.models import Sequential
import tensorflow as tf
import numpy as np
from Calc import *
import os
import copy

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
    
episode = 1
final = 0
eval_episode = 1
best_fund = 0
challenger_fund = 0
first = True
switch = False
  
class A3CAgent:
    def __init__(self):
        self.action_size = 3
        self.discount_factor = 0.9
        self.actor_lr = 0.00001
        self.critic_lr = 0.00001
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        
        self.training_states, self.training_actions, self.training_rewards = [], [], []
        
        self.best_actor, self.best_critic = self.build_actor_model(), self.build_critic_model()
        
        self.challenger_actor, self.challenger_critic = self.build_actor_model(), self.build_critic_model()
        
        self.optimizer = [self.bulid_actor_optimizer(), self.bulid_critic_optimizer()]
        
        #self.load_model('challenger')
        #self.load_model('best')
    
    def train(self):
        global episode 
        global final
        global eval_episode
        global best_fund
        global challenger_fund
        
        time.sleep(5)
        while final < 10:
            self.save_model("best")
            self.table(False)
            print('eval')
            eval_episode = 1
            self.table(True)
            
            if challenger_fund >= 20000:
                print('update')
                self.update_model()
                final = 0
            else:
                final += 1
                
            challenger_fund, best_fund = 0, 0
            first = False

    def build_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(254, input_dim=254, activation='selu', kernel_initializer='lecun_normal'))
        for _ in range(20):
            actor.add(Dense(350, activation='selu', kernel_initializer='lecun_normal'))
        
        actor.add(Dense(3, activation='softmax', kernel_initializer='lecun_normal'))
        
        actor.summary()

        return actor
    
    def build_critic_model(self):
        critic = Sequential()
        
        critic.add(Dense(254, input_dim=254, activation='selu', kernel_initializer='lecun_normal'))
        for _ in range(20):
            critic.add(Dense(350, activation='selu', kernel_initializer='lecun_normal'))
        
        critic.add(Dense(1, activation='linear', kernel_initializer='lecun_normal'))
        
        critic.summary()

        return critic
    
    def bulid_actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        
        policy = self.challenger_actor.output
        
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)
        
        loss = cross_entropy + 0.01 * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.challenger_actor.trainable_weights, [],loss)
        train = K.function([self.challenger_actor.input, action, advantages], [loss], updates=updates)
        return train
    
    def bulid_critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))
        
        value = self.challenger_critic.output
        
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = Adam(lr=self.critic_lr)
        
        updates = optimizer.get_updates(self.challenger_critic.trainable_weights, [],loss)
        train = K.function([self.challenger_critic.input, discounted_prediction], [loss], updates=updates)
            
        return train

    def load_model(self, name):
        self.challenger_actor.load_weights(name + "_actor.h5")
        self.challenger_critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.challenger_actor.save_weights(str(name) + "_actor.h5")
        self.challenger_critic.save_weights(str(name) + "_critic.h5")
        
    def discounted_prediction(self, rewards):
        global switch
        discounted_prediction = np.zeros_like(rewards, dtype=float)
        
        running_add = 0
            
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
            
        #if switch == True:
        #    discounted_prediction[len(rewards)-1] = rewards[len(rewards)-1] * 2
            
            
        return discounted_prediction
        
    def train_model(self):
        discounted_prediction = self.discounted_prediction(self.training_rewards)
        
        training_states = np.zeros((len(self.training_states), 254))
        for i in range(len(self.training_states)):
            training_states[i] = self.training_states[i]
            
        values = self.challenger_critic.predict(training_states)
        
        values = np.reshape(values, len(values))
        advantages = discounted_prediction - values

        self.optimizer[0]([training_states, self.training_actions, advantages])
        self.optimizer[1]([training_states, discounted_prediction])
            
    
    def update_model(self):
        self.best_actor.set_weights(self.challenger_actor.get_weights())
        self.best_critic.set_weights(self.challenger_critic.get_weights())
        
    def append_sample(self, card_state, betting_state, action, reward):
        state1 = copy.deepcopy(card_state)
        state2 = copy.deepcopy(betting_state)
        state = np.append(state1, state2)
        state = np.reshape(state, [1, 254])
        #state = np.reshape(card_state, [1, 208])
        self.training_states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.training_actions.append(act)
        self.training_rewards.append(reward)
        
    def choice(self, card_state, betting_state, dict, ai1_hands, ai1_board, ai2_board, hidden, eval, id):
        calc_deck = []
        result = np.zeros(3)
        number = 0
        md = [0, 1, 2]
        
        state = np.append(card_state, betting_state)
        state = np.reshape(state, [1, 254])
        #state = np.reshape(card_state, [1, 208])
        
        
        generate_deck(calc_deck)
        #print("start!")
        
        if eval == False:
            policy = self.best_actor.predict(state)[0]
            #print(state)
            if id == 1:
                print(policy)
            if first == True:
                return np.random.choice(self.action_size, 1, p=policy)[0]
            elif first == False:
                return np.random.choice(self.action_size, 1, p=policy)[0]
        
        elif eval == True:
            for i in ai1_hands + ai1_board + ai2_board:
                calc_deck.remove(i)
                
            for i in list(itertools.combinations(calc_deck, 2)):
                case = []
                for j in i:
                    case.append(dict[str(j)])
            
                state[0, 104 + case[0]] = 1
                state[0, 104 + case[0]] = 1
                
                if hidden == 0:
                    if id == 1:
                        result += self.best_actor.predict(state)[0]
                    elif id == 2:
                        result += self.challenger_actor.predict(state)[0]
                    number += 1
                    state[0, 104 + case[0]] = 0
                    state[0, 104 + case[0]] = 0
                    
                elif hidden == 1:
                    for k in calc_deck:
                        hid = dict[str(k)]
                        if hid == case[0] or hid == case[1]:
                            continue
                        
                        state[0, 104 + hid] = 1
                        
                        if id == 1:
                            result += self.best_actor.predict(state)[0]
                        elif id == 2:
                            result += self.challenger_actor.predict(state)[0]
                        number += 1
                        state[0, 104 + hid] = 0
                
            for i in range(3):
                result[i] /= number
            result /= result.sum()
            
            print(card_state)
            print(betting_state)
            print(result)
            print(' ')
            
            mx = np.where(result == max(result))
            mn = np.where(result == min(result))
            if result[mx] == 1:
                #print("er!")
                return mx[0][0]
            md.remove(mx[0])
            md.remove(mn[0])
            #print(md)
            #print("end!")
            
            
            if result[mx[0][0]] - result[md[0]] >= 0.2:
                return mx[0][0]
            #elif result[mx] - result[md[0]] < 0.2:
            #    return emotion
            else:
                return np.random.choice(self.action_size, 1, p=result)[0]
        
    def reward(self, table_stake):
        reward = table_stake
        if reward <= 4000:
            return 1
        elif reward <= 8000:
            return 1.01
        elif reward <= 16000:
            return 1.02
        elif reward <= 32000:
            return 1.03
        elif reward <= 64000:
            return 1.04
        elif reward <= 128000:
            return 1.06
        elif reward <= 256000:
            return 1.07
        elif reward <= 512800:
            return 1.08
        elif reward <= 1024000:
            return 1.09
        else:
            return 1.1
    
    def table(self, eval):
        global episode 
        global eval_episode
        global final
        global best_fund
        global challenger_fund
        global switch
        
        best_fund = 0
        challenger_fund = 0
        
        while True:
            print(episode, eval_episode, best_fund, challenger_fund, final)
            self.training_states, self.training_actions, self.training_rewards = [], [], []
            deck = []
            dict = {}
            l = 0
            r = 5
            e = 5
            switch = False
            
            generate_deck(deck)
            
            for i in deck:
                dict[str(i)] = l
                l += 1
                
            random.shuffle(deck)
                
            ai1_hands = []
            ai1_board = []
            
            ai2_hands = []
            ai2_board = []
            
            #seed money
            table_stake = 2000
            best_fund -= 1000
            challenger_fund -= 1000
            
            stake = 0
            winner = 0
                    
            ai1_card_state = np.zeros(208)
            ai2_card_state = np.zeros(208)
            
            ai1_betting_state = np.zeros(46)
            ai2_betting_state = np.zeros(46)
            
            ai1_card_state = np.reshape(ai1_card_state, [4, 52])
            ai2_card_state = np.reshape(ai2_card_state, [4, 52])
            
            ai1_betting_state = np.reshape(ai1_betting_state, [2, 23])
            ai2_betting_state = np.reshape(ai2_betting_state, [2, 23])
            
            while True :
                #turn1
                for _ in range(3):
                    ai1_hands.append(deck.pop())
                    ai2_hands.append(deck.pop())
                
                for i in ai1_hands:
                    ai1_card_state[0, dict[str(i)]] = 1
                    ai2_card_state[2, dict[str(i)]] = 1
                    
                for i in ai2_hands:
                    ai2_card_state[0, dict[str(i)]] = 1
                    ai1_card_state[2, dict[str(i)]] = 1
                
                ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                
                ai1_board.append(ai1_hands.pop(ai1_action))
                ai2_board.append(ai2_hands.pop(ai2_action))
                
                for i in ai1_board:
                    ai1_card_state[0, dict[str(i)]] = 0
                    ai2_card_state[2, dict[str(i)]] = 0
                    
                for i in ai2_board:
                    ai2_card_state[0, dict[str(i)]] = 0
                    ai1_card_state[2, dict[str(i)]] = 0
                
                for i in ai1_hands:
                    ai1_card_state[0, dict[str(i)]] = 1
                    ai2_card_state[2, dict[str(i)]] = 1
                    
                for i in ai2_hands:
                    ai2_card_state[0, dict[str(i)]] = 1
                    ai1_card_state[2, dict[str(i)]] = 1
                
                print('turn2')
                
                #turn2
                for _ in range(2):
                    ai1_board.append(deck.pop())
                    ai2_board.append(deck.pop())
                
                for i in ai1_board:
                    ai1_card_state[1, dict[str(i)]] = 1
                    ai2_card_state[3, dict[str(i)]] = 1
                
                for i in ai2_board:
                    ai2_card_state[1, dict[str(i)]] = 1
                    ai1_card_state[3, dict[str(i)]] = 1
                
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                    #print(ai1_action)
                    self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund = bet1(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund)
                    
                    
                    ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                    if(ai2_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund= bet1(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund)
                    
                    
                    if(ai2_action == 1):
                        ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                        if ai1_action == 1:
                            ai1_action = 0
                        self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                        if(ai1_action == 2):
                            self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                            winner = 2
                            break
                        stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund = bet1(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund)
                        
                        
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                    if(ai2_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund= bet1(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund)
                    
                    
                    ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                    self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund = bet1(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund)
                    
                    
                    if(ai1_action == 1):
                        ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                        if(ai2_action == 2):
                            self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                            winner = 1
                            break
                        ai2_action = 0
                        stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund= bet1(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund)
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror1")
                    print(check_turn(ai1_board, ai2_board))
                    break
                
                print('turn3')
                #turn3
                stake = 0
                
                ai1_temp = deck.pop()
                ai2_temp = deck.pop()
                
                ai1_board.append(ai1_temp)
                ai2_board.append(ai2_temp)
                
                ai1_card_state[1, dict[str(ai1_temp)]] = 1
                ai2_card_state[3, dict[str(ai1_temp)]] = 1
                
                ai2_card_state[1, dict[str(ai2_temp)]] = 1
                ai1_card_state[3, dict[str(ai2_temp)]] = 1
                
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                    self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet2(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                    
                    
                    ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                    if(ai2_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet2(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                    
                    if(ai2_action == 1):
                        while(True):
                            ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                            if(table_stake>1000000 and ai1_action == 1):
                                ai1_action = 0
                            self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                            if(ai1_action == 2):
                                self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet2(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                            if(ai1_action == 0):
                                break
                            
                            ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                            if(table_stake>1000000 and ai2_action == 1):
                                ai2_action = 0
                            if(ai2_action == 2):
                                self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet2(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                            if(ai2_action == 0):
                                break
                        
                
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                    if(ai2_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet2(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                    
                    
                    ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                    self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                    if(ai1_action == 2):
                        self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet2(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                    
                    if(ai1_action == 1):
                        while(True):
                            ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2)
                            if(table_stake>1000000 and ai2_action == 1):
                                ai2_action = 0
                            if(ai2_action == 2):
                                self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet2(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                            if(ai2_action == 0):
                                break
                            
                            ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                            if(table_stake>1000000 and ai1_action == 1):
                                ai1_action = 0
                            self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                            if(ai1_action == 2):
                                self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet2(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                            if(ai1_action == 0):
                                break
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror2")
                    print(check_turn(ai1_board, ai2_board))
                    break
                    
                if(winner != 0):
                    break
                
                print('final')
                #final
                stake = 0
                r = 15
                e = 15
                
                ai1_temp = deck.pop()
                ai2_temp = deck.pop()
                
                ai1_hands.append(ai1_temp)
                ai2_hands.append(ai2_temp)
                
                ai1_card_state[0, dict[str(ai1_temp)]] = 1
                ai2_card_state[2, dict[str(ai1_temp)]] = 1
                
                ai2_card_state[0, dict[str(ai2_temp)]] = 1
                ai1_card_state[2, dict[str(ai2_temp)]] = 1
                
                if(table_stake < 1000000):
                    if(check_turn(ai1_board, ai2_board) == 1):
                        ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                        self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                        if(ai1_action == 2):
                            self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                            winner = 2
                            break
                        stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet3(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                        
                        
                        ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2)
                        if(ai2_action == 2):
                            self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                            winner = 1
                            break
                        stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet3(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                        
                        
                        if(ai2_action == 1):
                            while(True):
                                ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                                if(table_stake>1000000 and ai1_action == 1):
                                    ai1_action = 0
                                self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                                if(ai1_action == 2):
                                    self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                    winner = 2
                                    break
                                stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet3(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                                if(ai1_action == 0):
                                    break
                                
                                ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2)
                                if(table_stake>1000000 and ai2_action == 1):
                                    ai2_action = 0
                                if(ai2_action == 2):
                                    self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                    winner = 1
                                    break
                                stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet3(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                                if(ai2_action == 0):
                                    break
                            
                    
                    elif(check_turn(ai1_board, ai2_board) == 2):
                        ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2)
                        if(ai2_action == 2):
                            self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                            winner = 1
                            break
                        stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet3(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                        
                        
                        ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                        self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                        if(ai1_action == 2):
                            self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                            winner = 2
                            break
                        stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet3(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                        
                        
                        if(ai1_action == 1):
                            while(True):
                                ai2_action = self.choice(ai2_card_state, ai2_betting_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2)
                                if(table_stake>1000000 and ai2_action == 1):
                                    ai2_action = 0
                                if(ai2_action == 2):
                                    self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                    winner = 1
                                    break
                                stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e = bet3(ai2_action, stake, table_stake, 2, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, e)
                                if(ai2_action == 0):
                                    break
                                
                                ai1_action = self.choice(ai1_card_state, ai1_betting_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                                if(table_stake>1000000 and ai1_action == 1):
                                    ai1_action = 0
                                self.append_sample(ai1_card_state, ai1_betting_state, ai1_action, 0)
                                if(ai1_action == 2):
                                    self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                    winner = 2
                                    break
                                stake, table_stake, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r = bet3(ai1_action, stake, table_stake, 1, ai1_betting_state, ai2_betting_state, best_fund, challenger_fund, r)
                                if(ai1_action == 0):
                                    break
                            
                    else:
                        print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror3")
                        print(check_turn(ai1_board, ai2_board))
                        break
                
                if(winner != 0):
                    if judge(ai1_hands + ai1_board, ai2_hands + ai2_board) == 1:
                        switch = True
                    break
                
                winner = judge(ai1_hands + ai1_board, ai2_hands + ai2_board)
                break
            
            if winner == 1:
                best_fund += table_stake
                if ai1_action == 0:
                    self.training_actions[-1][0] = 0
                    self.training_actions[-1][1] = 1 
                self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)     
            elif winner == 2:
                challenger_fund += table_stake
                if ai1_action == 0 and check_turn(ai1_board, ai2_board) == 2:
                    switch = True
                self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                
            
            if episode%10000 == 0:
                self.save_model(episode)
                
            if eval == True:
                eval_episode += 1
                if eval_episode == 100:
                    break
                
            elif eval == False:
                episode += 1
                self.train_model()
                self.update_model()
                
                if first == True:
                    if episode%10000000 == 0:
                        break
                elif first == False:
                    if episode%10000 ==0:
                        break
            
if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()
    
    
    
