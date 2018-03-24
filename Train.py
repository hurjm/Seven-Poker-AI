from keras.layers import Dense, AlphaDropout
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
actor_lr = 0.0001
critic_lr = 0.0001
  
class A3CAgent:
    def __init__(self):
        self.action_size = 3
        self.discount_factor = 0.9
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        K.set_learning_phase(1)
        self.sess.run(tf.global_variables_initializer())
        
        self.training_states, self.training_actions, self.training_rewards = [[], [], [], []], [[], [], [], []], []
        
        self.best_actor = [self.build_actor_model(), self.build_actor_model(), self.build_actor_model(), self.build_actor_model()]
        self.best_critic = [self.build_critic_model(), self.build_critic_model(), self.build_critic_model(), self.build_critic_model()]
        
        self.challenger_actor = [self.build_actor_model(), self.build_actor_model(), self.build_actor_model(), self.build_actor_model()]
        self.challenger_critic = [self.build_critic_model(), self.build_critic_model(), self.build_critic_model(), self.build_critic_model()]
        '''
        self.best_actor_turn1, self.best_critic_turn1 = self.build_actor_model(), self.build_critic_model()
        self.best_actor_turn2, self.best_critic_turn2 = self.build_actor_model(), self.build_critic_model()
        self.best_actor_turn3, self.best_critic_turn3 = self.build_actor_model(), self.build_critic_model()
        self.best_actor_final, self.best_critic_final = self.build_actor_model(), self.build_critic_model()
        self.challenger_actor_turn1, self.challenger_critic_turn1 = self.build_actor_model(), self.build_critic_model()
        self.challenger_actor_turn2, self.challenger_critic_turn2 = self.build_actor_model(), self.build_critic_model()
        self.challenger_actor_turn3, self.challenger_critic_turn3 = self.build_actor_model(), self.build_critic_model()
        self.challenger_actor_final, self.challenger_critic_final = self.build_actor_model(), self.build_critic_model()
        '''
        
        
        self.optimizer_actor = [self.bulid_actor_optimizer(0), self.bulid_actor_optimizer(1), self.bulid_actor_optimizer(2), self.bulid_actor_optimizer(3)]
        self.optimizer_critic = [self.bulid_critic_optimizer(0), self.bulid_critic_optimizer(1), self.bulid_critic_optimizer(2), self.bulid_critic_optimizer(3)]
        '''
        self.optimizer_turn1 = [self.bulid_actor_optimizer(1), self.bulid_critic_optimizer(1)]
        self.optimizer_turn2 = [self.bulid_actor_optimizer(2), self.bulid_critic_optimizer(2)]
        self.optimizer_turn3 = [self.bulid_actor_optimizer(3), self.bulid_critic_optimizer(3)]
        self.optimizer_final = [self.bulid_actor_optimizer(4), self.bulid_critic_optimizer(4)]
        '''
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
                self.save_best_model()
                
            first = False

    def build_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(660, input_dim=660, activation='selu', kernel_initializer='lecun_normal'))
        actor.add(AlphaDropout(0.5))
        for _ in range(10):
            actor.add(Dense(440, activation='selu', kernel_initializer='lecun_normal'))
            actor.add(AlphaDropout(0.5))
        
        actor.add(Dense(440, activation='selu', kernel_initializer='lecun_normal'))
        actor.add(Dense(3, activation='softmax', kernel_initializer='lecun_normal'))
        
        actor.summary()

        return actor
    
    def build_critic_model(self):
        critic = Sequential()
        
        critic.add(Dense(660, input_dim=660, activation='selu', kernel_initializer='lecun_normal'))
        critic.add(AlphaDropout(0.5))
        for _ in range(10):
            critic.add(Dense(440, activation='selu', kernel_initializer='lecun_normal'))
            critic.add(AlphaDropout(0.5))
    
        critic.add(Dense(440, activation='selu', kernel_initializer='lecun_normal'))
        critic.add(Dense(1, activation='linear', kernel_initializer='lecun_normal'))
        
        critic.summary()

        return critic
    
    def bulid_actor_optimizer(self, type):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])
        
        policy = self.challenger_actor[type].output
        
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)
        
        loss = cross_entropy + 0.01 * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.challenger_actor[type].trainable_weights, [],loss)
        train = K.function([self.challenger_actor[type].input, action, advantages], [loss], updates=updates)
        return train
    
    def bulid_critic_optimizer(self, type):
        discounted_prediction = K.placeholder(shape=(None,))
        
        value = self.challenger_critic[type].output
        
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = Adam(lr=self.critic_lr)
        
        updates = optimizer.get_updates(self.challenger_critic[type].trainable_weights, [],loss)
        train = K.function([self.challenger_critic[type].input, discounted_prediction], [loss], updates=updates)
            
        return train

    def load_challenger_model(self):
        for type in range(4):
            self.challenger_actor[type].load_weights('challengr_actor_' + str(type) + '.h5')
            self.challenger_critic[type].load_weights('challengr_critic_' + str(type) + '.h5')

    def save_challenger_model(self):
        for type in range(4):
            self.challenger_actor[type].save_weights('challengr_actor_' + str(type) + '.h5')
            self.challenger_critic[type].save_weights('challengr_critic_' + str(type) + '.h5')
            
    def load_best_model(self):
        for type in range(4):
            self.best_actor[type].load_weights('best_actor_' + str(type) + '.h5')
            self.best_critic[type].load_weights('best_critic_' + str(type) + '.h5')

    def save_best_model(self):
        for type in range(4):
            self.best_actor[type].save_weights('best_actor_' + str(type) + '.h5')
            self.best_critic[type].save_weights('best_critic_' + str(type) + '.h5')
        
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
        
    def train_model(self, fund):
        #discounted_prediction = self.discounted_prediction(self.training_rewards)
        
        for type in range(len(self.training_states)):
            training_states = np.zeros((len(self.training_states[type]), 660))
            for i in range(len(self.training_states[type])):
                training_states[i] = self.training_states[type][i]
                
            values = self.challenger_critic[type].predict(training_states)
            
            values = np.reshape(values, len(values))
            
            if fund >= 20000:
                reward = 1
            else:
                reward = -1
            reward = np.full_like(values, reward)
            advantages = reward - values
            
            self.optimizer_actor[type]([training_states, self.training_actions[type], advantages])
            self.optimizer_critic[type]([training_states, reward])
            
    def update_model(self):
        for type in range(4):
            self.best_actor[type].set_weights(self.challenger_actor[type].get_weights())
            self.best_critic[type].set_weights(self.challenger_critic[type].get_weights())
        
    def append_sample(self, state, action, type):
        state1 = copy.deepcopy(state)
        state1 = np.reshape(state1, [1, 660])
        
        self.training_states[type].append(state1)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.training_actions[type].append(act)
        
    def choice(self, state, dict, ai1_hands, ai1_board, ai2_board, hidden, eval, id, table_stake, type):
        calc_deck = []
        result = np.zeros(3)
        number = 0
        md = [0, 1, 2]
        case = []
        
        train_state = np.reshape(state, [1, 660])
        
        generate_deck(calc_deck)
        
        if eval == False:
            policy = self.best_actor[type].predict(train_state)[0]
            if id == 1:
                print(policy)
                
            if first == True:
                return np.random.choice(self.action_size, 1, p=policy)[0]
            elif first == False:
                return np.random.choice(self.action_size, 1, p=policy)[0]
        
        elif eval == True:
            for i in ai1_hands + ai1_board + ai2_board:
                calc_deck.remove(i)
            
            if hidden == 0:    
                for i in list(itertools.combinations(calc_deck, 2)):
                    ai2_hand = []
                    ai2_hand.append(i[0])
                    ai2_hand.append(i[1])
                    range = judge(ai2_hand + ai2_board)
                    
                    if range not in case:
                        case.append(range)
                        
            elif hidden == 1:
                for i in list(itertools.combinations(calc_deck, 3)):
                    ai2_hand = []
                    ai2_hand.append(i[0])
                    ai2_hand.append(i[1])
                    ai2_hand.append(i[2])
                    range = judge(ai2_hand + ai2_board)
                    if range not in case:
                        case.append(range)
            
            for i in case:
                state[self.table_stake(table_stake)][1][i] = 1
                eval_state = np.reshape(state, [1, 660])
                
                if id == 1:
                    result += self.best_actor[type].predict(eval_state)[0]
                elif id == 2:
                    result += self.challenger_actor[type].predict(eval_state)[0]
                    
                number += 1
                state[self.table_stake(table_stake)][1][i] = 0
                
            for i in [0, 1, 2]:
                result[i] /= number
            result /= result.sum()
            
            print(result)
            
            mx = np.where(result == max(result))
            mn = np.where(result == min(result))
            if result[mx] == 1:
                return mx[0][0]
            md.remove(mx[0])
            md.remove(mn[0])
            
            
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
            return 1.1
        elif reward <= 16000:
            return 1.2
        elif reward <= 32000:
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
    
    def table_stake(self, table_stake):
        if table_stake <= 4000:
            return 0
        elif table_stake <= 8000:
            return 1
        elif table_stake <= 16000:
            return 2
        elif table_stake <= 32000:
            return 3
        elif table_stake <= 64000:
            return 4
        elif table_stake <= 128000:
            return 5
        elif table_stake <= 256000:
            return 6
        elif table_stake <= 512800:
            return 7
        elif table_stake <= 1024000:
            return 8
        else:
            return 9
    
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
            
            deck = []
            dict = {}
            l = 0
            
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
                    
            ai1_state = np.zeros(660)
            ai2_state = np.zeros(660)
            
            ai1_state = np.reshape(ai1_state, [10, 2, 33])
            ai2_state = np.reshape(ai2_state, [10, 2, 33])
            
            while True :
                #turn1
                for _ in range(3):
                    ai1_hands.append(deck.pop())
                    ai2_hands.append(deck.pop())
                
                ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                
                ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 0)
                ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 0)
                if eval == False : self.append_sample(ai1_state, ai1_action, 0)
                
                ai1_board.append(ai1_hands.pop(ai1_action))
                ai2_board.append(ai2_hands.pop(ai2_action))
                
                print('turn2')
                
                #turn2
                for _ in range(2):
                    ai1_board.append(deck.pop())
                    ai2_board.append(deck.pop())
                    
                ai1_state = np.zeros_like(ai1_state)
                ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                ai2_state = np.zeros_like(ai2_state)
                ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 1)
                    if eval == False : self.append_sample(ai1_state, ai1_action, 1)
                    if(ai1_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                    
                    ai2_state = np.zeros_like(ai2_state)
                    ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                    ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                    
                    ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 1)
                    if(ai2_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                    
                    ai1_state = np.zeros_like(ai1_state)
                    ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                    ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                    
                    if(ai2_action == 1):
                        ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 1)
                        if ai1_action == 1:
                            ai1_action = 0
                        if eval == False : self.append_sample(ai1_state, ai1_action, 1)
                        if(ai1_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                            winner = 2
                            break
                        stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                        
                        
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 1)
                    if(ai2_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                    
                    ai1_state = np.zeros_like(ai1_state)
                    ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                    ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                    
                    ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 1)
                    if eval == False : self.append_sample(ai1_state, ai1_action, 1)
                    if(ai1_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                    
                    ai2_state = np.zeros_like(ai2_state)
                    ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                    ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                    
                    if(ai1_action == 1):
                        ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 1)
                        if(ai2_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                            winner = 1
                            break
                        ai2_action = 0
                        stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror1")
                    print(check_turn(ai1_board, ai2_board))
                    break
                
                print('turn3')
                #turn3
                stake = 0
            
                ai1_board.append(deck.pop())
                ai2_board.append(deck.pop())
                
                ai1_state = np.zeros_like(ai1_state)
                ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                ai2_state = np.zeros_like(ai2_state)
                ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 2)
                    if eval == False : self.append_sample(ai1_state, ai1_action, 2)
                    if(ai1_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                    
                    ai2_state = np.zeros_like(ai2_state)
                    ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                    ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                    
                    ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 2)
                    if(ai2_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                    
                    ai1_state = np.zeros_like(ai1_state)
                    ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                    ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
            
                    if(ai2_action == 1):
                        while(True):
                            ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 2)
                            if(table_stake>1000000 and ai1_action == 1):
                                ai1_action = 0
                            if eval == False : self.append_sample(ai1_state, ai1_action, 2)
                            if(ai1_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                            
                            ai2_state = np.zeros_like(ai2_state)
                            ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                            ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                            
                            if(ai1_action == 0):
                                break
                            
                            ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 2)
                            if(table_stake>1000000 and ai2_action == 1):
                                ai2_action = 0
                            if(ai2_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                            
                            ai1_state = np.zeros_like(ai1_state)
                            ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                            ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                            
                            if(ai2_action == 0):
                                break
                        
                
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 2)
                    if(ai2_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                    
                    ai1_state = np.zeros_like(ai1_state)
                    ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                    ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                    
                    ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 2)
                    if eval == False : self.append_sample(ai1_state, ai1_action, 2)
                    if(ai1_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                    
                    ai2_state = np.zeros_like(ai2_state)
                    ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                    ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                    
                    if(ai1_action == 1):
                        while(True):
                            ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 0, eval, 2, table_stake, 2)
                            if(table_stake>1000000 and ai2_action == 1):
                                ai2_action = 0
                            if(ai2_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                            
                            ai1_state = np.zeros_like(ai1_state)
                            ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                            ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                            
                            if(ai2_action == 0):
                                break
                            
                            ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1, table_stake, 2)
                            if(table_stake>1000000 and ai1_action == 1):
                                ai1_action = 0
                            if eval == False : self.append_sample(ai1_state, ai1_action, 2)
                            if(ai1_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                            
                            ai2_state = np.zeros_like(ai2_state)
                            ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                            ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                            
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
                
                ai1_hands.append(deck.pop())
                ai2_hands.append(deck.pop())
                
                ai1_state = np.zeros_like(ai1_state)
                ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                ai2_state = np.zeros_like(ai2_state)
                ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                
                if(table_stake < 1000000):
                    if(check_turn(ai1_board, ai2_board) == 1):
                        ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1, table_stake, 3)
                        if eval == False : self.append_sample(ai1_state, ai1_action, 3)
                        if(ai1_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                            winner = 2
                            break
                        stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                        
                        ai2_state = np.zeros_like(ai2_state)
                        ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                        ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                        
                        ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2, table_stake, 3)
                        if(ai2_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                            winner = 1
                            break
                        stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                        
                        ai1_state = np.zeros_like(ai1_state)
                        ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                        ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                        
                        if(ai2_action == 1):
                            while(True):
                                ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1, table_stake, 3)
                                if(table_stake>1000000 and ai1_action == 1):
                                    ai1_action = 0
                                if eval == False : self.append_sample(ai1_state, ai1_action, 3)
                                if(ai1_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                    winner = 2
                                    break
                                stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                                
                                ai2_state = np.zeros_like(ai2_state)
                                ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                                ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                                
                                if(ai1_action == 0):
                                    break
                                
                                ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2, table_stake, 3)
                                if(table_stake>1000000 and ai2_action == 1):
                                    ai2_action = 0
                                if(ai2_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                    winner = 1
                                    break
                                stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                                
                                ai1_state = np.zeros_like(ai1_state)
                                ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                                ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                                
                                if(ai2_action == 0):
                                    break
                            
                    
                    elif(check_turn(ai1_board, ai2_board) == 2):
                        ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2, table_stake, 3)
                        if(ai2_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                            winner = 1
                            break
                        stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                        
                        ai1_state = np.zeros_like(ai1_state)
                        ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                        ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                                  
                        ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1, table_stake, 3)
                        if eval == False : self.append_sample(ai1_state, ai1_action, 3)
                        if(ai1_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                            winner = 2
                            break
                        stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                        
                        ai2_state = np.zeros_like(ai2_state)
                        ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                        ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                        
                        if(ai1_action == 1):
                            while(True):
                                ai2_action = self.choice(ai2_state, dict, ai2_hands, ai2_board, ai1_board, 1, eval, 2, table_stake, 3)
                                if(table_stake>1000000 and ai2_action == 1):
                                    ai2_action = 0
                                if(ai2_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                    winner = 1
                                    break
                                stake, table_stake, best_fund, challenger_fund= bet(ai2_action, stake, table_stake, 2, best_fund, challenger_fund)
                                
                                ai1_state = np.zeros_like(ai1_state)
                                ai1_state[self.table_stake(table_stake)][0][judge(ai1_hands + ai1_board)] = 1
                                ai1_state[self.table_stake(table_stake)][1][judge(ai2_hands + ai2_board)] = 1
                                
                                if(ai2_action == 0):
                                    break
                                
                                ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1, table_stake, 3)
                                if(table_stake>1000000 and ai1_action == 1):
                                    ai1_action = 0
                                if eval == False : self.append_sample(ai1_state, ai1_action, 3)
                                if(ai1_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                    winner = 2
                                    break
                                stake, table_stake, best_fund, challenger_fund = bet(ai1_action, stake, table_stake, 1, best_fund, challenger_fund)
                                
                                ai2_state = np.zeros_like(ai2_state)
                                ai2_state[self.table_stake(table_stake)][0][judge(ai2_hands + ai2_board)] = 1
                                ai2_state[self.table_stake(table_stake)][1][judge(ai1_hands + ai1_board)] = 1
                                
                                if(ai1_action == 0):
                                    break
                            
                    else:
                        print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror3")
                        print(check_turn(ai1_board, ai2_board))
                        break
                
                if(winner != 0):
                    if judge_end(ai1_hands + ai1_board, ai2_hands + ai2_board) == 1:
                        switch = True
                    break
                
                winner = judge_end(ai1_hands + ai1_board, ai2_hands + ai2_board)
                break
            
            
            if winner == 1:
                best_fund += table_stake
                #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)     
            elif winner == 2:
                challenger_fund += table_stake
                #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
            
            
            if episode%1000 == 0:
                self.save_challenger_model()
                
            if eval == True:
                eval_episode += 1
                if eval_episode == 100:
                    break
                
            elif eval == False:
                episode += 1
                if episode%100 == 0:
                    self.train_model(best_fund)
                    best_fund = 0
                    challenger_fund = 0
                    self.training_states, self.training_actions = [[], [], [], []], [[], [], [], []]
                
                if first == True:
                    if episode%10000 == 0:
                        self.update_model()
                        break
                elif first == False:
                    if episode%1000 ==0:
                        break
            
if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()
    
    
    
