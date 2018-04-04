from keras.layers import Dense, AlphaDropout
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
import tensorflow as tf
import numpy as np
from Calc import *
import copy

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
first = False
switch = False
  
class A3CAgent:
    def __init__(self):
        self.action_size = 3
        self.actor_lr = 0.01
        self.critic_lr = 0.01
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        K.set_learning_phase(1)
        self.sess.run(tf.global_variables_initializer())
        
        self.training_states, self.training_actions, self.training_rewards = [[], [], [], []], [[], [], [], []], []
        
        self.best_actor = [self.build_actor_model(), self.build_actor_model(), self.build_actor_model(), self.build_actor_model()]
        self.best_critic = [self.build_critic_model(), self.build_critic_model(), self.build_critic_model(), self.build_critic_model()]
        
        self.challenger_actor = [self.build_actor_model(), self.build_actor_model(), self.build_actor_model(), self.build_actor_model()]
        self.challenger_critic = [self.build_critic_model(), self.build_critic_model(), self.build_critic_model(), self.build_critic_model()]
        
        self.optimizer_actor = [self.bulid_actor_optimizer(0), self.bulid_actor_optimizer(1), self.bulid_actor_optimizer(2), self.bulid_actor_optimizer(3)]
        self.optimizer_critic = [self.bulid_critic_optimizer(0), self.bulid_critic_optimizer(1), self.bulid_critic_optimizer(2), self.bulid_critic_optimizer(3)]

        #self.load_best_model()
        #self.load_challenger_model()
 
    def train(self):
        global episode
        global final
        global eval_episode
        global best_fund
        global challenger_fund
        global first
        time.sleep(5)
        first = False
        while final < 10:
            self.table(False)
            '''
            print('eval')
            eval_episode = 1
            if first == False:
                self.table(True)
            
            if challenger_fund >= 200000:
                print('update')
                self.update_model()
                final = 0
            else:
                final += 1
                self.save_best_model()
            '''
                
            #first = False

    def build_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(784, input_dim=784, activation='selu', kernel_initializer='lecun_normal'))
        actor.add(AlphaDropout(0.5))
        for _ in range(10):
            actor.add(Dense(578, activation='selu', kernel_initializer='lecun_normal'))
            actor.add(AlphaDropout(0.5))
        
        actor.add(Dense(578, activation='selu', kernel_initializer='lecun_normal'))
        actor.add(Dense(3, activation='softmax', kernel_initializer='lecun_normal'))
        
        actor.summary()

        return actor
    
    def build_critic_model(self):
        critic = Sequential()
        
        critic.add(Dense(784, input_dim=784, activation='selu', kernel_initializer='lecun_normal'))
        critic.add(AlphaDropout(0.5))
        for _ in range(10):
            critic.add(Dense(578, activation='selu', kernel_initializer='lecun_normal'))
            critic.add(AlphaDropout(0.5))
    
        critic.add(Dense(578, activation='selu', kernel_initializer='lecun_normal'))
        critic.add(Dense(1, activation='linear', kernel_initializer='lecun_normal'))
        
        critic.summary()

        return critic
    
    def bulid_actor_optimizer(self, type):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])
        
        policy = self.challenger_actor[type].output
        
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
            self.challenger_actor[type].load_weights('challenger_actor_' + str(type) + '.h5')
            self.challenger_critic[type].load_weights('challenger_critic_' + str(type) + '.h5')

    def save_challenger_model(self):
        for type in range(4):
            self.challenger_actor[type].save_weights('challenger_actor_' + str(type) + '.h5')
            self.challenger_critic[type].save_weights('challenger_critic_' + str(type) + '.h5')
            
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
        length = 0
        for i in range(4):
            if self.training_states[i]:
                length += 1
                
        for type in range(length):
            training_states = np.zeros((len(self.training_states[type]), 784))
            for i in range(len(self.training_states[type])):
                training_states[i] = self.training_states[type][i]
                
            values = self.challenger_critic[type].predict(training_states)
            
            values = np.reshape(values, len(values))
            
            if fund >= 40000:
                reward = 1
            else:
                reward = -1
                
            if type == 3:
                for i in range(len(self.training_rewards)):
                    if self.training_rewards[i] == 0:
                        self.training_rewards[i] = reward
                
                reward = np.copy(self.training_rewards)
            else:
                reward = np.full_like(values, reward)
            
            advantages = reward - values
            print(training_states)
            print(self.training_actions)
            print(values)

            self.optimizer_actor[type]([training_states, self.training_actions[type], advantages])
            self.optimizer_critic[type]([training_states, reward])
            
    def update_model(self):
        for type in range(4):
            self.best_actor[type].set_weights(self.challenger_actor[type].get_weights())
            self.best_critic[type].set_weights(self.challenger_critic[type].get_weights())
        
    def append_sample(self, state, action, type):
        state1 = copy.deepcopy(state)
        state1 = np.reshape(state1, [1, 784])
        
        self.training_states[type].append(state1)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.training_actions[type].append(act)
        
    def append_sample_final(self, state, action, ai1_cards, ai2_cards, turn):
        state1 = copy.deepcopy(state)
        state1 = np.reshape(state1, [1, 784])
        
        if judge_end(ai1_cards, ai2_cards) == 1:
            if action == 1:
                self.training_rewards.append(1)
            elif action == 2:
                self.training_rewards.append(-1)
            elif action == 0:
                if turn == 2:
                    self.training_rewards.append(-1)
                else:
                    self.training_rewards.append(0)
                    
        elif judge_end(ai1_cards, ai2_cards) == 2:
            if action == 1:
                self.training_rewards.append(-2.5)
            elif action == 2:
                self.training_rewards.append(1)
            elif action == 0:
                if turn == 2:
                    self.training_rewards.append(-2.5)
                else:
                    self.training_rewards.append(0)
                    
        self.training_states[3].append(state1)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.training_actions[3].append(act)
        
    def choice(self, state, dict, ai1_hands, ai1_board, ai2_board, hidden, eval, id, table_stake, type):
        calc_deck = []
        result = np.zeros(3)
        number = 0
        md = [0, 1, 2]
        case = []
        
        train_state = np.reshape(state, [1, 784])
        
        generate_deck(calc_deck)
        
        if eval == False:
            policy = self.best_actor[type].predict(train_state)[0]

            print(ai1_hands + ai1_board)
            print(ai2_board)
            print(table_stake)
            print(policy)
            return np.random.choice(3, 1, p=[0.5, 0.5, 0])[0]
            #return np.random.choice(3, 1, p=policy)[0]
        
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
                eval_state = np.reshape(state, [1, 784])
                
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

    def table(self):
        global episode
        
        while True:
            print(episode)
            
            deck = []
            #dict = {}
            #l = 0
            
            generate_deck(deck)
            
            #for i in deck:
            #    dict[str(i)] = l
            #    l += 1
                
            random.shuffle(deck)
                
            ai1_hands = np.array([])
            ai1_board = np.array([])
            
            ai2_hands = np.array([])
            ai2_board = np.array([])
            
            winner = 0
                    
            ai1_state = np.zeros(3136)
            ai2_state = np.zeros(3136)
            
            ai1_state = np.reshape(ai1_state, [14, 4, 14, 4])
            ai2_state = np.reshape(ai2_state, [14, 4, 14, 4])
            
            for _ in range(3):
                ai1_hands = np.append(ai1_hands, deck.pop())
                ai2_hands = np.append(ai2_hands, deck.pop())
            
            for _ in range(2):
                ai1_board = np.append(ai1_board, deck.pop())
                ai2_board = np.append(ai2_board, deck.pop())
            
            
    def preprocess(self, card):
        val_list = np.zeros(52)
        val_list = np.reshape(val_list, (13, 4))
        suit_list = np.zeros(4)
        straight_list = np.array([])
        count = 3
        
        for i in range(len(card)):
            for j in range(4):
                if val_list[card[i].value][count - j] == 0:
                    count -= 1
                    continue
                elif val_list[card[i].value][count - j] == 1:
                    val_list[card[i].value][count - j] = 0
                    val_list[card[i].value][count + 1 - j] = 1
                    count = 3
                    break
                
            suit_list[card[i].suit_index] += 1
            straight_list = np.append(straight_list, card[i].value)
        
        straight_list = np.sort(straight_list)
        count = 0
        for i in range(len(straight_list) - 1):
            straight_list[i]
            
            
            
            
               
            
if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()
    
    
    
