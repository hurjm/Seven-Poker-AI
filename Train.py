from keras.layers import Dense, AlphaDropout
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
import tensorflow as tf
import numpy as np
from Calc import *
import keras

episode = 1
  
class A3CAgent:
    def __init__(self):
        self.action_size = 3
        self.actor_lr = 0.0001
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        K.set_learning_phase(1)
        self.sess.run(tf.global_variables_initializer())
        
        self.training_states = [[], []]
        self.ai_actor = [self.build_ai1_actor_model(), self.build_ai2_actor_model()]
        
        self.optimizer_actor = [self.bulid_actor_optimizer(0), self.bulid_actor_optimizer(1)]
        
        #self.load_model()
 

    def build_ai1_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(45, input_dim=45, activation='relu', kernel_initializer='he_normal'))
        
        for _ in range(10):
            actor.add(Dense(36, activation='relu', kernel_initializer='he_normal'))
            
        actor.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor
    
    def build_ai2_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(54, input_dim=54, activation='relu', kernel_initializer='he_normal'))
        
        for _ in range(10):
            actor.add(Dense(36, activation='relu', kernel_initializer='he_normal'))
            
        actor.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor
        
    def bulid_actor_optimizer(self, num):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])
        
        policy = self.ai_actor[num].output
        
        action_prob = K.sum(action * policy, axis=1)
        
        cross_entropy = K.log(action_prob) * advantages
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.ai_actor[num].trainable_weights, [],loss)
        train = K.function([self.ai_actor[num].input, action, advantages], [loss], updates=updates)
        return train
    
    
    def load_model(self):
        for type in range(2):
            self.ai_actor[type].load_weights('actor_' + str(type) + '.h5')
            #self.critic_model[type].load_weights('critic_' + str(type) + '.h5')

    def save_model(self):
        for type in range(2):
            self.ai_actor[0].save_weights('player_model.h5')
            #self.critic_model[type].save_weights('critic_' + str(type) + '.h5')
        
        
    def table(self):
        global episode
        time.sleep(5)
        self.stack = [np.zeros(47, dtype=int), np.zeros(54, dtype=int)]
        
        for i in range(47):
            test = np.zeros(47).reshape(1, 47)
            test[0][i] = 1
            print(self.ai_actor[0].predict(test)[0], i)
        print(' ')
        for i in range(54):
            test = np.zeros(54).reshape(1, 54)
            test[0][i] = 1
            print(self.ai_actor[1].predict(test)[0], i)    
        '''
        
        action1 = np.zeros(3).reshape(1, 3)
        action1[0][1] = 1
        reward = np.ones(1)
        test = np.zeros(45).reshape(1, 45)
        test[0][28]=1
        '''
        while True:
            '''
            for i in range(45):
                test1 = np.zeros(45).reshape(1, 45)
                test1[0][i] = 1
                print(self.ai_actor[0].predict(test1)[0], i)
            print(' ')
            print(episode)
            self.optimizer_actor[0]([test, action1, reward])
            print(self.ai_actor[0].predict(test)[0])
            episode += 1

            self.save_model()
            if episode == 300:
                break
        '''
            ai1_state1 = np.zeros(47).reshape(1, 47)
            ai2_state1 = np.zeros(47).reshape(1, 47)
            ai1_state2 = np.zeros(54).reshape(1, 54)
            ai2_state2 = np.zeros(54).reshape(1, 54)
            
            deck = []
            
            generate_deck(deck)
            
            random.shuffle(deck)
                
            ai1_card = np.array([])
            ai2_card = np.array([])
            
            for _ in range(5):
                ai1_card = np.append(ai1_card, deck.pop())
                ai2_card = np.append(ai2_card, deck.pop())
            
            winner1, range1 = judge_state(ai1_card, ai2_card)
            winner2, range2 = judge_state(ai2_card, ai1_card)
            
            a1 = self.preprocess(winner1, range1, ai1_card, ai2_card, 0)
            a2 = self.preprocess(winner2, range2, ai2_card, ai1_card, 0)
            
            ai1_state1[0][a1] += 1
            ai2_state1[0][a2] += 1
            self.stack[0][a1] += 1
            self.stack[0][a2] += 1
        
            self.training_states[0].append(ai1_state1)
            self.training_states[1].append(ai2_state1)
            
            ai1_card = np.append(ai1_card, deck.pop())
            ai2_card = np.append(ai2_card, deck.pop())
            
            winner1, range1 = judge_state(ai1_card, ai2_card)
            winner2, range2 = judge_state(ai2_card, ai1_card)
            
            #print(ai1_card, ai2_card)
            
            
            b1 = self.preprocess(winner1, range1, ai1_card, ai2_card, 1)
            b2 = self.preprocess(winner2, range2, ai2_card, ai1_card, 1)
            
            ai1_state2[0][b1] += 1
            ai2_state2[0][b2] += 1
            self.stack[1][b1] += 1
            self.stack[1][b2] += 1
            
            self.training_states[0].append(ai1_state2)
            self.training_states[1].append(ai2_state2)
            
            ai1_card = np.append(ai1_card, deck.pop())
            ai2_card = np.append(ai2_card, deck.pop())
            
            winner, _ = judge_end(ai1_card, ai2_card)
            
            self.train(winner)
            episode += 1
            
            if episode%1000 == 0:
                self.save_model()
                
            print(self.stack[0])
            print(self.stack[1])
            self.training_states = [[], []]
            if (self.stack[0][:] > 10000).all() and (self.stack[1][:] > 10000).all():
                break
    
    def train(self, winner):
        action1 = np.zeros(2).reshape(1, 2)
        action2 = np.zeros(2).reshape(1, 2)
        action1[0][0] = 1
        action2[0][1] = 1
        reward = np.ones(1)
        
        for i in range(2):
            #if self.ai_actor[i].predict(self.training_states[winner][i])[0][0] < 0.99:
            if self.stack[i][np.where(self.training_states[winner][i][0] == 1)] < 10000:
                self.optimizer_actor[i]([self.training_states[winner][i], action1, reward])
                #print(self.ai_actor[i].predict(self.training_states[winner][i])[0])
                
            
            #if self.ai_actor[i].predict(self.training_states[winner + 1][i])[0][1] < 0.99:
            if self.stack[i][np.where(self.training_states[winner + 1][i][0] == 1)] < 10000:
                self.optimizer_actor[i]([self.training_states[winner + 1][i], action2, reward])
                #print(self.ai_actor[i].predict(self.training_states[winner + 1][i])[0])
                
    def preprocess(self, winner, range, ai_card, player_card, turn):
        count = 0
        count2 = 0
        ai_state = 0
        if range[winner][0] > 2:
            return False
        
        if turn == 0:
            if range[0][0] == 0 and range[1][0] == 0:
                for i in ai_card:
                    for j in player_card:
                        if i.value > j.value:
                            count += 1
                ai_state = count
                
            elif range[0][0] == 0 and range[1][0] == 1:
                for i in ai_card:
                    if i.value > range[1][1]:
                        count += 1
                ai_state = count + 26
            
            elif range[0][0] == 0 and range[1][0] == 2:
                for i in ai_card:
                    if i.value > range[1][1]:
                        count += 1
                ai_state = count + 32   
            
            elif range[0][0] == 1 and range[1][0] == 0:
                for i in player_card:
                    if i.value > range[0][1]:
                        count += 1
                ai_state = count + 38
                
            elif range[0][0] == 1 and range[1][0] == 1:
                if winner == 0:
                    ai_state = 44
                        
                elif winner == -1:
                    ai_state = 45
                
            elif range[0][0] == 1 and range[1][0] == 2:
                ai_state = 46
            else:
                return False
            
                        
                        
        elif turn == 1:
            
            if range[0][0] == 0 and range[1][0] == 0:
                for i in ai_card:
                    for j in player_card:
                        if i.value > j.value:
                            count += 1
                ai_state = count
                
            elif range[0][0] == 0 and range[1][0] == 1:
                for i in ai_card:
                    if i.value > range[1][1]:
                        count += 1
                        
                ai_state = count + 37
            
            elif range[0][0] == 1 and range[1][0] == 0:
                for i in player_card:
                    if i.value > range[0][1]:
                        count += 1
                ai_state = count + 44
            
            elif range[0][0] == 1 and range[1][0] == 1:
                if winner == 0:
                    ai_state = 51
                        
                elif winner == -1:
                    ai_state = 52
                    
            elif range[0][0] == 1 and range[1][0] == 2:
                ai_state = 53
            else:
                return False
                       
        return ai_state
            
if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.table()
    
    
