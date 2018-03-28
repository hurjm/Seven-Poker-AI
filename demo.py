from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from Calc import *
import tensorflow as tf
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from keras import backend as K

suit_index_dict = {"s": 0, "c": 1, "h": 2, "d": 3}
reverse_suit_index = ("s", "c", "h", "d")
val_string = "AKQJT98765433"
hand_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                 "Straight", "Back Straight", "Mountain", "Flush", "Full House", "Four of a Kind",
                 "Straight Flush", "Back Straight Flush", "Royal Flush")
suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
for num in range(2, 10):
    suit_value_dict[str(num)] = num


form_class = uic.loadUiType("form.ui")[0]

btn = False
my_action = 4

'''
self.player_hand.emit(ai_hands)
            while btn != True:
                time.sleep(0.1)
            btn = False
            player_action = my_action
'''

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.graph = tf.get_default_graph()
        
        self.best_actor = [self.build_actor_model(), self.build_actor_model(), self.build_actor_model(), self.build_actor_model()]
        #self.best_actor.load_weights('best_actor_' + str(type) + '.h5')
        
        self.setupUi(self)
        self.th = Table(self.best_actor, self.sess, self.graph)
        self.th.player_hand.connect(self.player_board)

        self.callorcheck.clicked.connect(self.callorcheck_click)
        self.bet.clicked.connect(self.bet_click)
        self.fold.clicked.connect(self.fold_click)
        self.start.clicked.connect(self.start_click)


    def callorcheck_click(self):
        global btn, my_action
        my_action = 0
        btn = True

    def bet_click(self):
        global btn, my_action
        my_action = 1
        btn = True

    def fold_click(self):
        global btn, my_action
        my_action = 2
        btn = True

    def start_click(self):
        self.th.start()
        self.start.setDisabled(True)

    @pyqtSlot(list)
    def player_board(self, cards):
        self.label.setPixmap(QPixmap('./resource/' + str(cards[0])))
        self.label_2.setPixmap(QPixmap('./resource/' + str(cards[1])))
        #self.label_3.setPixmap(QPixmap('./resource/' + str(cards[2])))
    
    def build_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(868, input_dim=868, activation='relu', kernel_initializer='he_normal'))
        
        for _ in range(10):
            actor.add(Dense(868, activation='relu', kernel_initializer='he_normal'))
            
        actor.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))
        
        actor._make_predict_function()
        actor.summary()
        
        return actor

class Table(QThread):
    player_hand = pyqtSignal(list)

    def __init__(self, best_actor, sess, graph):
        super().__init__()
        
        self.best_actor = best_actor
        self.sess = sess
        self.graph = graph
        print(self.best_actor)

    def run(self):
        global btn
        global my_action
        ai_fund = 200000
        player_fund = 200000
        
        while True:
            deck = []
            dict = {}
            l = 0
            
            generate_deck(deck)
            
            for i in deck:
                dict[str(i)] = l
                l += 1
                
            random.shuffle(deck)
                
            ai_hands = []
            ai_board = []
            
            player_hands = []
            player_board = []
            
            #seed money
            table_stake = 2000
            ai_fund -= 1000
            player_fund -= 1000
            
            stake = 0
            winner = 0
                    
            ai_state = np.zeros(868)
            
            ai_state = np.reshape(ai_state, [14, 2, 31])
            
            while True :
                #turn1
                for _ in range(3):
                    ai_hands.append(deck.pop())
                    player_hands.append(deck.pop())
                
                ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                
                self.player_hand.emit(player_hands)
                while btn != True:
                    time.sleep(0.1)
                btn = False
                player_action = my_action
                self.player_hand.emit(player_hands)
                while btn != True:
                    time.sleep(0.1)
                btn = False
                player_action = my_action
                ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 0)
                ai_board.append(ai_hands.pop(ai_action))
                player_board.append(player_hands.pop(player_action))
                print('turn2')
                
                #turn2
                for _ in range(2):
                    ai_board.append(deck.pop())
                    player_board.append(deck.pop())
                    
                ai_state = np.zeros_like(ai_state)
                ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                
                if(check_turn(ai_board, player_board) == 1):
                    ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 1)
                     
                    if(ai_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                    
                    
                    self.player_hand.emit(player_hands)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    player_action = my_action
                    if(player_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                    
                    ai_state = np.zeros_like(ai_state)
                    ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                    ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                    
                    if(player_action == 1):
                        ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 1)
                        if ai_action == 1:
                            ai_action = 0
                         
                        if(ai_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                            winner = 2
                            break
                        stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                        
                        
                elif(check_turn(ai_board, player_board) == 2):
                    
                    self.player_hand.emit(player_hands)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    player_action = my_action
                    if(player_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                    
                    ai_state = np.zeros_like(ai_state)
                    ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                    ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                    
                    ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 1)
                     
                    if(ai_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                    
                    
                    if(ai_action == 1):
                        self.player_hand.emit(player_hands)
                        while btn != True:
                            time.sleep(0.1)
                        btn = False
                        player_action = my_action
                        if(player_action == 2):
                            #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                            winner = 1
                            break
                        player_action = 0
                        stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror1")
                    print(check_turn(ai_board, player_board))
                    break
                
                print('turn3')
                #turn3
                stake = 0
            
                ai_board.append(deck.pop())
                player_board.append(deck.pop())
                
                ai_state = np.zeros_like(ai_state)
                ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                
                if(check_turn(ai_board, player_board) == 1):
                    ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 2)
                     
                    if(ai_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                    
                    
                    self.player_hand.emit(player_hands)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    player_action = my_action
                    if(player_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                    
                    ai_state = np.zeros_like(ai_state)
                    ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                    ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
            
                    if(player_action == 1):
                        while(True):
                            ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 2)
                            if(table_stake>=200000 and ai_action == 1):
                                ai_action = 0
                             
                            if(ai_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                            
                            
                            if(ai_action == 0):
                                break
                            
                            self.player_hand.emit(player_hands)
                            while btn != True:
                                time.sleep(0.1)
                            btn = False
                            player_action = my_action
                            if(table_stake>=200000and player_action == 1):
                                player_action = 0
                            if(player_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                            
                            ai_state = np.zeros_like(ai_state)
                            ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                            ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                            
                            if(player_action == 0):
                                break
                        
                
                elif(check_turn(ai_board, player_board) == 2):
                    self.player_hand.emit(player_hands)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    player_action = my_action
                    if(player_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                        winner = 1
                        break
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                    
                    ai_state = np.zeros_like(ai_state)
                    ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                    ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                    
                    ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 2)
                     
                    if(ai_action == 2):
                        #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                        winner = 2
                        break
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                    
                    
                    if(ai_action == 1):
                        while(True):
                            self.player_hand.emit(player_hands)
                            while btn != True:
                                time.sleep(0.1)
                            btn = False
                            player_action = my_action
                            if(table_stake>=200000 and player_action == 1):
                                player_action = 0
                            if(player_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                winner = 1
                                break
                            stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                            
                            ai_state = np.zeros_like(ai_state)
                            ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                            ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                            
                            if(player_action == 0):
                                break
                            
                            ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 0, table_stake, 2)
                            if(table_stake>=200000 and ai_action == 1):
                                ai_action = 0
                             
                            if(ai_action == 2):
                                #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                winner = 2
                                break
                            stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                            
                            
                            if(ai_action == 0):
                                break
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror2")
                    print(check_turn(ai_board, player_board))
                    break
                    
                if(winner != 0):
                    break
                
                print('final')
                #final
                stake = 0
                
                ai_hands.append(deck.pop())
                player_hands.append(deck.pop())
                
                ai_state = np.zeros_like(ai_state)
                ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                if(table_stake < 4000000):
                    if(check_turn(ai_board, player_board) == 1):
                        ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 1, table_stake, 3)
                        if(ai_action == 2):
                            winner = 2
                            break
                        stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                        
                        self.player_hand.emit(player_hands)
                        while btn != True:
                            time.sleep(0.1)
                        btn = False
                        player_action = my_action
                        
                        if(player_action == 2):
                            winner = 1
                            break
                        stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                        
                        ai_state = np.zeros_like(ai_state)
                        ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                        ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                        
                        if(player_action == 1):
                            while True:
                                ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 1, table_stake, 3)
                                if(table_stake>=200000 and ai_action == 1):
                                    ai_action = 0
                                 
                                if(ai_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                    winner = 2
                                    break
                                stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                                
                                
                                if(ai_action == 0):
                                    break
                                
                                self.player_hand.emit(player_hands)
                                while btn != True:
                                    time.sleep(0.1)
                                btn = False
                                player_action = my_action
                                
                                if(table_stake>=200000 and player_action == 1):
                                    player_action = 0
                                if(player_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                    winner = 1
                                    break
                                stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                                
                                ai_state = np.zeros_like(ai_state)
                                ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                                ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                                
                                if(player_action == 0):
                                    break
                            
                    
                    elif(check_turn(ai_board, player_board) == 2):
                        
                        self.player_hand.emit(player_hands)
                        while btn != True:
                            time.sleep(0.1)
                        btn = False
                        player_action = my_action
                        
                        if(player_action == 2):
                            winner = 1
                            break
                        stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                        
                        ai_state = np.zeros_like(ai_state)
                        ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                        ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                              
                        ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 1, table_stake, 3)
                         
                        if(ai_action == 2):
                            winner = 2
                            break
                        stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                        
                        
                        if(ai_action == 1):
                            while True:
                                self.player_hand.emit(player_hands)
                                while btn != True:
                                    time.sleep(0.1)
                                btn = False
                                player_action = my_action
                                
                                if(table_stake>=200000 and player_action == 1):
                                    player_action = 0
                                if(player_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = self.reward(table_stake)
                                    winner = 1
                                    break
                                stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, 2, ai_fund, player_fund)
                                
                                ai_state = np.zeros_like(ai_state)
                                ai_state[self.table_stake(table_stake)][0][judge(ai_hands + ai_board)] = 1
                                ai_state[self.table_stake(table_stake)][1][judge(player_hands + player_board)] = 1
                                
                                if(player_action == 0):
                                    break
                                
                                ai_action = self.choice(ai_state, dict, ai_hands, ai_board, player_board, 1, table_stake, 3)
                                if(table_stake>=200000 and ai_action == 1):
                                    ai_action = 0
                                 
                                if(ai_action == 2):
                                    #self.training_rewards[len(self.training_rewards) - 1] = -self.reward(table_stake)
                                    winner = 2
                                    break
                                stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 1, ai_fund, player_fund)
                                
                                
                                if(ai_action == 0):
                                    break
                            
                    else:
                        print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror3")
                        print(check_turn(ai_board, player_board))
                        break
                
                if(winner == 0):
                    winner = judge_end(ai_hands + ai_board, player_hands + player_board)
                    
                break
            
            if winner == 1:
                ai_fund += table_stake
            elif winner == 2:
                player_fund += table_stake
                
    
    def table_stake(self, table_stake):
        if table_stake == 2000:
            return 0
        elif table_stake == 3000:
            return 1
        elif table_stake == 4000:
            return 2
        elif table_stake == 6000:
            return 3
        elif table_stake == 8000:
            return 4
        elif table_stake == 12000:
            return 5
        elif table_stake == 16000:
            return 6
        elif table_stake == 24000:
            return 7
        elif table_stake == 32000:
            return 8
        elif table_stake == 48000:
            return 9
        elif table_stake == 64000:
            return 10
        elif table_stake == 96000:
            return 11
        elif table_stake == 128000:
            return 12
        else:
            return 13
        
    def choice(self, state, dict, ai_hands, ai_board, player_board, hidden, table_stake, type):
        calc_deck = []
        result = np.zeros(3)
        number = 0
        md = [0, 1, 2]
        case = []
        generate_deck(calc_deck)
        for i in ai_hands + ai_board + player_board:
            calc_deck.remove(i)
            
        if hidden == 0:    
            for i in list(itertools.combinations(calc_deck, 2)):
                player_hands = []
                player_hands.append(i[0])
                player_hands.append(i[1])
                range = judge(player_hands + player_board)
                
                if range not in case:
                    case.append(range)
                    
        elif hidden == 1:
            for i in list(itertools.combinations(calc_deck, 3)):
                player_hands = []
                player_hands.append(i[0])
                player_hands.append(i[1])
                player_hands.append(i[2])
                range = judge(player_hands + player_board)
                if range not in case:
                    case.append(range)
        
        for i in case:
            state[self.table_stake(table_stake)][1][i] = 1
            eval_state = np.reshape(state, [1, 868])
            
            with self.graph.as_default():
                result += self.best_actor[type].predict(eval_state)[0]
                
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
            return np.random.choice(3, 1, p=result)[0]





if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()

