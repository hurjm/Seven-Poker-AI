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

suit_index_dict = {"s": 0, "c": 1, "h": 2, "d": 3}
reverse_suit_index = ("s", "c", "h", "d")
val_string = "AKQJT98765433"
hand_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                 "Straight", "Back Straight", "Mountain", "Flush", "Full House", "Four of a Kind",
                 "Straight Flush", "Back Straight Flush", "Royal Flush")
suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
for num in range(2, 10):
    suit_value_dict[str(num)] = num

best_fund = 100000
challenger_fund = 100000
deck = []
ai1_temp1 = []
ai2_temp1 = []
ai1_temp2= []
ai2_temp2 = []
ai1_temp3 = []
ai2_temp3 = []
dict = {}
j = 0
h = 0
l = -1

ai1_hands = []
ai1_board = []

ai2_hands = []
ai2_board = []

#seed money
table_stake = 0

stake = 0
winner = 0
        
ai1_state = np.zeros(33)
ai2_state = np.zeros(33)

ai1_state = np.reshape(ai1_state, [1, 33])
ai2_state = np.reshape(ai2_state, [1, 33])

form_class = uic.loadUiType("test.ui")[0]

btn = False
player_action = 4

class MyWindow(QMainWindow, form_class):
    
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)
        
        self.th = Table()
        self.th.player_hand.connect(self.player_board)
        
        self.pushButton_1.clicked.connect(self.btn_1)
        self.start_Button.clicked.connect(self.btn_start)
        self.player_Fund.setText(str(challenger_fund))
        
    def btn_1(self):
        global btn, player_action
        player_action = 0
        btn = True
        
    def btn_start(self):
        self.th.start()
        
    @pyqtSlot(list)
    def player_board(self, cards):
        self.label1.setPixmap(QPixmap('./resource/' + str(cards[0])))
        self.label2.setPixmap(QPixmap('./resource/' + str(cards[1])))
        self.label3.setPixmap(QPixmap('./resource/' + str(cards[2])))
        
    
class Table(QThread):
    player_hand = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        
        
    def run(self):
        best_fund = 0
        challenger_fund = 0
        deck = []
        ai1_temp1 = []
        ai2_temp1 = []
        ai1_temp2= []
        ai2_temp2 = []
        ai1_temp3 = []
        ai2_temp3 = []
        dict = {}
        j = 0
        h = 0
        l = -1
        generate_deck(deck)
        
        for i in deck:
            dict[str(i)] = l
            l += 0.038
            
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
                
        ai1_state = np.zeros(33)
        ai2_state = np.zeros(33)
        
        ai1_state = np.reshape(ai1_state, [1, 33])
        ai2_state = np.reshape(ai2_state, [1, 33])
    
        #turn1
        for i in range(3):
            ai1_hands.append(deck.pop())
            ai2_hands.append(deck.pop())
        
        for i in ai1_hands:
            ai1_temp1.append(dict[str(i)])
        for i in ai2_hands:
            ai2_temp1.append(dict[str(i)])
            
        ai1_temp1.sort()
        ai2_temp1.sort()
        
        for i in range(3):
            ai1_state[0, j] = ai1_temp1[i]
            j += 1
        for i in range(3):
            ai2_state[0, h] = ai2_temp1[i]
            h += 1
        
        self.player_hand.emit(ai1_hands)
        
        while btn != True:
            print('not yet')
            time.sleep(1)
        action = player_action
        print(action)
    
    
    def table(self):
        best_fund = 0
        challenger_fund = 0
        deck = []
        ai1_temp1 = []
        ai2_temp1 = []
        ai1_temp2= []
        ai2_temp2 = []
        ai1_temp3 = []
        ai2_temp3 = []
        dict = {}
        j = 0
        h = 0
        l = -1
        generate_deck(deck)
        
        for i in deck:
            dict[str(i)] = l
            l += 0.038
            
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
                
        ai1_state = np.zeros(33)
        ai2_state = np.zeros(33)
        
        ai1_state = np.reshape(ai1_state, [1, 33])
        ai2_state = np.reshape(ai2_state, [1, 33])
        
        while winner == 0 :
            #turn1
            for i in range(3):
                ai1_hands.append(deck.pop())
                ai2_hands.append(deck.pop())
            
            for i in ai1_hands:
                ai1_temp1.append(dict[str(i)])
            for i in ai2_hands:
                ai2_temp1.append(dict[str(i)])
                
            ai1_temp1.sort()
            ai2_temp1.sort()
            
            for i in range(3):
                ai1_state[0, j] = ai1_temp1[i]
                j += 1
            for i in range(3):
                ai2_state[0, h] = ai2_temp1[i]
                h += 1
            
            ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
            ai2_action = 0
             
            
            temp1 = ai1_hands.pop(ai1_action)
            temp2 = ai2_hands.pop(ai2_action)
        
            for i in ai1_hands:
                ai1_temp2.append(dict[str(i)])
            for i in ai2_hands:
                ai2_temp2.append(dict[str(i)])
                
            ai1_temp2.sort()
            ai2_temp2.sort()
            
            ai1_state[0, 17] = ai2_temp2[0]
            ai1_state[0, 18] = ai2_temp2[1]
            ai2_state[0, 17] = ai1_temp2[0]
            ai2_state[0, 18] = ai1_temp2[1]
            print('turn2')
            #turn2
            for _ in range(2):
                ai1_board.append(deck.pop())
                ai2_board.append(deck.pop())
            
            ai1_temp3.append(dict[str(ai1_board[0])])
            ai1_temp3.append(dict[str(ai1_board[1])]) 
            ai2_temp3.append(dict[str(ai2_board[0])])
            ai2_temp3.append(dict[str(ai2_board[1])]) 
              
            ai1_temp3.sort()
            ai2_temp3.sort()
            
            ai1_state[0, 4] = ai1_temp3[0]
            ai1_state[0, 5] = ai1_temp3[1]
            ai1_state[0, 20] = ai2_temp3[0]
            ai1_state[0, 21] = ai2_temp3[1]
            
            ai2_state[0, 4] = ai2_temp3[0]
            ai2_state[0, 5] = ai2_temp3[1]
            ai2_state[0, 20] = ai1_temp3[0]
            ai2_state[0, 21] = ai1_temp3[1]
            
            ai1_board.append(temp1)
            ai2_board.append(temp2)
            
            ai1_state[0, 3] = dict[str(ai1_board[2])]
            ai1_state[0, 19] = dict[str(ai2_board[2])]
            
            ai2_state[0, 3] = dict[str(ai2_board[2])]
            ai2_state[0, 19] = dict[str(ai1_board[2])]
            
            j += 3
            h += 3
                
            if(check_turn(ai1_board, ai2_board) == 1):
                ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                #print(ai1_action)
                 
                if(ai1_action == 2):
                    winner = 2
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet1(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                
                
                ai2_action = 0
                if(ai2_action == 2):
                    winner = 1
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet1(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                
                
                if(ai2_action == 1):
                    ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                     
                    if(ai1_action == 2):
                        winner = 2
                        break
                    ai1_action = 0
                    stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet1(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                    
                    
            elif(check_turn(ai1_board, ai2_board) == 2):
                ai2_action = 0
                if(ai2_action == 2):
                     
                    winner = 1
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet1(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                
                
                ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                 
                if(ai1_action == 2):
                     
                    winner = 2
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet1(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                
                
                if(ai1_action == 1):
                    ai2_action = 0
                    if(ai2_action == 2):
                         
                        winner = 1
                        break
                    ai2_action = 0
                    stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet1(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                    
            else:
                print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror1")
                print(check_turn(ai1_board, ai2_board))
                break
            
            print('turn3')       
            #turn3
            stake = 0
            
            ai1_board.append(deck.pop())
            ai2_board.append(deck.pop())    
            
            ai1_state[0, j] = dict[str(ai1_board[3])]
            ai2_state[0, j + 16] = dict[str(ai1_board[3])]
            j += 1 
            
            ai2_state[0, h] = dict[str(ai2_board[3])]
            ai1_state[0, h + 16] = dict[str(ai2_board[3])]
            h += 1 
            
            if(check_turn(ai1_board, ai2_board) == 1):
                ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                 
                if(ai1_action == 2):
                     
                    winner = 2
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                
                
                ai2_action = 0
                if(ai2_action == 2):
                     
                    winner = 1
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                
                if(ai2_action == 1):
                    while(True):
                        ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                        if(table_stake>1500000):
                            ai1_action = 0
                         
                        if(ai1_action == 2):
                             
                            winner = 2
                            break
                        stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                        if(ai1_action == 0):
                            break
                        
                        ai2_action = 0
                        if(table_stake>1500000):
                            ai2_action = 0
                        if(ai2_action == 2):
                             
                            winner = 1
                            break
                        stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                        if(ai2_action == 0):
                            break
                    
            
            elif(check_turn(ai1_board, ai2_board) == 2):
                ai2_action = 0
                if(ai2_action == 2):
                     
                    winner = 1
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                
                
                ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                 
                if(ai1_action == 2):
                     
                    winner = 2
                    break
                stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                
                if(ai1_action == 1):
                    while(True):
                        ai2_action = 0
                        if(table_stake>1500000):
                            ai2_action = 0
                        if(ai2_action == 2):
                             
                            winner = 1
                            break
                        stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                        if(ai2_action == 0):
                            break
                        
                        ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 0, eval, 1)
                        if(table_stake>1500000):
                            ai1_action = 0
                         
                        if(ai1_action == 2):
                             
                            winner = 2
                            break
                        stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
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
            
            ai1_state[0, j] = dict[str(ai1_hands[2])]
            ai2_state[0, j + 16] = dict[str(ai1_hands[2])]
            j += 1 
        
            ai2_state[0, h] = dict[str(ai2_hands[2])]
            ai1_state[0, h + 16] = dict[str(ai2_hands[2])]
            h += 1
            if(table_stake < 1500000):
                if(check_turn(ai1_board, ai2_board) == 1):
                    ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                     
                    if(ai1_action == 2):
                         
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet3(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                    
                    
                    ai2_action = 0
                    if(ai2_action == 2):
                         
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet3(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                    
                    
                    if(ai2_action == 1):
                        while(True):
                            ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                            if(table_stake>1500000):
                                ai1_action = 0
                             
                            if(ai1_action == 2):
                                 
                                winner = 2
                                break
                            stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                            if(ai1_action == 0):
                                break
                            
                            ai2_action = 0
                            if(table_stake>1500000):
                                ai2_action = 0
                            if(ai2_action == 2):
                                 
                                winner = 1
                                break
                            stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet2(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                            if(ai2_action == 0):
                                break
                        
                
                elif(check_turn(ai1_board, ai2_board) == 2):
                    ai2_action = 0
                    if(ai2_action == 2):
                         
                        winner = 1
                        break
                    stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet3(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                    
                    
                    ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                     
                    if(ai1_action == 2):
                         
                        winner = 2
                        break
                    stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet3(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                    
                    
                    if(ai1_action == 1):
                        while(True):
                            ai2_action = 0
                            if(table_stake>1500000):
                                ai2_action = 0
                            if(ai2_action == 2):
                                 
                                winner = 1
                                break
                            stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet3(ai2_action, stake, table_stake, 2, ai1_state, ai2_state, best_fund, challenger_fund)
                            if(ai2_action == 0):
                                break
                            
                            ai1_action = self.choice(ai1_state, dict, ai1_hands, ai1_board, ai2_board, 1, eval, 1)
                            if(table_stake>1500000):
                                ai1_action = 0
                             
                            if(ai1_action == 2):
                                 
                                winner = 2
                                break
                            stake, table_stake, ai1_state, ai2_state, best_fund, challenger_fund = bet3(ai1_action, stake, table_stake, 1, ai1_state, ai2_state, best_fund, challenger_fund)
                            if(ai1_action == 0):
                                break
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror3")
                    print(check_turn(ai1_board, ai2_board))
                    break
                
                if(winner != 0):
                    break
                print(ai1_hands, ai1_board)
                print(ai2_hands, ai2_board)
                
                winner = judge(ai1_hands + ai1_board, ai2_hands + ai2_board)
         
            if winner == 1:
                best_fund += table_stake
                      
            elif winner == 2:
                challenger_fund += table_stake
                
    def load_model(self, name):
        self.best_actor.load_weights(str(name) + "_actor.h5")
        self.best_critic.load_weights(str(name) + "_critic.h5")
    
    def build_best_actor_model(self):
        best_actor = Sequential()
        best_actor.add(Dense(16, input_dim=self.state_size, activation='selu', kernel_initializer='he_uniform'))
        for _ in range(10):
            best_actor.add(Dense(16, activation='selu', kernel_initializer='he_uniform'))
        best_actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
    
        best_actor._make_predict_function()
        best_actor.summary()
    
        return best_actor
        
    def choice(self, state, dict, ai1_hands, ai1_board, ai2_board, hidden, eval, id):
        calc_deck = []
        result = [0, 0, 0]
        number = 0
        md = [0, 1, 2]
        
        generate_deck(calc_deck)
        #print("start!")
        
        if eval == False:
            with self.graph.as_default():
                policy = self.best_actor.predict(state)[0]
            
            return np.random.choice(self.action_size, 1, p=policy)[0]
        
        for i in ai1_hands + ai1_board + ai2_board:
            calc_deck.remove(i)
            
        for i in list(itertools.combinations(calc_deck, 2)):
            case = []
            for j in i:
                case.append(dict[str(j)])
            
            case.sort()
        
            state[0, 17] = case[0]
            state[0, 18] = case[1]
            
            if hidden == 0:
                with self.graph.as_default():
                    if id == 1:
                        result += self.best_actor.predict(state)[0]
                    elif id == 2:
                        result += self.challenger_actor.predict(state)[0]
                number += 1
                
            elif hidden == 1:
                for k in calc_deck:
                    hid = dict[str(k)]
                    if hid == case[0] or hid == case[1]:
                        continue
                    state[0, 23] = hid
                    with self.graph.as_default():
                        if id == 1:
                            result += self.best_actor.predict(state)[0]
                        elif id == 2:
                            result += self.challenger_actor.predict(state)[0]
                    number += 1
        
        for i in range(3):
            result[i] /= number
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
             
         
    