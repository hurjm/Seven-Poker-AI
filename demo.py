from copy import deepcopy
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from keras import backend as K
import itertools
import random
import time

suit_index_dict = {"c": 0, "h": 1, "d": 2, "s": 3}
suit_index = ("c", "h", "d", "s")
val_string = 'AKQJT98765432'
val_string1 = '23456789'
val_string2 = 'TJQKA'
hand_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                 "Straight", "Back Straight", "Mountain", "Flush", "Full House", "Four of a Kind",
                 "Straight Flush", "Back Straight Flush", "Royal Flush")
value_dict = {"T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
for num in range(2, 10):
    value_dict[str(num)] = num - 2

count = 0

class Card:
    # Takes in strings of the format: "As", "Tc", "6d"
    def __init__(self, card_string):
        value, self.suit = card_string[0], card_string[1]
        self.value = value_dict[value]
        self.suit_index = suit_index_dict[self.suit]

    def __str__(self):
        return val_string[12 - self.value] + self.suit

    def __repr__(self):
        return val_string[12 - self.value] + self.suit
    
    def __hash__(self):
        return (hash(str(self)))

    def __eq__(self, other):
        if self is None:
            return other is None
        elif other is None:
            return False
        return self.value == other.value and self.suit == other.suit

def check_turn(ai1_board, ai2_board):
    ai1_temp, ai2_temp= [0] * 13, [0] * 13
    ai1_histogram, ai2_histogram, ai1_full, ai2_full = [], [], [], []
    result, result_full = [], []
    j = 0
    
    for card in ai1_board:
        ai1_temp[card.value] += 1
        ai1_full.append((card.value, card.suit_index))
    
    for card in ai2_board:
        ai2_temp[card.value] += 1
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

def generate_deck(deck):
    for value in val_string1:
        for suit in suit_index:
            deck.append(Card(value + suit))
    for value in val_string2:
        for suit in suit_index:
            deck.append(Card(value + suit))

# Generate all possible hole card combinations
def generate_hole_cards(deck):
    return itertools.combinations(deck, 2)

# Generate num_iterations random boards
def generate_random_drawn_cards(deck, empty_length):
    random.seed(time.time())
    global count
    for _ in range(10000):
        yield random.sample(deck, empty_length)

def generate_play_cards(drawn_cards, player_empty_length):
    return itertools.combinations(drawn_cards, player_empty_length)

# Generate all possible boards
def generate_exhaustive_cards(deck, empty_length):
    return itertools.combinations(deck, empty_length)

# Returns a board of cards all with suit = flush_index
def generate_suit_board(flat_board, flush_index):
    histogram = [card.value for card in flat_board if card.suit_index == flush_index]
    histogram.sort(reverse=True)
    return histogram

# Returns a list of two tuples of the form: (value of card, frequency of card)
def preprocess(histogram):
    return [(12 - index, frequency) for index, frequency in enumerate(histogram) if frequency]


# Takes an iterable sequence and returns two items in a tuple:
# 1: 4-long list showing how often each card suit appears in the sequence
# 2: 13-long list showing how often each card value appears in the sequence
def preprocess_board(flat_board):
    suit_histogram, histogram, full_histogram = [0] * 4, [0] * 13, []
    # Reversing the order in histogram so in the future, we can traverse
    # starting from index 0
    for card in flat_board:
        histogram[12 - card.value] += 1
        suit_histogram[card.suit_index] += 1
        full_histogram.append((card.value, card.suit_index))
    return suit_histogram, histogram, max(suit_histogram), full_histogram

# Returns tuple: (Is there a straight flush?, high card)
def detect_straight_flush(suit_board):
    contiguous_length, fail_index = 1, len(suit_board) - 5
    # Won't overflow list because we fail fast and check ahead
    for index, elem in enumerate(suit_board):
        current_val, next_val = elem, suit_board[index + 1]
        if next_val == current_val - 1:
            contiguous_length += 1
            if contiguous_length == 5:
                return True, current_val + 3
        else:
            # Fail fast if straight not possible
            if index >= fail_index:
                if (index == fail_index and next_val == 4 and suit_board[0] == 12):
                    return True, 4
                break
            contiguous_length = 1
    return False,

# Returns the highest kicker available
def detect_highest_quad_kicker(histogram_board):
    for elem in histogram_board:
        if elem[1] < 4:
            return elem[0]

# Returns tuple: (Is there a straight?, high card)
def detect_straight(histogram_board):
    contiguous_length, fail_index = 1, len(histogram_board) - 5
    # Won't overflow list because we fail fast and check ahead
    for index, elem in enumerate(histogram_board):
        current_val, next_val = elem[0], histogram_board[index + 1][0]
        if next_val == current_val - 1:
            contiguous_length += 1
            if contiguous_length == 5:
                return True, current_val + 3
        else:
            # Fail fast if straight not possible
            if index >= fail_index:
                if (index == fail_index and next_val == 4 and histogram_board[0][0] == 12):
                    return True, 4
                break
            contiguous_length = 1
    return False,

# Returns tuple of the two highest kickers that result from the three of a kind

def detect_three_of_a_kind_kickers(histogram_board):
    kicker1 = -1
    for elem in histogram_board:
        if elem[1] != 3:
            if kicker1 == -1:
                kicker1 = elem[0]
            else:
                return kicker1, elem[0]

# Returns the highest kicker available
def detect_highest_kicker(histogram_board):
    for elem in histogram_board:
        if elem[1] == 1:
            return elem[0]

# Returns tuple: (kicker1, kicker2, kicker3)
def detect_pair_kickers(histogram_board):
    kicker1, kicker2 = -1, -1
    for elem in histogram_board:
        if elem[1] != 2:
            if kicker1 == -1:
                kicker1 = elem[0]
            elif kicker2 == -1:
                kicker2 = elem[0]
            else:
                return kicker1, kicker2, elem[0]

# Returns a list of the five highest cards in the given board
# Note: Requires a sorted board to be given as an argument
def get_high_cards(histogram_board):
    return histogram_board[:5]

# Return Values:
# Royal Flush: (9,)
# Straight Flush: (8, high card)
# Four of a Kind: (7, quad card, kicker)
# Full House: (6, trips card, pair card)
# Flush: (5, [flush high card, flush second high card, ..., flush low card])
# Straight: (4, high card)
# Three of a Kind: (3, trips card, (kicker high card, kicker low card))
# Two Pair: (2, high pair card, low pair card, kicker)
# Pair: (1, pair card, (kicker high card, kicker med card, kicker low card))
# High Card: (0, [high card, second high card, third high card, etc.])

def detect_high_suit(given_num, full_histogram):
    suit = []
    for i in full_histogram:
        if i[0] == given_num:
            suit.append(i[1])
    return max(suit)

def detect_hand(given_cards, suit_histogram, num_histogram, max_suit, full_histogram):
    # Determine if flush possible. If yes, four of a kind and full house are
    # impossible, so return royal, straight, or regular flush.
    if max_suit >= 5:
        flush_index = suit_histogram.index(max_suit)
        flat_board = list(given_cards)
        suit_board = generate_suit_board(flat_board, flush_index)
        result = detect_straight_flush(suit_board)
        
        if result[0]:
            if result[1] == 12:
                return 12, suit_histogram.index(max_suit)
            elif result[1] == 4:
                return 11, suit_histogram.index(max_suit)
            else:
                return 10, result[1], suit_histogram.index(max_suit)
            
        return 7, get_high_cards(suit_board), suit_histogram.index(max_suit)

    # Add hole cards to histogram data structure and process it
    num_histogram = num_histogram[:]
    histogram_board = preprocess(num_histogram)
    # Find which card value shows up the most and second most times
    current_max, max_val, second_max, second_max_val = 0, 0, 0, 0
    for item in histogram_board:
        val, frequency = item[0], item[1]
        if frequency > current_max:
            second_max, second_max_val = current_max, max_val
            current_max, max_val = frequency, val
        elif frequency > second_max:
            second_max, second_max_val = frequency, val

    # Check to see if there is a four of a kind
    if current_max == 4:
        return 9, max_val, detect_highest_quad_kicker(histogram_board)
    # Check to see if there is a full house
    if current_max == 3 and second_max >= 2:
        return 8, max_val, second_max_val
    # Check to see if there is a straight
    if len(histogram_board) >= 5:
        result = detect_straight(histogram_board)
        if result[0]:
            if result[1] == 12:
                return 6, detect_high_suit(result[1], full_histogram)
            elif result[1] == 4:
                return 5, detect_high_suit(12, full_histogram)
            else:
                return 4, result[1], detect_high_suit(result[1], full_histogram)
    # Check to see if there is a three of a kind
    if current_max == 3:
        return 3, max_val, detect_three_of_a_kind_kickers(histogram_board)
    if current_max == 2:
        # Check to see if there is a two pair
        if second_max == 2:
            return 2, max_val, second_max_val, detect_highest_kicker(histogram_board), detect_high_suit(max_val, full_histogram)
        # Return pair
        else:
            return 1, max_val, detect_pair_kickers(histogram_board), detect_high_suit(max_val, full_histogram)
    # Check for high cards
    return 0, get_high_cards(histogram_board), detect_high_suit(max(max(get_high_cards(histogram_board))), full_histogram)

def detect_hand_for_state(given_cards, suit_histogram, num_histogram, max_suit, full_histogram):
    # Determine if flush possible. If yes, four of a kind and full house are
    # impossible, so return royal, straight, or regular flush.
    if max_suit >= 5:
        flush_index = suit_histogram.index(max_suit)
        flat_board = list(given_cards)
        suit_board = generate_suit_board(flat_board, flush_index)
        result = detect_straight_flush(suit_board)
        
        if result[0]:
            return 36
        
        return 33

    # Add hole cards to histogram data structure and process it
    num_histogram = num_histogram[:]
    histogram_board = preprocess(num_histogram)

    # Find which card value shows up the most and second most times
    current_max, max_val, second_max, second_max_val = 0, 0, 0, 0
    for item in histogram_board:
        val, frequency = item[0], item[1]
        if frequency > current_max:
            second_max, second_max_val = current_max, max_val
            current_max, max_val = frequency, val
        elif frequency > second_max:
            second_max, second_max_val = frequency, val
    # Check to see if there is a four of a kind
    if current_max == 4:
        return 35, max_val
    # Check to see if there is a full house
    if current_max == 3 and second_max >= 2:
        return 34, max_val, second_max_val
    # Check to see if there is a straight
    if len(histogram_board) >= 5:
        result = detect_straight(histogram_board)
        if result[0]:
            return 32
    # Check to see if there is a three of a kind
    if current_max == 3:
        return 31, max_val
    if current_max == 2:
        # Check to see if there is a two pair
        if second_max == 2:
            return max_val + 19, max_val, second_max_val
        # Return pair
        else:
            return max_val + 6, max_val
    # Check for high cards
    top = get_high_cards(histogram_board)
    return 0 + top[0][0], top[0][0]

# Returns the index of the player with the winning hand

def compare_hands(result_list):
    best_hand = max(result_list)
    winning_player_index = result_list.index(best_hand) + 1
    # Check for ties
    if best_hand in result_list[winning_player_index:]:
        return 0
    return winning_player_index

def bet(action, stake, table_stake, id, fund):
    if action == 0:
        #Check
        if stake == 0:
            return stake, table_stake, fund[0], fund[1]
        #Call    
        else:
            if fund[id] < stake:
                table_stake += fund[id]
                fund[id] = 0
            else:
                table_stake += stake
                fund[id] -= stake
    #Raise
    elif action == 1:
        if stake == 0:
            if fund[id] < table_stake/2:
                stake = fund[id]
                fund[id] = 0
            else:
                stake = table_stake/2
                table_stake += stake
                fund[id] -= stake
        else:
            table_stake += stake
            fund[id] -= stake
            if fund[id] < table_stake/2:
                stake = fund[id]
                fund[id] = 0
            else:
                stake = table_stake/2
                fund[id] -= stake
                table_stake += stake
                
    return stake, table_stake, fund[0], fund[1]

def judge_end(ai1_cards, ai2_cards):
    result_list = [None] * 2
    winner_list = [0] * 3
                
    ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram = preprocess_board(ai1_cards)
    ai2_suit_histogram, ai2_histogram, ai2_max_suit, ai2_full_histogram = preprocess_board(ai2_cards)
    
    result_list[0] = detect_hand(ai1_cards, ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram)
    result_list[1] = detect_hand(ai2_cards, ai2_suit_histogram, ai2_histogram, ai2_max_suit, ai2_full_histogram)
    print(result_list)
    winner_index = compare_hands(result_list)
    winner_list[winner_index] += 1
            
    if winner_list[1] == 1:
        winner = 0
    else:
        winner = -1
    return winner

def judge_state(ai1_cards, ai2_cards):
    result_list = [None] * 2
    winner_list = [0] * 3
                
    ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram = preprocess_board(ai1_cards)
    ai2_suit_histogram, ai2_histogram, ai2_max_suit, ai2_full_histogram = preprocess_board(ai2_cards)
    
    result_list[0] = detect_hand(ai1_cards, ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram)
    result_list[1] = detect_hand(ai2_cards, ai2_suit_histogram, ai2_histogram, ai2_max_suit, ai2_full_histogram)
    print(result_list)
    winner_index = compare_hands(result_list)
    winner_list[winner_index] += 1
            
    if winner_list[1] == 1:
        winner = 0
    else:
        winner = -1
    return winner, result_list

def judge_demo(cards):
    ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram = preprocess_board(cards)
    
    return detect_hand(cards, ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram)

high_card = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

form_class = uic.loadUiType("form.ui")[0]

btn = False
my_action = 4
regame = False

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.graph = tf.get_default_graph()
        
        self.actor_model = [self.build_flop_actor_model(), self.build_turn_actor_model()]
        for type in range(2):
            self.actor_model[type].load_weights('actor_' + str(type) + '.h5')
        
        self.setupUi(self)
        self.background.setStyleSheet("border-image: url(./resource/table);")
        
        self.th = Table(self.actor_model, self.sess, self.graph)
        self.th.cards_status.connect(self.cards_status)
        self.th.end.connect(self.end)
        self.th.fund.connect(self.fund)
        self.th.bet_stake.connect(self.bet_stake)
        self.th.init_table.connect(self.init_table)
        self.th.card_range.connect(self.card_range)
        self.th.ai_card_range.connect(self.ai_card_range)
        self.th.judge_winner.connect(self.judge_winner)
        self.th.btn_status.connect(self.btn_status)
        self.th.action_status.connect(self.action_status)
        self.th.regame.connect(self.regame)
        self.th.ai_turn.connect(self.ai_turn)

        self.callorcheck.clicked.connect(self.callorcheck_click)
        self.bet.clicked.connect(self.bet_click)
        self.fold.clicked.connect(self.fold_click)
        self.start.clicked.connect(self.start_click)

    def ai_turn(self, status):
        if status == 0:
            self.label_26.setText('계산하는 중...')
        elif status == 1:
            self.label_26.setText('기다리는 중...')
        elif status == 2:
            self.label_26.setText('기뻐하는 중...')
        elif status == 3:
            self.label_26.setText('슬퍼하는 중...')
    
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
        global btn, regame
        if regame == False:
            self.th.start()
            self.start.setDisabled(True)
            regame = True
        elif regame == True:
            self.start.setDisabled(True)
            btn = True
    
    def regame(self):
        self.start.setText('Next Game')
        self.start.setEnabled(True)
        
    
    @pyqtSlot(int, int, int)
    def fund(self, player_fund, ai_fund, table_stake):
        self.label_15.setText(str(player_fund))
        self.label_16.setText(str(ai_fund))
        self.label_17.setText(str(table_stake))
        
    @pyqtSlot(list, list, list, list)
    def cards_status(self, player_hands, player_board, ai_hands, ai_board):
        
        self.label.setPixmap(QPixmap('./resource/' + str(player_hands[0])))
        self.label_2.setPixmap(QPixmap('./resource/' + str(player_hands[1])))
        
        if player_board:
            self.label_3.setPixmap(QPixmap('./resource/' + str(player_board[0])))
        elif not player_board and len(player_hands) == 3:
            self.label_3.setPixmap(QPixmap('./resource/' + str(player_hands[2])))
        else:
            self.label_3.clear()
        if len(player_board) > 1:
            self.label_4.setPixmap(QPixmap('./resource/' + str(player_board[1])))
        else:
            self.label_4.clear()
        if len(player_board) > 2:
            self.label_5.setPixmap(QPixmap('./resource/' + str(player_board[2])))
        else:
            self.label_5.clear()
        if len(player_board) > 3:
            self.label_6.setPixmap(QPixmap('./resource/' + str(player_board[3])))
        else:
            self.label_6.clear()
        if len(player_hands) > 2 and len(player_board) > 3:
            self.label_7.setPixmap(QPixmap('./resource/' + str(player_hands[2])))
        else:
            self.label_7.clear()
            
        self.label_8.setPixmap(QPixmap('./resource/b2fv'))
        self.label_9.setPixmap(QPixmap('./resource/b2fv'))
        
        if ai_board:
            self.label_10.setPixmap(QPixmap('./resource/' + str(ai_board[0])))
        elif not ai_board and len(ai_hands) == 3:
            self.label_10.setPixmap(QPixmap('./resource/b2fv'))
        else:
            self.label_10.clear()
        if len(ai_board) > 1:
            self.label_11.setPixmap(QPixmap('./resource/' + str(ai_board[1])))
        else:
            self.label_11.clear()
        if len(ai_board) > 2:
            self.label_12.setPixmap(QPixmap('./resource/' + str(ai_board[2])))
        else:
            self.label_12.clear()
        if len(ai_board) > 3:
            self.label_13.setPixmap(QPixmap('./resource/' + str(ai_board[3])))
        else:
            self.label_13.clear()
        if len(ai_hands) > 2 and len(ai_board) > 3:
            self.label_14.setPixmap(QPixmap('./resource/b2fv'))
        else:
            self.label_14.clear()
            
    @pyqtSlot(list, list, list, list)
    def end(self, player_hands, player_board, ai_hands, ai_board):
        
        self.label.setPixmap(QPixmap('./resource/' + str(player_hands[0])))
        self.label_2.setPixmap(QPixmap('./resource/' + str(player_hands[1])))
        
        if player_board:
            self.label_3.setPixmap(QPixmap('./resource/' + str(player_board[0])))
        else:
            self.label_3.clear()
        if len(player_board) > 1:
            self.label_4.setPixmap(QPixmap('./resource/' + str(player_board[1])))
        else:
            self.label_4.clear()
        if len(player_board) > 2:
            self.label_5.setPixmap(QPixmap('./resource/' + str(player_board[2])))
        else:
            self.label_5.clear()
        if len(player_board) > 3:
            self.label_6.setPixmap(QPixmap('./resource/' + str(player_board[3])))
        else:
            self.label_6.clear()
        if len(player_hands) > 2:
            self.label_7.setPixmap(QPixmap('./resource/' + str(player_hands[2])))
        else:
            self.label_7.clear()
        
        self.label_8.setPixmap(QPixmap('./resource/' + str(ai_hands[0])))
        self.label_9.setPixmap(QPixmap('./resource/' + str(ai_hands[1])))
        
        if ai_board:
            self.label_10.setPixmap(QPixmap('./resource/' + str(ai_board[0])))
        else:
            self.label_10.clear()
        if len(ai_board) > 1:
            self.label_11.setPixmap(QPixmap('./resource/' + str(ai_board[1])))
        else:
            self.label_11.clear()
        if len(ai_board) > 2:
            self.label_12.setPixmap(QPixmap('./resource/' + str(ai_board[2])))
        else:
            self.label_12.clear()
        if len(ai_board) > 3:
            self.label_13.setPixmap(QPixmap('./resource/' + str(ai_board[3])))
        else:
            self.label_13.clear()
        if len(ai_hands) > 2:
            self.label_14.setPixmap(QPixmap('./resource/' + str(ai_hands[2])))
        else:
            self.label_14.clear()
    
    def bet_stake(self, table_stake, stake):
        self.label_18.setText(str(stake))
        self.label_19.setText(str(int((table_stake + stake)/2)))
    
    def init_table(self):
        self.label_18.clear()
        self.label_19.clear()
        self.label_21.clear()
        self.label_22.clear()
        self.label_23.clear()
        self.label_24.clear()
        self.label_25.clear()
        self.label_26.clear()
        self.callorcheck.setDisabled(True)
        self.bet.setDisabled(True)
        self.fold.setDisabled(True)
    
    @pyqtSlot(tuple)
    def card_range(self, card_range):
        if card_range[0] == 0:
            self.label_21.setText(high_card[card_range[1][0][0] + 2] + ' High Card')
        elif card_range[0] == 1:
            self.label_21.setText(high_card[card_range[1] + 2] + ' One Pair')
        elif card_range[0] == 2:
            self.label_21.setText(high_card[card_range[1] + 2] + ' ' + high_card[card_range[2] + 2] + ' Two Pair')
        elif card_range[0] == 3:
            self.label_21.setText(high_card[card_range[1] + 2] + ' Three of a Kind')
        elif card_range[0] == 4:
            self.label_21.setText(high_card[card_range[1] + 2] + ' Straight')
        elif card_range[0] == 5:
            self.label_21.setText('Back Straight')
        elif card_range[0] == 6:
            self.label_21.setText('Mountain')
        elif card_range[0] == 7:
            self.label_21.setText(high_card[card_range[1][0] + 2] + ' Flush')
        elif card_range[0] == 8:
            self.label_21.setText(high_card[card_range[1] + 2] + ' ' + high_card[card_range[2] + 2] + ' Full House')
        elif card_range[0] == 9:
            self.label_21.setText(high_card[card_range[1] + 2] + ' Four of a Kind')
        elif card_range[0] == 10:
            self.label_21.setText(high_card[card_range[1] + 2] + ' Straight Flush')
        elif card_range[0] == 11:
            self.label_21.setText('Back Straight Flush')
        elif card_range[0] == 12:
            self.label_21.setText('Royal Straight Flush')
            
    @pyqtSlot(tuple)
    def ai_card_range(self, card_range):
        if card_range[0] == 0:
            self.label_22.setText(high_card[card_range[1][0][0] + 2] + ' High Card')
        elif card_range[0] == 1:
            self.label_22.setText(high_card[card_range[1] + 2] + ' One Pair')
        elif card_range[0] == 2:
            self.label_22.setText(high_card[card_range[1] + 2] + ' ' + high_card[card_range[2]] + ' Two Pair')
        elif card_range[0] == 3:
            self.label_22.setText(high_card[card_range[1] + 2] + ' Three of a Kind')
        elif card_range[0] == 4:
            self.label_22.setText(high_card[card_range[1] + 2] + ' Straight')
        elif card_range[0] == 5:
            self.label_22.setText('Back Straight')
        elif card_range[0] == 6:
            self.label_22.setText('Mountain')
        elif card_range[0] == 7:
            self.label_22.setText(high_card[card_range[1][0][0] + 2] + ' Flush')
        elif card_range[0] == 8:
            self.label_22.setText(high_card[card_range[1] + 2] + ' ' + high_card[card_range[2]] + ' Full House')
        elif card_range[0] == 9:
            self.label_22.setText(high_card[card_range[1] + 2] + ' Four of a Kind')
        elif card_range[0] == 10:
            self.label_22.setText(high_card[card_range[1] + 2] + ' Straight Flush')
        elif card_range[0] == 11:
            self.label_22.setText('Back Straight Flush')
        elif card_range[0] == 12:
            self.label_22.setText('Royal Straight Flush')
            
    def judge_winner(self, winner):
        if winner == 0:
            self.label_23.setText('AI Win!')
        elif winner == -1:
            self.label_23.setText('Player Win!')

    def btn_status(self, btn, my_all_in, ai_all_in):
        if btn:
            if my_all_in or ai_all_in:
                self.callorcheck.setEnabled(btn)
                self.fold.setEnabled(btn)
            else:
                self.callorcheck.setEnabled(btn)
                self.fold.setEnabled(btn)
                self.bet.setEnabled(btn)
        else:
            self.callorcheck.setEnabled(btn)
            self.bet.setEnabled(btn)
            self.fold.setEnabled(btn)
        
    
    def action_status(self, id, action, stake):
        if id == 1:
            self.label_24.clear()
            if action == 0:
                if stake == 0:
                    self.label_25.setText('Check')
                elif stake > 0:
                    self.label_25.setText('Call ' + str(stake))
            elif action == 1:
                self.label_25.setText('Raise ' + str(stake))
            elif action == 2:
                self.label_25.setText('Fold')
        if id == 2:
            self.label_25.clear()
            if action == 0:
                if stake == 0:
                    self.label_24.setText('Check')
                elif stake > 0:
                    self.label_24.setText('Call ' + str(stake))
            elif action == 1:
                self.label_24.setText('Raise ' + str(stake))
            elif action == 2:
                self.label_24.setText('Fold')
                   
    def build_flop_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(200, input_dim=98, activation='relu', kernel_initializer='he_normal'))
        
        for _ in range(10):
            actor.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
            
        actor.add(Dense(2, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor
    
    def build_turn_actor_model(self):
        actor = Sequential()
        
        actor.add(Dense(300, input_dim=144, activation='relu', kernel_initializer='he_normal'))
        for _ in range(10):
            actor.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
        
        actor.add(Dense(2, activation='softmax', kernel_initializer='he_normal'))
        
        actor.summary()

        return actor

class Table(QThread):
    cards_status = pyqtSignal(list, list, list, list)
    end = pyqtSignal(list, list, list, list)
    fund = pyqtSignal(int, int, int)
    bet_stake = pyqtSignal(int, int)
    init_table = pyqtSignal()
    card_range = pyqtSignal(tuple)
    ai_card_range = pyqtSignal(tuple)
    judge_winner = pyqtSignal(int)
    btn_status = pyqtSignal(bool, bool, bool)
    action_status = pyqtSignal(int, int, int)
    regame = pyqtSignal()
    ai_turn = pyqtSignal(int)
    
    def __init__(self, actor_model, sess, graph):
        super().__init__()
        
        self.actor_model = actor_model
        self.sess = sess
        self.graph = graph

    def run(self):
        global btn
        global my_action
        f = open("save", 'r')
        data = f.readlines()
        ai_fund = int(data[0])
        player_fund = int(data[1])
        ai_win = int(data[2])
        player_win = int(data[3])
        total_game = int(data[4])
        f.close()
        
        while True:
            if ai_fund == 0:
                ai_fund = 200000
            if player_fund == 0:
                player_fund = 200000
            self.init_table.emit()
            deck = []
            
            generate_deck(deck)
                
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
                    
            while True :
                #turn1
                for _ in range(3):
                    ai_hands.append(deck.pop())
                    player_hands.append(deck.pop())
                
                self.fund.emit(player_fund, ai_fund, table_stake)
                
                self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                while btn != True:
                    time.sleep(0.1)
                btn = False
                self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                player_action = my_action
                
                self.ai_turn.emit(0)
                #time.sleep(3)
                    #따로
                ai_action = 0
                self.ai_turn.emit(1)
                ai_board.append(ai_hands.pop(ai_action))
                player_board.append(player_hands.pop(player_action))
                
                print('turn2')
                
                #turn2
                for _ in range(2):
                    ai_board.append(deck.pop())
                    player_board.append(deck.pop())
                    
                self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                self.card_range.emit(judge_demo(player_hands + player_board))
                time.sleep(1)
                
                if check_turn(ai_board, player_board) == 1 and ai_fund > 0 and player_fund > 0:
                    self.ai_turn.emit(0)
                    #time.sleep(3)
                    ai_action = self.choice(ai_hands, ai_board, player_board, 0)
                    self.ai_turn.emit(1)
                     
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                    
                    self.action_status.emit(1, ai_action, stake)
                    if(ai_action == 2):
                        winner = 2
                        break
                    
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                    self.bet_stake.emit(table_stake, stake)
                    self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                    player_action = my_action
                    
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                    self.action_status.emit(2, player_action, stake)
                    if(player_action == 2):
                        winner = 1
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    if(player_action == 1):
                        self.ai_turn.emit(0)
                        #time.sleep(3)   
                        ai_action = self.choice(ai_hands, ai_board, player_board, 0)
                        self.ai_turn.emit(1)
                        if ai_action == 1:
                            ai_action = 0
                         
                        stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                        self.action_status.emit(1, ai_action, stake)
                        if(ai_action == 2):
                            winner = 2
                            break
                        self.fund.emit(player_fund, ai_fund, table_stake)
                        
                elif(check_turn(ai_board, player_board) == 2 and ai_fund > 0 and player_fund > 0):
                    
                    self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                    self.bet_stake.emit(table_stake, stake)
                    self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                    player_action = my_action
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                    self.action_status.emit(2, player_action, stake)
                    if(player_action == 2):
                        winner = 1
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    self.ai_turn.emit(0)
                    #time.sleep(3)
                    ai_action = self.choice(ai_hands, ai_board, player_board, 0)
                    if ai_action == 1 and ai_fund < stake or player_fund == 0:
                        ai_action = 0
                    self.ai_turn.emit(1)
                     
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                    self.action_status.emit(1, ai_action, stake)
                    if(ai_action == 2):
                        winner = 2
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    if(ai_action == 1):
                        self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                        self.bet_stake.emit(table_stake, stake)
                        self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                        while btn != True:
                            time.sleep(0.1)
                        btn = False
                        self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                        player_action = my_action
                        stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                        self.action_status.emit(2, player_action, stake)
                        if(player_action == 2):
                            winner = 1
                            break
                        self.fund.emit(player_fund, ai_fund, table_stake)
                        
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror1")
                    print(check_turn(ai_board, player_board))
                    break
                
                print('turn3')
                #turn3
                stake = 0
            
                ai_board.append(deck.pop())
                player_board.append(deck.pop())
                
                self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                self.card_range.emit(judge_demo(player_hands + player_board))
                time.sleep(1)
                
                if(check_turn(ai_board, player_board) == 1 and ai_fund > 0 and player_fund > 0):
                    self.ai_turn.emit(0)
                    #time.sleep(3)
                    ai_action = self.choice(ai_hands, ai_board, player_board, 1)
                    self.ai_turn.emit(1)
                     
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                    self.action_status.emit(1, ai_action, stake)
                    if(ai_action == 2):
                        winner = 2
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                    self.bet_stake.emit(table_stake, stake)
                    self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                    player_action = my_action
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                    self.action_status.emit(2, player_action, stake)
                    if(player_action == 2):
                        winner = 1
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
            
                    if player_action == 1:
                        while True:
                            self.ai_turn.emit(0)
                            #time.sleep(3)
                            ai_action = self.choice(ai_hands, ai_board, player_board, 1)
                            if ai_action == 1 and ai_fund < stake or player_fund == 0:
                                ai_action = 0
                            self.ai_turn.emit(1)
                             
                            stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                            self.action_status.emit(1, ai_action, stake)
                            if(ai_action == 2):
                                winner = 2
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
                            if(ai_action == 0):
                                break
                            
                            self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                            self.bet_stake.emit(table_stake, stake)
                            self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                            while btn != True:
                                time.sleep(0.1)
                            btn = False
                            self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                            player_action = my_action
                            if(player_action == 1):
                                player_action = 0
                            stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                            self.action_status.emit(2, player_action, stake)
                            if(player_action == 2):
                                winner = 1
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
                            if(player_action == 0):
                                break
                        
                
                elif check_turn(ai_board, player_board) == 2 and ai_fund > 0 and player_fund > 0:
                    self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                    self.bet_stake.emit(table_stake, stake)
                    self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                    player_action = my_action
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                    self.action_status.emit(2, player_action, stake)
                    if(player_action == 2):
                        winner = 1
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    self.ai_turn.emit(0)
                    #time.sleep(3)
                    ai_action = self.choice(ai_hands, ai_board, player_board, 1)
                    if ai_action == 1 and ai_fund < stake or player_fund == 0:
                        ai_action = 0
                    self.ai_turn.emit(1)
                     
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                    self.action_status.emit(1, ai_action, stake)
                    if(ai_action == 2):
                        winner = 2
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    if ai_action == 1:
                        while True:
                            self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                            self.bet_stake.emit(table_stake, stake)
                            self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                            while btn != True:
                                time.sleep(0.1)
                            btn = False
                            self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                            player_action = my_action
                            if(player_action == 1):
                                player_action = 0
                            stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                            self.action_status.emit(2, player_action, stake)
                            if(player_action == 2):
                                winner = 1
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
                            if(player_action == 0):
                                break
                            
                            self.ai_turn.emit(0)
                            #time.sleep(3)
                            ai_action = self.choice(ai_hands, ai_board, player_board, 1)
                            if ai_action == 1 and ai_fund < stake or player_fund == 0:
                                ai_action = 0
                            self.ai_turn.emit(1)
                             
                            stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                            self.action_status.emit(1, ai_action, stake)
                            if(ai_action == 2):
                                winner = 2
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
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
                
                self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                self.card_range.emit(judge_demo(player_hands + player_board))
                time.sleep(1)
                
                if(check_turn(ai_board, player_board) == 1 and ai_fund > 0 and player_fund > 0):
                    self.ai_turn.emit(0)
                    #time.sleep(3)
                    ai_action = self.choice(ai_hands, ai_board, player_board, 2)
                    self.ai_turn.emit(1)
                    
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                    self.action_status.emit(1, ai_action, stake)
                    if(ai_action == 2):
                        winner = 2
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                    self.bet_stake.emit(table_stake, stake)
                    self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                    player_action = my_action
                    
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                    self.action_status.emit(2, player_action, stake)
                    if(player_action == 2):
                        winner = 1
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    if player_action == 1:
                        while True:
                            self.ai_turn.emit(0)
                            #time.sleep(3)
                            ai_action = self.choice(ai_hands, ai_board, player_board, 2)
                            if ai_action == 1 and ai_fund < stake or player_fund == 0:
                                ai_action = 0
                            self.ai_turn.emit(1)
                             
                            stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                            self.action_status.emit(1, ai_action, stake)
                            if(ai_action == 2):
                                winner = 2
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
                            if(ai_action == 0):
                                break
                            
                            self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                            self.bet_stake.emit(table_stake, stake)
                            self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                            while btn != True:
                                time.sleep(0.1)
                            btn = False
                            self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                            player_action = my_action
                            
                            if(player_action == 1):
                                player_action = 0
                            stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                            self.action_status.emit(2, player_action, stake)
                            if(player_action == 2):
                                winner = 1
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
                            if(player_action == 0):
                                break
                
                elif check_turn(ai_board, player_board) == 2 and ai_fund > 0 and player_fund > 0:
                    
                    self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                    self.bet_stake.emit(table_stake, stake)
                    self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                    while btn != True:
                        time.sleep(0.1)
                    btn = False
                    self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                    player_action = my_action
                    
                    stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                    self.action_status.emit(2, player_action, stake)
                    if(player_action == 2):
                        winner = 1
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                      
                    self.ai_turn.emit(0)
                    #time.sleep(3)    
                    ai_action = self.choice(ai_hands, ai_board, player_board, 2)
                    if ai_action == 1 and ai_fund < stake or player_fund == 0:
                        ai_action = 0
                    self.ai_turn.emit(1)
                     
                    stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                    self.action_status.emit(1, ai_action, stake)
                    if(ai_action == 2):
                        winner = 2
                        break
                    self.fund.emit(player_fund, ai_fund, table_stake)
                    
                    if ai_action == 1:
                        while True:
                            self.cards_status.emit(player_hands, player_board, ai_hands, ai_board)
                            self.bet_stake.emit(table_stake, stake)
                            self.btn_status.emit(True, player_fund < stake, ai_fund == 0)
                            while btn != True:
                                time.sleep(0.1)
                            btn = False
                            self.btn_status.emit(False, player_fund < stake, ai_fund == 0)
                            player_action = my_action
                            
                            if(player_action == 1):
                                player_action = 0
                            stake, table_stake, ai_fund, player_fund= bet(player_action, stake, table_stake, -1, [ai_fund, player_fund])
                            self.action_status.emit(2, player_action, stake)
                            if(player_action == 2):
                                winner = 1
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
                            if(player_action == 0):
                                break
                            
                            self.ai_turn.emit(0)
                            #time.sleep(3)
                            ai_action = self.choice(ai_hands, ai_board, player_board, 2)
                            if ai_action == 1 and ai_fund < stake or player_fund == 0:
                                ai_action = 0
                            self.ai_turn.emit(1)
                             
                            stake, table_stake, ai_fund, player_fund = bet(ai_action, stake, table_stake, 0, [ai_fund, player_fund])
                            self.action_status.emit(1, ai_action, stake)
                            if(ai_action == 2):
                                winner = 2
                                break
                            self.fund.emit(player_fund, ai_fund, table_stake)
                            
                            if(ai_action == 0):
                                break
                            
                else:
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrror3")
                    print(check_turn(ai_board, player_board))
                    break
                
                if(winner == 0):
                    winner = judge_end(ai_hands + ai_board, player_hands + player_board)
                    
                break
            
            if winner == 0:
                ai_fund += table_stake
                ai_win += 1
            elif winner == -1:
                player_fund += table_stake
                player_win += 1
            total_game += 1
            
            f = open("save", 'w')
            data = '%d \n%d \n%d \n%d \n%d' % (ai_fund, player_fund, ai_win, player_win, total_game)
            f.write(data)
            f.close()
            if winner == 0:
                self.ai_turn.emit(2)
            elif winner == -1:
                self.ai_turn.emit(3)
            
            self.end.emit(player_hands, player_board, ai_hands, ai_board)
            self.ai_card_range.emit(judge_demo(ai_hands + ai_board))
            self.judge_winner.emit(winner)
            self.regame.emit()
            while btn != True:
                    time.sleep(0.1)
            btn = False
                
        
    def choice(self, ai_hands, ai_board, player_board, turn):
        calc_deck = []
        player_hands = [0, 0]
        result = np.zeros(2)
        case_number = 0
        
        generate_deck(calc_deck)
        
        for i in (ai_hands + ai_board + player_board):
            calc_deck.remove(i)
        
        if turn <= 1:
            for i in itertools.combinations(calc_deck, 2):
                #player_hands.append(i[0])
                #player_hands.append(i[1])
                player_hands[0] = i[0]
                player_hands[1] = i[1]
                
                winner, ai_range = judge_state(ai_hands + ai_board, player_hands + player_board)
                
                if ai_range[winner][0] <= 2:
                    ai_state = self.preprocess(winner, ai_range, ai_hands + ai_board, player_hands + player_board, turn)
                    
                    if type(ai_state) == np.ndarray:
                        with self.graph.as_default():
                            result += self.actor_model[turn].predict(ai_state)[0]
                            print(self.actor_model[turn].predict(ai_state)[0])
                else:
                    result[winner] += 1
                    
                case_number += 1
                #player_hands.clear()
                    
        elif turn == 2:
            player_hands = [0, 0, 0]
            sub_calc_deck = deepcopy(calc_deck)
            for i in itertools.combinations(calc_deck, 2):
                #player_hands.append(i[0])
                #player_hands.append(i[1])
                #player_hands.append(i[2])
                player_hands[0] = i[0]
                player_hands[1] = i[1]
                sub_calc_deck.remove(i[0])
                sub_calc_deck.remove(i[1])
                
                for j in sub_calc_deck:
                    player_hands[2] = j
                    result[judge_end(ai_hands + ai_board, player_hands + player_board)] += 1
                    case_number += 1
                    
                sub_calc_deck.append(i[0])
                sub_calc_deck.append(i[1])
                #player_hands.clear()

        print(case_number)
        for i in range(2):
            result[i] /= case_number
        result /= result.sum()
        print(result)
        if result[0] >= 0.6:
            return 1
        #elif result[0] > 0.4 and result[0] < 0.6:
        #emotion    
        elif result[0] > 0.2 and result[0] < 0.6:
            return 0
        elif  result[0] <= 0.2:
            return 2
                
    def preprocess(self, winner, range, ai_card, player_card, turn):
        count = 0
        count2 = 0
        
        if range[winner][0] > 2:
            return False
        
        if turn == 0:
            ai_state = np.zeros(98)
            
            if range[0][0] == 0 and range[1][0] == 0:
                for i in ai_card:
                    for j in player_card:
                        if i.value > j.value:
                            count += 1
                ai_state[count] = 1
                
            elif range[0][0] == 0 and range[1][0] == 1:
                for i in ai_card:
                    if i.value > range[1][1]:
                        count += 1
                ai_state[count + 31] = 1
            
            elif range[0][0] == 0 and range[1][0] == 2:
                for i in ai_card:
                    if i.value > range[1][1]:
                        count += 1
                ai_state[count + 37] = 1   
            
            elif range[0][0] == 1 and range[1][0] == 0:
                for i in player_card:
                    if i.value > range[0][1]:
                        count += 1
                ai_state[count + 43] = 1
                
            elif range[0][0] == 1 and range[1][0] == 1:
                if winner == 0:
                    for i in player_card:
                        if i.value > range[0][1]:
                            count2 += 1
                            for j in ai_card:
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
                    for i in ai_card:
                        if i.value > range[1][1]:
                            count2 += 1
                            for j in player_card:
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
                    for i in ai_card:
                        if i.value > range[0][1]:
                            if i.value > range[1][1]:
                                count += 1
                    ai_state[count + 94] = 1
            else:
                return False
            
            return np.reshape(ai_state, (1, 98))
                        
                        
        elif turn == 1:
            ai_state = np.zeros(144)
            
            if range[0][0] == 0 and range[1][0] == 0:
                for i in ai_card:
                    for j in player_card:
                        if i.value > j.value:
                            count += 1
                ai_state[count] = 1
                
            elif range[0][0] == 0 and range[1][0] == 1:
                for i in ai_card:
                    if i.value > range[1][1]:
                        count += 1
                if count == 0:
                    return False
                ai_state[count + 37 - 1] = 1
            
            elif range[0][0] == 1 and range[1][0] == 0:
                for i in player_card:
                    if i.value > range[0][1]:
                        count += 1
                if count == 0:
                    return False
                ai_state[count + 43 - 1] = 1
            
            elif range[0][0] == 1 and range[1][0] == 1:
                if winner == 0:
                    for i in player_card:
                        if i.value > range[0][1]:
                            count2 += 1
                            for j in ai_card:
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
                    for i in ai_card:
                        if i.value > range[1][1]:
                            count2 += 1
                            for j in player_card:
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
                    for i in ai_card:
                        if i.value != range[0][1]:
                            if i.value > range[1][1]:
                                count += 1
                    if count == 0:
                        return False
                    ai_state[count + 140 - 1] = 1
            else:
                return False
                        
            return np.reshape(ai_state, (1, 144))
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
    
