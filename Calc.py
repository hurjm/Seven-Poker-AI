# Constants
import itertools
import random
import time

suit_index_dict = {"c": 0, "h": 1, "d": 2, "s": 3}
suit_index = ("c", "h", "d", "s")
val_string = 'AKQJT98765432'
val_string1 = "23456789"
val_string2 = 'TJQKA'
hand_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind",
                 "Straight", "Back Straight", "Mountain", "Flush", "Full House", "Four of a Kind",
                 "Straight Flush", "Back Straight Flush", "Royal Flush")
suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
for num in range(2, 10):
    suit_value_dict[str(num)] = num
   

count = 0

class Card:
    # Takes in strings of the format: "As", "Tc", "6d"
    def __init__(self, card_string):
        value, self.suit = card_string[0], card_string[1]
        self.value = suit_value_dict[value]
        self.suit_index = suit_index_dict[self.suit]

    def __str__(self):
        return val_string[14 - self.value] + self.suit

    def __repr__(self):
        return val_string[14 - self.value] + self.suit
    
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
    return [(14 - index, frequency) for index, frequency in enumerate(histogram) if frequency]


# Takes an iterable sequence and returns two items in a tuple:
# 1: 4-long list showing how often each card suit appears in the sequence
# 2: 13-long list showing how often each card value appears in the sequence
def preprocess_board(flat_board):
    suit_histogram, histogram, full_histogram = [0] * 4, [0] * 13, []
    # Reversing the order in histogram so in the future, we can traverse
    # starting from index 0
    for card in flat_board:
        histogram[14 - card.value] += 1
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
                if (index == fail_index and next_val == 5 and suit_board[0] == 14):
                    return True, 5
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
                if (index == fail_index and next_val == 5 and histogram_board[0][0] == 14):
                    return True, 5
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
    
    #print(given_board)????????????????????????????????????????????????????????????????????
    
    # Determine if flush possible. If yes, four of a kind and full house are
    # impossible, so return royal, straight, or regular flush.
    if max_suit >= 5:
        flush_index = suit_histogram.index(max_suit)
        flat_board = list(given_cards)
        suit_board = generate_suit_board(flat_board, flush_index)
        result = detect_straight_flush(suit_board)
        
        if result[0]:
            if result[1] == 14:
                return 12, suit_histogram.index(max_suit)
            elif result[1] == 5:
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
            if result[1] == 14:
                return 6, detect_high_suit(result[1], full_histogram)
            elif result[1] == 5:
                return 5, detect_high_suit(14, full_histogram)
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

# Returns the index of the player with the winning hand
def compare_hands(result_list):
    best_hand = max(result_list)
    winning_player_index = result_list.index(best_hand) + 1
    # Check for ties
    if best_hand in result_list[winning_player_index:]:
        return 0
    return winning_player_index

# Print results
def print_results(hole_cards, winner_list, result_histograms):
    float_iterations = float(sum(winner_list))
    print ("Winning Percentages:")
    for index, hole_card in enumerate(hole_cards):
        winning_percentage = float(winner_list[index + 1]) / float_iterations
        if hole_card == (None, None):
            print("(?, ?) : ", winning_percentage)
        else:
            print (hole_card, ": ", winning_percentage)
    print ("Ties: ", float(winner_list[0]) / float_iterations, "\n")
    for player_index, histogram in enumerate(result_histograms):
        print ("Player" + str(player_index + 1) + " Histogram: ")
        for index, elem in enumerate(histogram):
            print (hand_rankings[index], ": ", float(elem) / float_iterations)
        print("\n")    
        
# Returns the winning percentages
def find_winning_percentage(winner_list):
    float_iterations = float(sum(winner_list))
    percentages = []
    for num_wins in winner_list:
        winning_percentage = float(num_wins) / float_iterations
        percentages.append(winning_percentage)
    
    return percentages[1]
'''
# Populate provided data structures with results from simulation
def odds_calc(calc_deck, player_cards, AI_cards, empty_length, player_empty_length, monte):
    # Run simulations
    global count
    result_list = [None] * 2
    result_histograms, winner_list = [], [0] * 3
    
    for _ in range(2):
        result_histograms.append([0] * len(hand_rankings))
    
    if monte == True:
        generate_drawn_cards = generate_random_drawn_cards
    else:
        generate_drawn_cards = generate_exhaustive_cards    
        
    for generated_cards in generate_drawn_cards(calc_deck, empty_length):
        count += 1
        #print(count)
        # Generate a new board
        for player_drawn_cards in generate_play_cards(generated_cards, player_empty_length):
            
            calc_player_cards = player_cards[:]
            calc_AI_cards = AI_cards[:]
            calc_player_cards.extend(player_drawn_cards)
            calc_AI_cards.extend(generated_cards)
            
            for i in player_drawn_cards:
                if i in calc_AI_cards:
                    calc_AI_cards.remove(i)
                    
            player_suit_histogram, player_histogram, player_max_suit, player_full_histogram = (preprocess_board(calc_player_cards))
            AI_suit_histogram, AI_histogram, AI_max_suit, AI_full_histogram = (preprocess_board(calc_AI_cards))
            
            #print(player_full_histogram, AI_full_histogram)
            
            result_list[0] = detect_hand(calc_player_cards, player_suit_histogram, player_histogram, player_max_suit, player_full_histogram)
            result_list[1] = detect_hand(calc_AI_cards, AI_suit_histogram, AI_histogram, AI_max_suit, AI_full_histogram)
            winner_index = compare_hands(result_list)
            winner_list[winner_index] += 1
            
            
            #if result_list[1][0] >= 4:
            #    print(result_list)
                
            for index, result in enumerate(result_list):
                result_histograms[index][result[0]] += 1
            
    if winner_list[1] == 1:
        winner = 'Player'
    else:
        winner = 'AI'
    
    print(result_list)
    
    
    return winner
'''



def bet1(action, stake, table_stake, id, ai1_state, ai2_state, ai1_fund, ai2_fund):
        if action == 0:
            #Check
            if stake == 0:
                if id == 1:
                    ai1_state[0, 8] += 1
                    ai2_state[0, 24] += 1
                    return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
                else:
                    ai2_state[0, 8] += 1
                    ai1_state[0, 24] += 1
                    return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            #Call    
            else:
                if id == 1:
                    ai1_state[0, 9] += 1
                    ai2_state[0, 25] += 1
                    table_stake += stake
                    ai1_fund -= stake
                    return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
                else:
                    ai2_state[0, 9] += 1
                    ai1_state[0, 25] += 1
                    table_stake += stake
                    ai2_fund -= stake
                    return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        #Raise
        elif action == 1:
            if stake == 0:
                if id == 1:
                    ai1_state[0, 10] += 1
                    ai2_state[0, 26] += 1
                    stake = table_stake/2
                    table_stake += stake
                    ai1_fund -= stake
                    return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
                else:
                    ai2_state[0, 10] += 1
                    ai1_state[0, 26] += 1
                    stake = table_stake/2
                    table_stake += stake
                    ai2_fund -= stake
                    return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                if id == 1:
                    ai1_state[0, 10] += 1
                    ai2_state[0, 26] += 1
                    table_stake += stake
                    ai1_fund -= stake
                    stake = table_stake/2
                    ai1_fund -= stake
                    table_stake += stake
                    return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
                else:
                    ai2_state[0, 10] += 1
                    ai1_state[0, 26] += 1
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
                ai2_state[0, 27] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 11] += 1
                ai1_state[0, 27] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        #Call    
        else:
            if id == 1:
                ai1_state[0, 12] += 1
                ai2_state[0, 28] += 1
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 12] += 1
                ai1_state[0, 28] += 1
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
    #Raise
    elif action == 1:
        if stake == 0:
            if id == 1:
                ai1_state[0, 13] += 1
                ai2_state[0, 29] += 1
                stake = table_stake/2
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 13] += 1
                ai1_state[0, 29] += 1
                stake = table_stake/2
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        else:
            if id == 1:
                ai1_state[0, 13] += 1
                ai2_state[0, 29] += 1
                table_stake += stake
                ai1_fund -= stake
                stake = table_stake/2
                ai1_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 13] += 1
                ai1_state[0, 29] += 1
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
                ai2_state[0, 30] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 14] += 1
                ai1_state[0, 30] += 1
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        #Call    
        else:
            if id == 1:
                ai1_state[0, 15] += 1
                ai2_state[0, 31] += 1
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 15] += 1
                ai1_state[0, 31] += 1
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
    #Raise
    elif action == 1:
        if stake == 0:
            if id == 1:
                ai1_state[0, 16] += 1
                ai2_state[0, 32] += 1
                stake = table_stake/2
                table_stake += stake
                ai1_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 16] += 1
                ai1_state[0, 32] += 1
                stake = table_stake/2
                table_stake += stake
                ai2_fund -= stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
        else:
            if id == 1:
                ai1_state[0, 16] += 1
                ai2_state[0, 32] += 1
                table_stake += stake
                ai1_fund -= stake
                stake = table_stake/2
                ai1_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund
            else:
                ai2_state[0, 16] += 1
                ai1_state[0, 32] += 1
                table_stake += stake
                ai2_fund -= stake
                stake = table_stake/2
                ai2_fund -= stake
                table_stake += stake
                return stake, table_stake, ai1_state, ai2_state, ai1_fund, ai2_fund

def judge(ai1_cards, ai2_cards):
    # Run simulations
    global count
    result_list = [None] * 2
    result_histograms, winner_list = [], [0] * 3
    
    
    result_histograms.append([0] * len(hand_rankings))
                
    ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram = preprocess_board(ai1_cards)
    ai2_suit_histogram, ai2_histogram, ai2_max_suit, ai2_full_histogram = preprocess_board(ai2_cards)
    
    #print(player_full_histogram, AI_full_histogram)
    
    result_list[0] = detect_hand(ai1_cards, ai1_suit_histogram, ai1_histogram, ai1_max_suit, ai1_full_histogram)
    result_list[1] = detect_hand(ai2_cards, ai2_suit_histogram, ai2_histogram, ai2_max_suit, ai2_full_histogram)
    winner_index = compare_hands(result_list)
    winner_list[winner_index] += 1
        
        
        #if result_list[1][0] >= 4:
        #    print(result_list)
            
        #for index, result in enumerate(result_list):
        #    result_histograms[index][result[0]] += 1
            
    if winner_list[1] == 1:
        winner = 1
    else:
        winner = 2
        
    return winner