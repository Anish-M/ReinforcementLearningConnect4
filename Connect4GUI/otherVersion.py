import random

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from flask import Flask, render_template

BLANK = ' '
AI_PLAYER = 'R'
HUMAN_PLAYER = 'Y'
TRAINING_EPOCHS = 5
TRAINING_EPSILON = 0.4
REWARD_WIN = 50
REWARD_BLOCKOPPONTHREE = 20
REWARD_MAKETHREE = 10
REWARD_WHENOPPMAKESTHREE = -20
REWARD_LOSE = -100
REWARD_TIE = -50
epochNumber = 100


class Player:

    @staticmethod
    def show_board(board):
        print('|'.join(board[0:7]))
        print('|'.join(board[7:14]))
        print('|'.join(board[14:21]))
        print('|'.join(board[21:28]))
        print('|'.join(board[28:35]))
        print('|'.join(board[35:42]))


class HumanPlayer(Player):

    def reward(self, value, board):
        pass

    def make_move(self, board):

        while True:
            try:
                self.show_board(board)
                move = input('Your next move (cell index 1-7):')
                move = int(move)
                if not (move - 1 in range(7)):
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move - 1


class AIPlayer(Player):
    # previous was 0.4, 0.3, 0.9
    # future reward is more important with gamma discount factor
    def __init__(self, epsilon=0.4, alpha=0.8, gamma=0.99):
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.q = Sequential()
        self.q.add(Dense(32, input_dim=133, activation='relu'))
        self.q.add(Dense(1, activation='relu'))
        self.q.compile(optimizer='adam', loss='mean_squared_error')
        self.move = None
        self.board = (' ',) * 42

    def available_moves(self, board):
        return [i for i in range(7) if board[i] == ' ']

    def encode_input(self, board, action):
        vector_representation = []
        for cell in board:
            for ticker in ['R', 'Y', ' ']:
                if cell == ticker:
                    vector_representation.append(1)
                else:
                    vector_representation.append(0)

        for move in range(7):
            if action == move:
                vector_representation.append(1)
            else:
                vector_representation.append(0)
        return np.array([vector_representation])

    def make_move(self, board):

        self.board = tuple(board)
        actions = self.available_moves(board)
        if random.random() < self.EPSILON:
            self.move = random.choice(actions)
            return self.move
        q_values = [self.get_q(self.board, a) for a in actions]
        max_q_value = max(q_values)

        if q_values.count(max_q_value) > 1:
            best_actions = [i for i in range(len(actions)) if q_values[i] == max_q_value]
            best_move = actions[random.choice(best_actions)]
        else:
            best_move = actions[q_values.index(max_q_value)]

        self.move = best_move
        return self.move

    def get_q(self, state, action):
        return self.q.predict([self.encode_input(state, action)], batch_size=1)

    # first 100 games are completely random moves to fill the replay set
    # at the end of a round, sample a fixed number of observations from the replay set to do the upadte
    # Replay(old_state, new state, action, reward)
    # for a single sample
    # have a nuerla network that takes in state, and has multiple output, one for each of the possible actions
    # feed in old state and store the output q value for the action as prev_q
    # feed in new state and compute the maximum q value for eavh of the 7 actions (columns) and store it as max_q_new

    # max_q_new = max(self.get_q(tuple(board)))
    # max_q_new should only be over the allowable actions

    # this should be done for all the samples together (do the line below for as many times as samples)
    # the action output of old state should be updated so that its equal to the (1-ALPHA) * prev_q + ALPHA*(reward + self.GAMMA * max_q_new)

    # for the reply set reomve the oldest first
    # during the warm up, epsilon should be one
    # after every game ends decrease epsilon a little bit down to a certain threshold ie 0.05
    def reward(self, reward, board):
        if self.move:
            prev_q = self.get_q(self.board, self.move)
            max_q_new = max([self.get_q(tuple(board), a) for a in self.available_moves(self.board)])
            # train the neural network with the new (s,a) and Q value
            self.q.fit(self.encode_input(self.board, self.move),
                       prev_q + self.ALPHA * ((reward + self.GAMMA * max_q_new) - prev_q), epochs=3, verbose=0)

        self.move = None
        self.board = None


import random
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

BLANK = ' '
AI_PLAYER = 'R'
HUMAN_PLAYER = 'Y'
TRAINING_EPOCHS = 100
TRAINING_EPSILON = 0.4
REWARD_WIN = 50
REWARD_BLOCKOPPONTHREE = 20
REWARD_MAKETHREE = 10
REWARD_WHENOPPMAKESTHREE = -20
# Add reward that encourages lesser moves
# reward bonus for how many squares won by
REWARD_LOSE = -100
REWARD_TIE = -50


class Player:

    @staticmethod
    def show_board(board):
        print('|'.join(board[0:7]))
        print('|'.join(board[7:14]))
        print('|'.join(board[14:21]))
        print('|'.join(board[21:28]))
        print('|'.join(board[28:35]))
        print('|'.join(board[35:42]))


class HumanPlayer(Player):

    def reward(self, value, board):
        pass

    def make_move(self, board):

        while True:
            try:
                self.show_board(board)
                move = input('Your next move (cell index 1-7):')
                move = int(move)
                if not (move - 1 in range(7)):
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move - 1


class AIPlayer(Player):

    def game_over_checking(self, player_tickers, board):

        for player_ticker in player_tickers:
            for i in range(0, 6):
                stringRow = ""
                tickerRow = player_ticker * 4
                for j in range(0, 7):
                    stringRow = stringRow + board[(j) + (7 * i)]
                if tickerRow in stringRow:
                    return True, player_ticker

            for i in range(0, 7):
                stringColumn = ""
                tickerColumn = player_ticker * 4
                for j in range(0, 6):
                    stringColumn = stringColumn + board[(i) + (7 * j)]
                if tickerColumn in stringColumn:
                    return True, player_ticker

            # Arrangement 1 for top-right bottom-left diagonals
            if board[3] == player_ticker and board[9] == player_ticker and board[15] == player_ticker and board[
                21] == player_ticker:
                return True, player_ticker

            # Arrangement 2
            if board[4] == player_ticker and board[10] == player_ticker and board[16] == player_ticker and board[
                22] == player_ticker and board[28] == player_ticker and board[28] == player_ticker:
                return True, player_ticker

            # Arrangement 3
            if board[5] == player_ticker and board[11] == player_ticker and board[17] == player_ticker and board[
                23] == player_ticker and board[29] == player_ticker and board[35] == player_ticker:
                return True, player_ticker

            # Arrangement 4
            print(type(board))
            print(board)
            if board[6] == player_ticker and board[12] == player_ticker and board[18] == player_ticker and board[
                24] == player_ticker and board[30] == player_ticker and self.board[36] == player_ticker:
                return True, player_ticker

            # Arrangement 5
            if board[13] == player_ticker and board[19] == player_ticker and board[25] == player_ticker and board[
                31] == player_ticker and board[37] == player_ticker:
                return True, player_ticker

            # Arrangement 6
            if board[20] == player_ticker and board[26] == player_ticker and board[32] == player_ticker and board[
                38] == player_ticker:
                return True, player_ticker

        if board.count(' ') == 0:
            return True, 'T'
        else:
            return False, None

    def __init__(self, epsilon=0.4, alpha=0.8, gamma=0.99):
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.q = Sequential()
        self.previousBoard = ''
        self.other = Sequential()
        self.warmup = 100

        self.q.add(Dense(32, input_dim=133, activation='relu'))
        self.q.add(Dense(32, activation='relu'))
        self.q.add(Dense(1, activation='relu'))
        self.q.compile(optimizer='adam', loss='mean_squared_error')

        self.move = None
        self.board = (' ',) * 42

        self.model = []

    def available_moves(self, board):
        return [i for i in range(7) if board[i] == ' ']

    def encode_input(self, board, action):
        vector_representation = []
        print(type(board))
        for cell in board:
            for ticker in ['R', 'Y', ' ']:
                if cell == ticker:
                    vector_representation.append(1)
                else:
                    vector_representation.append(0)

        for move in range(7):
            if action == move:
                vector_representation.append(1)
            else:
                vector_representation.append(0)

        return np.array([vector_representation])

    def make_move(self, board):

        self.previousBoard = board
        actions = self.available_moves(board)
        print(actions)
        if random.random() < self.EPSILON or epochNumber < self.warmup:
            self.move = random.choice(actions)
            return self.move
        q_values = [self.get_q(self.board, a) for a in actions]

        # max_q_value = np.max(q_values for a in actions)

        # if len(np.where(q_values == max_q_value)) > 1:
        #      best_actions = [i for i in range(len(actions)) if q_values[i] == max_q_value]
        #      best_move = actions[random.choice(best_actions)]
        # else:
        #      best_move = actions[np.where(q_values == max_q_value)]

        max_q_value = np.NINF
        for a in actions:
            # print(type(q_values[a]))
            # print(q_values[a])
            if q_values[a] > max_q_value:
                max_q_value = q_values[a]
                best_move = a

        # best_move = actions[q_values.index(max_q_value)]

        self.move = best_move
        return self.move

    def get_q(self, board, action):
        return self.q.predict(self.encode_input(board, action), batch_size=1)

    # first 100 games are completely random moves to fill the replay set
    # at the end of a round, sample a fixed number of observations from the replay set to do the upadte
    # Replay(old_state, new state, action, reward)
    # for a single sample
    # have a nuerla network that takes in state, and has multiple output, one for each of the possible actions
    # feed in old state and store the output q value for the action as prev_q
    # feed in new state and compute the maximum q value for eavh of the 7 actions (columns) and store it as max_q_new

    # max_q_new = max(self.get_q(tuple(board)))
    # max_q_new should only be over the allowable actions

    # this should be done for all the samples together (do the line below for as many times as samples)
    # the action output of old state should be updated so that its equal to the (1-ALPHA) * prev_q + ALPHA*(reward + self.GAMMA * max_q_new)

    # for the reply set reomve the oldest first
    # during the warm up, epsilon should be one
    # after every game ends decrease epsilon a little bit down to a certain threshold ie 0.05

    def update(self, board, move, player_tickers):

        check, extra = self.game_over_checking(['R', 'Y'], board)

        if check == True:
            self.q.fit(self.encode_input(board, move), np.array([0]), epochs=3, verbose=0)
            return

        prev_q = self.get_q(board, move)

        # construct board after you make move
        board_new = board
        # self.board = list(self.board)
        counter = 42
        if board_new[counter - (7 - move)] == ' ':
            board_new[counter - (7 - move)] = player_tickers[0]
        else:
            while board_new[counter - (7 - move)] != ' ':
                counter -= 7
            board_new[counter - (7 - move)] = player_tickers[0]

        # check if "move" wins the game
        # if it does then
        max_q_new_avg = 0
        reward_avg = 0

        check, extra = self.game_over_checking(['R', 'Y'], board_new)
        if check == True:
            self.q.fit(self.encode_input(board, move), np.array([50]), epochs=3, verbose=0)
            return
        else:
            # for each allowable action a on board_new
            # compute board after taking action a board_new
            for a in self.available_moves(board_new):
                board_opp = board_new
                counter = 42
                if board_opp[counter - (7 - a)] == ' ':
                    board_opp[counter - (7 - a)] = player_tickers[0]
                else:
                    while board_opp[counter - (7 - a)] != ' ':
                        counter -= 7
                    board_opp[counter - (7 - a)] = player_tickers[0]

                # if a wins the game then
                check, extra = self.game_over_checking(['R', 'Y'], board_opp)
                if check == True:
                    max_q_new_avg = max_q_new_avg + 0
                    reward_avg = reward_avg - 50
                    return
                else:
                    max_q_new_avg = max_q_new_avg + max(
                        [self.get_q(board_opp, a) for a in self.available_moves(board_opp)])
                    reward_avg = reward_avg + 0
        # perform the update
        num_allowable_actions = self.available_moves(board_new)

        if len(num_allowable_actions) > 0:
            max_q_new_avg = max_q_new_avg / len(num_allowable_actions)
            reward_avg = reward_avg / len(num_allowable_actions)
        # train the neural network with the new (s,a) and Q value
        self.q.fit(self.encode_input(board, move),
                   np.array([prev_q + self.ALPHA * ((reward_avg + self.GAMMA * max_q_new_avg) - prev_q)]), epochs=3,
                   verbose=0)

        self.move = None
        self.board = None


class Connect4:

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.first_player_turn = random.choice([True, False])
        self.board = [' '] * 42

    def play(self):

        while True:
            if self.first_player_turn:
                player = self.player1
                other_player = self.player2
                player_tickers = (AI_PLAYER, HUMAN_PLAYER)
            else:
                player = self.player2
                other_player = self.player1
                player_tickers = (HUMAN_PLAYER, AI_PLAYER)

            game_over, winner = self.is_game_over(player_tickers)

            if game_over:
                if winner == player_tickers[0]:
                    player.show_board(self.board[:])
                    print('Winning Color: %s' % player_tickers[0])
                    print('\n %s won!' % player.__class__.__name__)

                if winner == player_tickers[1]:
                    player.show_board(self.board[:])
                    print('Winning Color: %s' % player_tickers[1])
                    print('\n %s won!' % other_player.__class__.__name__)

                else:
                    player.show_board(self.board[:])
                    print('Tie!')

                break

            self.first_player_turn = not self.first_player_turn

            move = player.make_move(self.board)



            counter = 42
            if self.board[counter - (7 - move)] == ' ':
                self.board[counter - (7 - move)] = player_tickers[0]
            else:
                while self.board[counter - (7 - move)] != ' ':
                    counter -= 7
                self.board[counter - (7 - move)] = player_tickers[0]

            player.update(self.board[:], move, player_tickers)
            other_player.update(self.board[:], move, player_tickers)

            # three_in_row, who_has_three_in_a_row, startingX, endingX, startingY, endingY = self.threeInARow(player_tickers)
            # if three_in_row:
            #     if who_has_three_in_a_row == player_tickers[0]:
            #         # player.show_board(self.board[:])
            #         # print('\n %s won!' % player.__class__.__name__)
            #         player.reward(REWARD_MAKETHREE, self.board[:])
            #         other_player.reward(REWARD_WHENOPPMAKESTHREE, self.board[:])
            #     if who_has_three_in_a_row == player_tickers[1]:
            #         # player.show_board(self.board[:])
            #         # print('\n %s won!' % other_player.__class__.__name__)
            #         other_player.reward(REWARD_MAKETHREE, self.board[:])
            #         player.reward(REWARD_WHENOPPMAKESTHREE, self.board[:])

    def threeInARow(self, player_tickers):

        for player_ticker in player_tickers:
            startingX = 0;
            startingY = 0;
            endingX = 0;
            endingY = 0;
            for i in range(0, 6):
                stringRow = ""
                tickerRow = player_ticker * 3
                startingX = i
                for j in range(0, 7):
                    startingY = j;
                    stringRow = stringRow + self.board[(j) + (7 * i)]
                if tickerRow in stringRow:
                    startingX = stringRow.find(tickerRow)
                    endingX = startingX + 3
                    endingY = startingY
                    return True, player_ticker, startingX, endingX, startingY, endingY

            for i in range(0, 7):
                stringColumn = ""
                tickerColumn = player_ticker * 3
                startingY = i
                for j in range(0, 6):
                    startingX = j
                    stringColumn = stringColumn + self.board[(i) + (7 * j)]
                if tickerColumn in stringColumn:
                    startingY = stringColumn.find(tickerColumn)
                    endingY = startingY + 3
                    endingX = startingX
                    return True, player_ticker, startingX, endingX, startingY, endingY

        return False, 'B', 0, 0, 0, 0

    def is_game_over(self, player_tickers):
        print(type(self.board))
        for player_ticker in player_tickers:
            for i in range(0, 6):
                stringRow = ""
                tickerRow = player_ticker * 4
                for j in range(0, 7):
                    stringRow = stringRow + self.board[(j) + (7 * i)]
                if tickerRow in stringRow:
                    return True, player_ticker

            for i in range(0, 7):
                stringColumn = ""
                tickerColumn = player_ticker * 4
                for j in range(0, 6):
                    stringColumn = stringColumn + self.board[(i) + (7 * j)]
                if tickerColumn in stringColumn:
                    return True, player_ticker

            # Arrangement 1 for top-right bottom-left diagonals
            if self.board[3] == player_ticker and self.board[9] == player_ticker and self.board[15] == player_ticker and \
                    self.board[21] == player_ticker:
                return True, player_ticker

            # Arrangement 2
            if self.board[4] == player_ticker and self.board[10] == player_ticker and self.board[
                16] == player_ticker and self.board[22] == player_ticker and self.board[28] == player_ticker and \
                    self.board[28] == player_ticker:
                return True, player_ticker

            # Arrangement 3
            if self.board[5] == player_ticker and self.board[11] == player_ticker and self.board[
                17] == player_ticker and self.board[23] == player_ticker and self.board[29] == player_ticker and \
                    self.board[35] == player_ticker:
                return True, player_ticker

            # Arrangement 4
            if self.board[6] == player_ticker and self.board[12] == player_ticker and self.board[
                18] == player_ticker and self.board[24] == player_ticker and self.board[30] == player_ticker and \
                    self.board[36] == player_ticker:
                return True, player_ticker

            # Arrangement 5
            if self.board[13] == player_ticker and self.board[19] == player_ticker and self.board[
                25] == player_ticker and self.board[31] == player_ticker and self.board[37] == player_ticker:
                return True, player_ticker

            # Arrangement 6
            if self.board[20] == player_ticker and self.board[26] == player_ticker and self.board[
                32] == player_ticker and self.board[38] == player_ticker:
                return True, player_ticker

        if self.board.count(' ') == 0:
            return True, None
        else:
            return False, None
#TKinter work--------------------------
currentPage = "NULL"

try:
    import tkinter as tk                # python 3
    from tkinter import font as tkfont  # python 3
except ImportError:
    import Tkinter as tk     # python 2
    import tkFont as tkfont  # python 2




ai_player_1 = AIPlayer()
ai_player_2 = AIPlayer()







class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("800x800")
        # self.wm_attributes("-alpha", 0.5)
        # self.wait_visibility(self)

        # #----------------------------
        #     #Tkinter
        # root = Tk()
        #
        #     # Adjust size
        # root.geometry("800x800")
        #     # Add image file
        #
        #
        #     # Show image using label
        # label1 = Label(root, image=bg)
        # label1.place(x=0, y=0)
        # #--------------------------------
        self.title("Connect 4")
        self.title_font = tkfont.Font(family='Helvetica', size=28, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")


        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def update_clock(self):
        print("UPDATE")


    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


        self.bg = tk.PhotoImage(file="image-3.png")
        label1 = tk.Label(self, image=self.bg)
        label1.place(x=0, y=0)
        currentPage = "START_PAGE"

        self.controller = controller
        label = tk.Label(self, text="Welcome to Connect 4!", font=controller.title_font)
        label.pack(side="top", pady=10)

        button1 = tk.Button(self, text="Train the Agent",
                            command=lambda: controller.show_frame("PageOne"))




        button2 = tk.Button(self, text="Play the Game",
                            command=lambda: controller.show_frame("PageTwo"))
        button1.pack()
        button2.pack()



class PageOne(tk.Frame):

    def updateScreen(self, redList, yellowList):
        #board placed 75, 200
        # column 0, 1, 2, 3, 4, 5, 6 = 90, 180, 269.166, 358, 448, 537, 629
        # row 0, 1, 2, 3, 4, 5 = 210, 300, 390, 480, 570,660

        print("UPDATE SCREEN HAS BEEN REACHED")


        if self.runningCount == 0:
            redLabels = []
            yellowLabels = []

        redLabels = self.oldRedLabels
        yellowLabels = self.oldYellowLabels


        for a in redLabels:
            a.destroy()

        for b in yellowLabels:
            b.destroy()

        redLabels = []
        yellowLabels=[]
        for a in redList:
           redLabels.append(tk.Label(self, image=self.red))

        for b in yellowList:
            yellowLabels.append(tk.Label(self, image=self.yellow))


        self.oldRedLabels = redLabels
        self.oldYellowLabels = yellowLabels

        count = 0
        for a in yellowLabels:
            yellowColumn = -10
            if yellowList[count] < 7:
                yellowColumn = yellowList[count]
            else:
                yellowColumn = yellowList[count] % 7

            yellowRow = int(yellowList[count] / 7)
            # print("YELLOW ROW: ")
            # print(yellowRow)
            # print("YELLOW COLUMN")
            # print(yellowColumn)

            a.place(x = (90 +(90*yellowColumn)), y = (210 + (90 * yellowRow)))
            count = count + 1


        count = 0
        for b in redLabels:
            redColumn = 0
            if redList[count] < 7:
                redColumn = redList[count]
            else:
                redColumn = redList[count] % 7
            redRow = int(redList[count] / 7)
            b.place(x = (90 +(90*redColumn)), y = (210 + (90 * redRow)))
            count = count + 1
        self.runningCount = self.runningCount + 1
        self.update()


    def train_agent(self):
        ai_player_1 = AIPlayer()
        ai_player_2 = AIPlayer()

        print('Training')
        ai_player_1.EPSILON = TRAINING_EPSILON
        ai_player_2.EPSILON = TRAINING_EPSILON

        epochNumber = 0
        for _ in range(TRAINING_EPOCHS):
            print(epochNumber)
            epochNumber = epochNumber + 1
            game = Connect4(ai_player_1, ai_player_2)
            game.play()
            redList = []
            yellowList = []
            count = -1
            for a in game.board:
                count = count + 1
                if a == 'R':
                    redList.append(count)
                if a == 'Y':
                    yellowList.append(count)
            print(redList)
            self.updateScreen(redList, yellowList)

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.runningCount = 0

        self.oldRedLabels = []
        self.oldYellowLabels=[]

        self.bg = tk.PhotoImage(file="image-3.png")
        label1 = tk.Label(self, image=self.bg)
        label1.place(x=0, y=0)

        self.pic = tk.PhotoImage(file="Connect4_Empty.PNG")
        board = tk.Label(self, image=self.pic)
        board.place(x=75, y=200)

        self.yellow = tk.PhotoImage(file="SmallYellowCircle.png")
        self.red = tk.PhotoImage(file="SmallRedCircle.png")

        self.controller = controller
        label = tk.Label(self, text="Agent Training", font=controller.title_font)
        label.pack(side="top", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: [controller.show_frame("StartPage"), self.updateVariableScreen()])

        buttonTrain = tk.Button(self, text="Start Training",
                           command=lambda: self.train_agent())

        # status = tk.Label(root, text="Working")

        button.pack()
        buttonTrain.pack()




class PageTwo(tk.Frame):

    def updateScreen(self, redList, yellowList):
        # board placed 75, 200
        # column 0, 1, 2, 3, 4, 5, 6 = 90, 180, 269.166, 358, 448, 537, 629
        # row 0, 1, 2, 3, 4, 5 = 210, 300, 390, 480, 570,660

        print("UPDATE SCREEN HAS BEEN REACHED")

        if self.runningCount == 0:
            redLabels = []
            yellowLabels = []

        redLabels = self.oldRedLabels
        yellowLabels = self.oldYellowLabels

        for a in redLabels:
            a.destroy()

        for b in yellowLabels:
            b.destroy()

        redLabels = []
        yellowLabels = []
        for a in redList:
            redLabels.append(tk.Label(self, image=self.red))

        for b in yellowList:
            yellowLabels.append(tk.Label(self, image=self.yellow))

        self.oldRedLabels = redLabels
        self.oldYellowLabels = yellowLabels

        count = 0
        for a in yellowLabels:
            yellowColumn = -10
            if yellowList[count] < 7:
                yellowColumn = yellowList[count]
            else:
                yellowColumn = yellowList[count] % 7

            yellowRow = int(yellowList[count] / 7)
            # print("YELLOW ROW: ")
            # print(yellowRow)
            # print("YELLOW COLUMN")
            # print(yellowColumn)

            a.place(x=(90 + (90 * yellowColumn)), y=(210 + (90 * yellowRow)))
            count = count + 1

        count = 0
        for b in redLabels:
            redColumn = 0
            if redList[count] < 7:
                redColumn = redList[count]
            else:
                redColumn = redList[count] % 7
            redRow = int(redList[count] / 7)
            b.place(x=(90 + (90 * redColumn)), y=(210 + (90 * redRow)))
            count = count + 1
        self.runningCount = self.runningCount + 1
        self.update()

    def play_agent(self):
        print('\nTraining is Done')
        x = 0
        while x < 1:
            ai_player_1.EPSILON = 0
            human_player = HumanPlayer()
            game = Connect4(ai_player_1, human_player)
            game.play()
            redList = []
            yellowList = []
            count = -1
            for a in game.board:
                count = count + 1
                if a == 'R':
                    redList.append(count)
                if a == 'Y':
                    yellowList.append(count)
            print(redList)
            self.updateScreen(redList, yellowList)
            x = x + 1

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.bg = tk.PhotoImage(file="image-3.png")
        label1 = tk.Label(self, image=self.bg)
        label1.place(x=0, y=0)

        self.runningCount = 0
        self.oldRedLabels = []
        self.oldYellowLabels = []
        self.yellow = tk.PhotoImage(file="SmallYellowCircle.png")
        self.red = tk.PhotoImage(file="SmallRedCircle.png")

        self.pic = tk.PhotoImage(file="Connect4_Empty.PNG")
        board = tk.Label(self, image=self.pic)
        board.place(x=75, y=200)

        currentPage = "TEST_PAGE"

        self.controller = controller
        label = tk.Label(self, text="Playing the Agent", font=controller.title_font)
        label.pack(side="top", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: [controller.show_frame("StartPage"), self.updateVariableScreen()])
        buttonPlay = tk.Button(self, text="Start Playing",
                                command=lambda: self.play_agent())
        button.pack()
        buttonPlay.pack()



if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
