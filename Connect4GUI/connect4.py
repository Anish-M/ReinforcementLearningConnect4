import random

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from flask import Flask, render_template

BLANK = ' '
AI_PLAYER = 'R'
HUMAN_PLAYER = 'Y'
TRAINING_EPOCHS = 1000
TRAINING_EPSILON = 0.4
REWARD_WIN = 50
REWARD_BLOCKOPPONTHREE = 20
REWARD_MAKETHREE = 10
REWARD_WHENOPPMAKESTHREE = -20
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
                    player.reward(REWARD_WIN, self.board[:])
                    other_player.reward(REWARD_LOSE, self.board[:])
                if winner == player_tickers[1]:
                    player.show_board(self.board[:])
                    print('Winning Color: %s' % player_tickers[1])
                    print('\n %s won!' % other_player.__class__.__name__)
                    other_player.reward(REWARD_WIN, self.board[:])
                    player.reward(REWARD_LOSE, self.board[:])
                else:
                    player.show_board(self.board[:])
                    print('Tie!')
                    player.reward(REWARD_TIE, self.board[:])
                    other_player.reward(REWARD_TIE, self.board[:])
                break

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

            self.first_player_turn = not self.first_player_turn

            move = player.make_move(self.board)
            counter = 42
            if self.board[counter - (7 - move)] == ' ':
                self.board[counter - (7 - move)] = player_tickers[0]
            else:
                while self.board[counter - (7 - move)] != ' ':
                    counter -= 7
                self.board[counter - (7 - move)] = player_tickers[0]

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

            if self.board[0] == player_ticker and self.board[4] == player_ticker and self.board[8] == player_ticker:
                return True, player_ticker

            if self.board[2] == player_ticker and self.board[4] == player_ticker and self.board[6] == player_ticker:
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
