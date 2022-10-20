import numpy as np #importing numpy to use for making arrays and generating random numbers
import tkinter as tk #importing tkinter for the GUI

try:
    import pygame #importing pygame to play sounds. 
except:
    print("Install pygame by typing 'pip install pygame' in the terminal")
    # If this library is not already installed on the computer, you can install it via the terminal

#The Game class, in which everything for the game is contained
class Game(tk.Frame):
    
    # The __init__ method initialises everything needed for the game, e.g. the GUI
    def __init__(self):
        
        #A dictionary with all the colours of the tiles, so that they can easily be accessed
        self.COLOURS = {
            "background": "#c4c4c4",
            "empty": "#9b9ea2",
            1: "#ffe1ff",
            2: "#d8bfd8",
            3: "#ffbbff",
            5: "#e6e6fa",
            8: "#ffc0cb",
            13: "#fff0f5",
            21: "#eea9b8",
            34: "#87ceeb",
            55: "#b0e2ff",
            89: "#cae1ff",
            144: "#b0e0e6",
            233: "#f5fffa",
            377: "#7fffd4",
            610: "#ffefd5",
            987: "#ffdab9",
            1597: "#eecbad",
            2584: "#ffa07a"   
        }
        
        # To be able to check whether two numbers are consecutive Fibonacci numbers, we made a dictionary. This allows us to look up a number 
        # and check whether the neighbouring numbers match the value for that specific key.
        self.consecutive_num={1:2, 2:3, 3:5, 5:8, 8:13, 13:21, 21:34, 34:55, 55:89, 89:144, 144:233, 233:377, 377:610, 610:987, 987:1597, 1597:2584}
        
        try:
            pygame.mixer.init() # Initialises the mixer that allows sound to be played
        except:
            print("Install pygame by typing 'pip install pygame' in the terminal")
        
        #The GUI is initialised as a 600x600 window with a grid and the title '2584'
        tk.Frame.__init__(self)
        self.grid()
        self.master.title("2584")
        self.main_grid = tk.Frame(self, bg=self.COLOURS["background"], width = 600, height = 600)
        self.main_grid.grid(pady=(130,0))
        self.setup_GUI()
        self.start_board()
        self.game_message = ""
        lbl_frame = tk.Frame(self)
        lbl_frame.place(relx=0.8, y=10)
        self.lbl = tk.Label(lbl_frame, text=self.game_message)
        self.lbl.grid()

        
        #Binds all the keys to their corresponding methods
        self.master.bind("<Left>", self.move_left_little_square)
        self.master.bind("<Right>", self.move_right_little_square)
        self.master.bind("<Up>", self.move_up_little_square)
        self.master.bind("<Down>", self.move_down_little_square)
        self.master.bind("<a>", self.leftswipeA)
        self.master.bind("<s>", self.downswipeS)
        self.master.bind("<w>", self.upswipeW)
        self.master.bind("<d>", self.rightswipeD)
        
        self.mainloop() # Starts the main loop in which the code runs

    # This method makes use of the pygame library to play a sound when tiles are fused together.
    def play_fusion_sound(self):
        try:
            # Try-except block to check that the sound file is available
            pygame.mixer.music.load("fusion-sound.wav")
            pygame.mixer.music.play(loops=0)
        except:
            print("The 'fusion-sound.wav' cannot be found. Make sure you have it downloaded and saved under that name!")
    
    # This method makes use of the pygame library to play a sound when the player wins the game.    
    def play_winning_sound(self):
        try:
            pygame.mixer.music.load("win-sound.wav")
            pygame.mixer.music.play(loops=0)
        except:
            print("The 'win-sound.wav' cannot be found. Make sure you have it downloaded and saved under that name!")
      
    # This method makes use of the pygame library to play a sound when the player loses the game.    
    def play_losing_sound(self):
        try:
            pygame.mixer.music.load("loss-sound.mp3")
            pygame.mixer.music.play(loops=0)
        except:
            print("The 'loss-sound.mp3' cannot be found. Make sure you have it downloaded and saved under that name!")
            
            
    # This method initialises a start board with two randomly placed tiles.    
    def start_board(self):
        # Creates a list of four 2x2 boards filled with zeros
        self.listOfBoards = [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))]
        # Generates two random boards in which the first two tiles should be placed
        randomTile1 = np.random.randint(4)
        randomTile2 = np.random.randint(4)
        self.add_tile(self.listOfBoards[randomTile1], randomTile1)
        self.add_tile(self.listOfBoards[randomTile2], randomTile2)
        # The big board is composed of the four smaller boards using the put_together_big_board() method
        self.board = self.put_together_big_board()
        self.score = 0 #the starting score is zero.
        self.GUI_update()
        try: #we try to find if there is a previous highscore by opening the highscore file. If there is none, the highscore is set as zero. Otherwise, the highscore of the file is taken.
            file = open('highscore.txt', 'r')
            self.highscore = int(file.readline())
            file.close()
        except:
            self.highscore = 0
        self.update_highscore()
        self.label_highscore.configure(text=int(self.highscore)) #the highscore in the GUI is updated.
        
    #This method is used for when the newgame button is clicked. It updates the high score and then calls the function start board initialise a new board.
    def new_game(self):
        self.game_message = ""
        self.lbl.configure(text=self.game_message)
        self.update_highscore()
        self.start_board()
        
    #This method sets up the grid and initialises the empty squares. 
    #The idea of making the cell data a dictionary was taken from https://www.youtube.com/watch?v=b4XP2IcI-Bg&t=318s 
    def setup_GUI(self):
        self.cells = [] # a list of all cells is initialised
        for i in range(4):
            row = [] # a list for the data of this specific row is initialised
            for j in range(4):
                cell = tk.Frame(self.main_grid, bg=self.COLOURS["empty"], width=150,height=150) # creates the actual cell and initialises the colour as 'empty'
                cell.grid(row=i, column=j, padx=5, pady=5) # places the cell in the grid
                cell_number=tk.Label(self.main_grid, font = ('Arial', 25), bg=self.COLOURS["empty"]) #initialises the number of the cell
                cell_number.grid(row=i, column=j)
                cell_data = {"frame":cell, "number":cell_number} # places the data of this cell into a dictionary so that it can be referenced later
                row.append(cell_data) # appends the cell data to the row, so that in the end there is all the data for the cells in that row
            self.cells.append(row) # appends the row data to the cells list, so that this contains all data for all cells
            frame_title = tk.Frame(self) #create a frame for the instructions of the game.
            frame_title.place(relx=0.5, y=20, anchor="center") #it is positioned at the center of the top of the window, and anchored at the center of the screen.
            tk.Label(
                frame_title, #add label text, which includes the instructions.
                text="2584",
                font=('Arial', 40)
                ).grid()
            frame_instructions = tk.Frame(self) #create a frame for the instructions of the game.
            frame_instructions.place(relx=0.5, y=70, anchor="center") #it is positioned at the center of the top of the window, and anchored at the center of the screen.
            tk.Label(
                frame_instructions, #add label text, which includes the instructions.
                text="Welcome to 2584! Match the corresponding Fibonacci number! \n Use ASWD keys to move the complete board and use \n the arrow keys to move the small boards. Reach 2484 to win!"
                ).grid()
            frame_score = tk.Frame(self) #here we create the frame to save the score. This is another frame variable.
            frame_score.place(relx=0.1, y=95, anchor="center") #it is positioned at the center of the top of the window, and anchored at the center of the screen.
            tk.Label(
                frame_score, #add a label name and place it above the number.
                text="Score"
                ).grid()
            self.label_score = tk.Label(frame_score, text="0") #display score, and show it beneath the score label.
            self.label_score.grid()
            frame_highscore = tk.Frame(self) #here we create the frame to show the high score. This is another frame variable.
            frame_highscore.place(relx=0.9, y=95, anchor="center") #it is positioned at the center of the top of the window, and anchored at the center of the screen.
            tk.Label(
                frame_highscore, #add a label name and place it above the number.
                text="High score"
                ).grid() #show it in the grid
            self.label_highscore = tk.Label(frame_highscore, text="0") #display score, and show it beneath the score label.
            self.label_highscore.grid() #show it in the grid
            btn_newgame = tk.Button(self, text="New Game", command=self.new_game) #Create a button to create a new game. It uses the function new_game to save highscore and intialise a new board.
            btn_newgame.place(relx=0.5, y=110, anchor="center") #Position the button.
    
    # This method randomly generates a new tile that is either 1, 2 or 3.
    def new_tile(self):
        # To distribute the occurence of the numbers unequally, a random number between 0 and 1 is generated
        prob = np.random.random_sample() 
        # If the number is less than 0.5, a 1 is placed on the board, so a 1 is created 50% of the time
        if prob <= 0.5:
            return 1
        # If the number is between 0.5 and 0.85, a 2 is placed on the board, so a 2 is created 35% of the time
        elif prob <= 0.85:
            return 2
        # If the number is betwen 0.85 and 1, a 3 is placed, so a 3 occurs 15% of the time
        else:
            return 3
        
    # The add_tile method takes a board and the number of that board (possible values are 0, 1, 2 and 3) and adds a tile in that board. 
    def add_tile(self, board, boardNumber = 0):
        a = 2 #Sets the range variably, so that this method works for both the 2x2 and 4x4 squares.
        if len(board[0]) == 4:
            a = 4
        
        # Returns if there is no space to add a new tile to prevent an infinite while loop.
        if self.board_is_full(board):
            return
        
        # The tile needs to be randomly placed, so a random row and random column is generated
        randomRow = np.random.randint(0,a)
        randomCol = np.random.randint(0,a)
        
        # New values are generated until the row and column point to an empty cell
        while board[randomRow][randomCol] != 0:
            randomRow = np.random.randint(0,a)
            randomCol = np.random.randint(0,a)
        
        # The value of the new tile is generated using the new_tile() method    
        number = self.new_tile()
        
        # Depending on the board number, the row and column values need to be adjusted to correspond with the correct cell in the big board
        if boardNumber == 0:
            # Board 0 is the upper left board, so the column and row values don't need adjustment
            r = 0
            c = 0
        elif boardNumber == 1:
            # Board 1 is the upper right board, so the column numbers need to be shifted over by 2
            r = 0
            c = 2
        elif boardNumber == 2:
            # Board 2 is the lower left board, so the row needs to be shifted by 2
            r = 2
            c = 0
        else:
            # The remaining board, board 3 is the lower right board, so both the column and the row need to be adjusted by 2
            r = 2
            c = 2
          
        # The cell in the GUI is updated using the cells list and then the cell_data dictionary within that list    
        self.cells[randomRow + r][randomCol + c]["frame"].configure(bg=self.COLOURS[number])
        self.cells[randomRow + r][randomCol + c]["number"].configure(bg=self.COLOURS[number], text=str(number))
        # The board is also updated - note that no cell or row adjusment is needed here, because 'board' is the small 2x2 board, rather than the bigger 4x4 board
        board[randomRow][randomCol] = number
    
    # Checks if the inputted board is full or not.   
    def board_is_full(self, board):
        boolean = True
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    boolean = False
        return boolean
                
        
    # This method combines the four smaller boards into the bigger 4x4 board using np.hstack and np.vstack    
    def put_together_big_board(self):
        return np.vstack((np.hstack((self.listOfBoards[0], self.listOfBoards[1])), 
                          np.hstack((self.listOfBoards[2], self.listOfBoards[3]))))
        
    # This method moves all the tiles in the little square to the left and is called when the left arrow key is pressed.
    def move_left_little_square(self, event):
        if not self.board_is_full(self.board): # If the entire board is not yet full, that means there is space to add a new tile.
            randomBoard = np.random.randint(4) # a random board is chosen to add a tile
            while self.board_is_full(self.listOfBoards[randomBoard]): # If the randomly chosen board is full, another board needs to be chosen.
                randomBoard = np.random.randint(4)
        else:
            randomBoard = -1 # If the 4x4 board is completely full, the randomBoard is set to -1 to never make the upcoming if statement true and thus never add a tile.
        for i, board in enumerate(self.listOfBoards):
            board = self.stack(board) # The stack method is called to push all non-zero tiles to the left
            board = self.fusion(board)
            board = self.stack(board) # After potential merges, the stack method is called again
            if i == randomBoard:
                # If the current board is the board in which a new tile should be added, the add_tile method is called
                self.add_tile(board, i)
            self.listOfBoards[i] = board # The board is updated in the listOfBoards
        
        self.board = self.put_together_big_board() # The 4x4 board is updated with the new 2x2 boards
        self.GUI_update() # The GUI is updated
        self.game_over()
    
    # This method moves all the tiles in the little square to the right and is called when the right arrow key is pressed.
    def move_right_little_square(self, event):
        if not self.board_is_full(self.board):
            randomBoard = np.random.randint(4) # a random board is chosen to add a tile
            while self.board_is_full(self.listOfBoards[randomBoard]):
                randomBoard = np.random.randint(4)
        else:
            randomBoard = -1
        for i, board in enumerate(self.listOfBoards):
            board = self.reverse(board) # Since the stack method pushes everything to the left, the board needs to be reversed first
            board = self.stack(board) # Everything in the now flipped board is pushed to the left
            board = self.fusion(board)
            board = self.stack(board) # After merging, everything is pushed to the left again
            board = self.reverse(board) # The board needs to be reversed back, so that everything is now on the right
            if i == randomBoard:
                # If the current board is the board in which a new tile should be added, the add_tile method is called
                self.add_tile(board, i)
            self.listOfBoards[i] = board # The board is updated in the listOfBoards
        self.board = self.put_together_big_board() # The 4x4 board is updated with the new 2x2 boards
        self.GUI_update() # The GUI is updated
        self.game_over()
    
    # This method moves all the tiles in the little square down and is called when the down arrow key is pressed
    def move_down_little_square(self, event):
        if not self.board_is_full(self.board):
            randomBoard = np.random.randint(4) # a random board is chosen to add a tile
            while self.board_is_full(self.listOfBoards[randomBoard]):
                randomBoard = np.random.randint(4)
        else:
            randomBoard = -1
        for i, board in enumerate(self.listOfBoards):
            board = self.transpose(board) # The matrix is transposed, i.e. the columns and rows are swapped
            board = self.reverse(board) # The now transposed matrix is reversed
            board = self.stack(board)  # Everything is pushed to the left
            board = self.fusion(board)
            board = self.stack(board) # After potential merging, everything is pushed to the left again
            board = self.reverse(board) # The matrix is reversed again, so that everything is now on the right
            board = self.transpose(board) # The matrix is transposed, so that everything is at the bottom of the boards
            if i == randomBoard:
                self.add_tile(board, i)
            self.listOfBoards[i] = board
        self.board = self.put_together_big_board()
        self.GUI_update()
        self.game_over()
    
    # This method moves all the tiles in the little square up and is called when the up arrow key is pressed.
    def move_up_little_square(self, event):
        if not self.board_is_full(self.board):
            randomBoard = np.random.randint(4) # a random board is chosen to add a tile
            while self.board_is_full(self.listOfBoards[randomBoard]):
                randomBoard = np.random.randint(4)
        else:
            randomBoard = -1
        for i, board in enumerate(self.listOfBoards):
            board = self.transpose(board) # The board is transposed
            board = self.stack(board) # Everything is pushed to the left
            board = self.fusion(board)
            board = self.stack(board) # After possible merges, everything is pushed left again
            board = self.transpose(board) # The board is transposed back, so that everything is now at the top of the boards
            if i == randomBoard:
                self.add_tile(board, i)
            self.listOfBoards[i] = board
        self.board = self.put_together_big_board()
        self.GUI_update()
        self.game_over()
        
    # This method pushes all non-zero tiles to the left of the board
    # This method was adapted from https://techvidvan.com/tutorials/python-2048-game-project-with-source-code/ 
    def stack(self, board):
        a = 2
        if len(board) == 4:
            a = 4
        matrix = np.zeros((a,a)) # a 2x2 matrix filled with zeros is created
        for i in range(a):
            position_fill = 0 # position_fill keeps track of the next column that needs to be filled
            for j in range(a):
                # In every row, the columns are looped through
                if board[i][j] != 0:
                    # If the board has a non-empty tile, that value is written into 'matrix' at the first unfilled position (established by position_fill)
                    matrix[i][position_fill] = board[i][j]
                    position_fill += 1 # position_fill is updated
        return matrix # the matrix is returned, so that it can overwrite the board where it is called
 
    # This method flips the board along the left-right axis
    # This method was adapted from https://techvidvan.com/tutorials/python-2048-game-project-with-source-code/
    def reverse(self, board):
        a = 2
        if len(board) == 4:
            a = 4
        matrix = [] # an empty list is created
        for i in range(a):
            matrix.append([]) # an empty list is appended, creating a nested list
            for j in range(a):
                # The columns of the board are traversed from right to left and appended in that direction, thereby flipping the board
                matrix[i].append(board[i][a-1-j])
        return matrix # the matrix is returned, so that it can overwrite the board where it is called
 
    # This method transposes the board, so it flips the rows with the column and vice versa
    # This method was adapted from https://techvidvan.com/tutorials/python-2048-game-project-with-source-code/
    def transpose(self, board):
        a = 2
        if len(board) == 4:
            a = 4
        matrix = np.zeros((a,a))
        # The nested for loop loops through all rows and column
        for i in range(a):
            for j in range(a):
                matrix[i][j] = board[j][i] # At every position, the row and column is flipped
        return matrix # the matrix is returned, so that it can overwrite the board where it is called
            
    # This method updates the GUI after the 4x4 board has been changed.
    def GUI_update(self):
        for i in range(4):
            for j in range(4):
                # The cell_value is the actual number in the cell, which can just be read from the board
                cell_value = int(self.board[i][j])
                if cell_value == 0:
                    # If the cell_value is 0, the cell is considered empty and the frame and number are set to the empty colour with an empty string
                    self.cells[i][j]["frame"].configure(bg=self.COLOURS["empty"])
                    self.cells[i][j]["number"].configure(bg=self.COLOURS["empty"], text="")
                else:
                    # If the cell contains an actual number, the tile is coloured in the corresponding colour and the text is updated to the number
                    self.cells[i][j]["frame"].configure(bg=self.COLOURS[cell_value])
                    self.cells[i][j]["number"].configure(bg=self.COLOURS[cell_value], text=str(cell_value))
        self.label_score.configure(text=int(self.score)) #the score in the GUI is updated.

    # This method moves the tiles in the 4x4 square to the left.
    def leftswipeA(self, event): #create a function, for the left swipe across the board. 
        new_board = self.stack(self.board) #compress non zero numbers to the left of the matrix.
        new_board = self.fusion(new_board) # fuse the adjacent numbers.
        new_board = self.stack(new_board) #get rid of all newly created zeros.
        self.add_tile(new_board) #add a new tile.
        self.board = new_board 
        self.write_big_board_into_little_boards() #rewrite the board into a complete board and a small board that also can be used.
        self.GUI_update() #update the GUI accordingly.
        self.game_over() #check if game is over.

    # This method moves the tiles in the 4x4 square to the right.
    def rightswipeD(self, event):
        new_board = self.reverse(self.board) #reverse function to transform the right swipe into left swipem then we do the same things as in the left swipe.
        new_board = self.stack(new_board) #compress non zero numbers to the left of the matrix.
        new_board = self.fusion(new_board) # fuse the adjacent numbers.
        new_board = self.stack(new_board) #get rid of all newly created zeros.
        new_board = self.reverse(new_board) #reverse board back to original orientation.
        self.add_tile(new_board) #add a new tile
        self.board = new_board
        self.write_big_board_into_little_boards() #rewrite the board into a complete board and a small board that also can be used.
        self.GUI_update() #update the GUI accordingly.
        self.game_over() #check if game is over.
    
    # This method moves the tiles in the 4x4 square upwards.
    def upswipeW(self, event):
        new_board = self.transpose(self.board) #transpose board so it works the same as a left swipe.
        new_board = self.stack(new_board) #compress non zero numbers to the left of the matrix.
        new_board = self.fusion(new_board) # fuse the adjacent numbers.
        new_board = self.stack(new_board) #get rid of all newly created zeros.
        new_board = self.transpose(new_board) #transpose board back to original orientation.
        self.add_tile(new_board) #add a new tile.
        self.board = new_board
        self.write_big_board_into_little_boards() #rewrite the board into a complete board and a small board that also can be used.
        self.GUI_update() #update the GUI accordingly.
        self.game_over() #check if game is over.
 
    # This method moves the tiles in the 4x4 square downwards.
    def downswipeS(self, event):
        new_board = self.transpose(self.board) #transpose board...
        new_board = self.reverse(new_board) #and reverse board so it works the same as the left swipe.
        new_board = self.stack(new_board) #compress non zero numbers to the left of the matrix.
        new_board = self.fusion(new_board) # fuse the adjacent numbers.
        new_board = self.stack(new_board) #get rid of all newly created zeros.
        new_board = self.reverse(new_board) #reverse board and...
        new_board = self.transpose(new_board) #transpose board back to original orientation.
        self.add_tile(new_board) #add a new tile.
        self.board = new_board
        self.write_big_board_into_little_boards() #rewrite the board into a complete board and a small board that also can be used.
        self.GUI_update() #update the GUI accordingly.
        self.game_over() #check if game is over.
     
    # This method is used to update the list of 2x2 boards if the 4x4 board is changed. 
    def write_big_board_into_little_boards(self):
        board1 = []
        board2 = []
        board3 = []
        board4 = []
        
        # The big board is transcribed into the four smaller boards according to their indices in these for loops.
        for i in range(2):
            board1.append([])
            for j in range(2):
                board1[i].append(self.board[i][j])
                
        for i in range(2):
            board2.append([])
            for j in range(2,4):
                board2[i].append(self.board[i][j])
        
        for i in range(2,4):
            board3.append([])
            for j in range(2):
                board3[i-2].append(self.board[i][j])
        
        for i in range(2,4):
            board4.append([])
            for j in range(2,4):
                board4[i-2].append(self.board[i][j])
         # The listOfBoards is then updated, so that the 2x2 boards and the 4x4 boards are always synchronised
        self.listOfBoards = [board1, board2, board3, board4]

    # This method updates the highscore if it is lower than the current score
    def update_highscore(self): #definition to save highscore
        if self.score > self.highscore: #if the score is bigger than the high score...
           file = open ('highscore.txt', 'wt') #the high score file is opened and it is saved in it.
           file.write(str(int(self.score)))
           file.close()

    # This method checks whether the game is over and if it is, it initialises the winning or losing process
    def game_over(self):
        # First, we loop through the entire board to check whether 2584 has been achieved, which would mean that the player won.
        for i in range (4):
            for j in range (4):
                if (self.board [i][j]==2584):
                    # If the player wins, the winning sound is played, the highscore is compared to the current score and a new game is started after 10 seconds
                    self.game_message="YOU WIN! :)"
                    self.lbl.configure(text=self.game_message)
                    self.lbl.configure(fg='green')
                    self.play_winning_sound()
                    self.update_highscore()
                    self.after(10000, lambda: self.new_game())
                    return
        
        # If any 0s are still on the board, the game is not over yet, so we can return.
        for i in range (4):
            for j in range (4):
            		if (self.board[i][j]==0):
            			return
        
        # This methods checks whether there are any neighbouring Fibonacci numbers on the board. It uses range(3) so that no exception occurs when adding 1 to the 
        # index of the row or column. If there are neighbouring Fibonacci numbers, the game is not over yet so we can return.
        for i in range (3):
        	for j in range (3):
                    if (self.board[i][j+1] == self.consecutive_num.get(self.board[i][j])) or (self.board[i][j] == self.consecutive_num.get(self.board[i][j+1])):
                       	 return
                    elif self.board[i+1][j] == self.consecutive_num.get(self.board[i][j]) or self.board[i][j] == self.consecutive_num.get(self.board[i+1][j]):
                        return
                    elif self.board[i][j] == 1 and self.board[i][j] == self.board[i][j+1]:
                        return
                    elif self.board[i][j] == 1 and self.board[i+1][j] == self.board[i][j]:
                        return
	    
        # Since we used range(3) earlier, two more for loops are required to check the last row and the last column for neighbouring consecutive Fibonacci numbers.
        for j in range(3):
            if self.board[3][j+1] == self.consecutive_num.get(self.board[3][j]):
                return
            elif self.board[3][j] == self.consecutive_num.get(self.board[3][j+1]):
                return
            elif self.board[3][j] == 1 and self.board[3][j+1] == self.board[3][j]:
                return

        for i in range(3):
            if self.board[i+1][3] == self.consecutive_num.get(self.board[i][3]):
   	        	return
            elif self.board[i][3] == self.consecutive_num.get(self.board[i+1][3]):
                return
            elif self.board[i][3] == 1 and self.board[i+1][3] == self.board[i][3]:
                return
           
        # If none of the previous loops returned from the method, there are no more potential moves and the game is over.
        # This means that the losing sound is played, the highscore is compared to the current score and after 10 seconds a new game is started.
        self.game_message="YOU LOSE!"
        self.lbl.configure(text=self.game_message)
        self.lbl.configure(fg='red')
        self.play_losing_sound()
        self.update_highscore()
        self.after(10000, lambda: self.new_game())
      
    # This method fuses corresponding tiles. 
    def fusion(self, board):
        a = 2
        if len(board) == 4:
            a = 4
        # This method is always called after transposing/reversing/stacking the board into the 'left' form, so we only need to check neighbouring columns and not rows.
        for i in range (a):
            for j in range (a-1):
                if board[i][j]<board[i][j+1]:
                    if board[i][j+1] == self.consecutive_num.get(board[i][j]): # For any cell, we compare if the cell in the neighbouring column is the next Fibonacci number.
                        board[i][j]= self.consecutive_num.get(board[i][j+1]) # If it is, we set the value of the leftmost cell to the value of the dictionary key-value pair of the higher number.
                        board[i][j+1]=0 # The other cell is set to 0
                        self.score += self.board[i][j] #add the amount of the fusion to the score.
                        self.play_fusion_sound() # The fusion sound is played.
                elif board[i][j]>board[i][j+1]: # Since the cells should fuse no matter whether the left or right value is bigger, we also need to do the same checks the other way around.
                    if board[i][j] == self.consecutive_num.get(board[i][j+1]):
                        board[i][j]= self.consecutive_num.get(board[i][j])
                        board[i][j+1]=0
                        self.score += self.board[i][j] #add the amount of the fusion to the score.
                        self.play_fusion_sound()
                elif board[i][j] == 1 and board[i][j] == board[i][j+1]:
                    # The 1 fuses with both 1s and 2s, but there cannot be two 1s as keys in the dictionary. Therefore, we do an individual check to see whether there are neighbouring 1s.
                    board[i][j]= 2
                    board[i][j+1]=0
                    self.score += self.board[i][j] #add the amount of the fusion to the score.
                    self.play_fusion_sound()
        
        return board

       
Game() #The game class is called to start the game



