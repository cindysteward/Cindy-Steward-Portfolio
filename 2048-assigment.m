%This is our game 2048!  By Cindy Steward, Anastazia Vojvodic and Bionda
%Blakaj. We all cooperated to think of how we would do this, start the
%coding and the general game code. Bionda worked mostly on the starting
%parts, with the high score, putting together the general code and the
%order of the functions used. Also on implementing the music. Anastazia
%worked on the starting board and the number generator and random position
%generator. Cindy worked mostly on the plotting, the swiping control and
%the continuation of the game. Of course, this separation is a bit weirdly
%worded, as we all worked on the code together and parts of the code could
%not do without each other. Anastazia and Cindy together made the flow
%chart, and Bionda made the power point presentation implementing/including
%the flow chart. At the end we all perfected our comments in the code
%together.

%Just to be sure, so other code or variables won't interfere.
clear;
clc;


%When the game has been played before, it will show a highscore. This is
%the score that has been tracked when the game was played before. An if-loop
%is used. If there is no high score yet, it will be set to 0. After the
%game is played, that score will replace the 0. %Exist shows whether a variable
%exists. "==2", as exist returns a 2 if a file with this name exists. A file
%is used to save the score. When saved as a variable, it makes the loading of
%the score, here, at the start, more difficult.

if exist('high_score2048.mat', 'file') == 2  
   load('high_score2048.mat');                
else                                      
    high_score = 0;                               
end                                                           
                                                                    
%This plays the sound for the game. It will terminate once the code is
%terminated. We used calm Zelda music, so it does not distract from the
%game and because it is ICONIC.
ZeldaMusic = 'C:\Users\wwwci\Downloads\CSMFinalAssignmentZelda.mp3';
[y, Fs]= audioread(ZeldaMusic);
music = audioplayer(y, Fs);
play(music) 

%This is the size of the board, or the gamefield. The blocks will appear on this board.
%If 2048 occurs on the board, display once that user won, then go on. All
%the variables are used in the functions at the end of the document.
N = 4; %Size of playfield. We use 4, just like the example given.                        
act = 1; %We use act as an indicator for when the game is 'active'.
wins = 0; %The game is not finished, the player has not won yet. When there is a 2048, it will show a win message (later in code). This variable is solely for that.

[GameField, score] = StartingBoard(N); %Here we use the function StartingBoard for the start of the game.
%The output of this function wll define the Gamefield and score in this
%case.

disp(['Previous High Score: ',num2str(high_score)])

%This plots the board, so we get a nice GUI. This function is defined at the end of the file.
plotGameField(GameField)

while act == 1 %When the game is going on!
    direction = ContinueBoard(GameField,score,high_score,wins); %The function ContinueBoard, which brings us the direction of the swipe.
    
    if direction == 'end game'
       stop(music) %so the music quits even when the game is not aborted by finishing the game.
       error(['The game was ended. The score is not saved.'])
    end 
     
    [GameField, score] = SwipingControl(N,GameField,score,direction);
    %The new game field is drawn.
    close;
    plotGameField(GameField)    
    
    if direction ~= 'a'|'s'|'w'|'d'|'end game' %If the swipe happens in a direction not allowed, it will go back to the start of the code, until valid code was added.
        continue %Back to the start.
    end

    %Below checks whether the field is full. The game does not immediately
    %end if so, as a swipe can create an empty space. That is why swipes in
    %every direction are done, so the program knows whether a swipe can
    %still be done. The game will end if no swipe will create an empty
    %"block".
    notzero = find(GameField==0, 1);
    if isempty(notzero) == 1
        direction = 1:4; %There are 4 directions.
        for ii = direction
            GameField2 = GameField;
            [GameField, score] = SwipingControl(N, GameField,score,direction); %Uses the function SwipingControl.
            %It generates the new game field.
            if GameField ~= GameField2
                break %Break, because no swipe emptied a space.
            end
        end
        act = 0; %The game is not active anymore.
    end
    
end
   
    
%Below is the code for saving the newly aqcuired score and setting it as
%the high score. This is closely related with code earlier that loades the
%high_score, to show it on the field.
disp(GameField)
disp(' ')
disp(['Score: ',num2str(score)])
new_score = score; %Using new_score to compare with high_score. In case the program would confuse score.

if new_score > high_score 
    high_score = new_score; %Update high_score.                     
    save('high_score2048.mat','high_score')
end

stop(music) %Otherwise the music will keep running even after the game is over. Very annoying.

disp(' ')
error('Game Over!') %So when the game breaks, it shows "Game Over!".


%%%%%%%%Functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [row,column,value] = CreateRandomNumber(N)
%This function creates the random number that is added to the matrix. Its
%input is N, which is the size (rows and columns) of the GameField. Value,
%as output, describes the random number that is added to the matrix. This is either a 2 or 4.
%This is used in the function RandomPosition. row and column
%as outputs, are randomly generated. These are used in StartingBoard to generate the board.
        row = randi(N); %Random numbers are generated.
        column = randi(N);

        value = randi(10); %It is assigned a random number.
        if value < 9 %When this random number is smaller than 9, the random value added to the matrix is 2.
%We do it like this so more 2's are added than 4.
            value = 2;
        else
            value = 4;
        end
end

function [GameField, score] = StartingBoard(N)
%This function creates a matrix(NxN), using the input N, which is
%described in the code. It then creates two numbers in this matrix at
%random positions. One number is a 2, the other is a 4. It sets the initial
%score at 0, so it can be tracked from here on out. The output is GameField (the board with the
%numbers that are put initially in the matrix) and score (the score being tracked).

GameField = zeros(N); %Creates the matrix, full of zeros.
score = 0; %Starting score in the game is 0.

%This creates te number that will be added to the GameField first. This is then situated in the GameField.
[row1,column1,value1] = CreateRandomNumber(N); %The function generates the values for row1,
%column1 and value1. These are the same as the outputs of the function. 
GameField(row1,column1) = value1; %Position on the GameField becomes the randomly generates number.

%The position of a number must differ from the other generated value. It cannot be the same as the other number. It
%thus generates new positions or the second generated number until it also has its own positions.  
differentposition = 0;
while differentposition~=1
    [row2,column2,value2] = CreateRandomNumber(N); %Use function again.
    if row1 == row2 && column1 == column2
        differentposition = 0;
    else
        differentposition = 1;
        GameField(row2,column2) = value2;
    end
end

%The code below displays the game and the score (which is updates
%continiously).
clc
disp(GameField)
disp(' ')
disp(['Score: ',num2str(score)]) 

end

function [] = plotGameField(GameField)
%This function plots the gamefield. As input is the GameField, defined in
%earlier functions. This is the matrix with the gamefield and the numbers
%on it. The output is the plotted gamefield (as an image), with colours we have chosen.

GameField_Expon = log(GameField)/log(2); %We write the Game field like this (as a graph) so it can be plotted. We learnt how to do this by using external sources, as this part was really confusing.
GameField_Expon(GameField_Expon==-Inf)=0;

%These are the colours in the game that we have chosen. We used the RGB
%from some colours we thought were nice from Google. Each value has a
%different colour. When playing, it is easy to differentiate between each
%value.
Colormap = [255, 255, 255 %0 is white, so it is clear.
    227, 200, 0 %For value 2.
    240, 163, 10 %For value 4.
    250, 104, 0 %For value 8.
    229, 20, 0 %For value 16.
    162, 0, 37 %For value 32.
    216, 0, 115 %For value 64.
    244, 114, 208 %For value 128.
    170, 0, 255 %For value 256.
    106, 0, 255 %For value 512.
    0, 80, 239 %For value 1024
    27, 161, 226 %For value 2048.
    0, 171, 169]; % In case... for values bigger than 2048. (Although we barely get to 512 (':).
Colormap = Colormap./255; %Because of the way a colormap must be presented!!

hFig = figure; %Creating a figure window.
set(hFig, 'Position', [250 200 600 600])
imagesc(GameField_Expon) %Displays values or data from an array as an image with colors (using the colormap).
colormap(Colormap) %Colormap used, as we defined before.
caxis([0 log(4096)/log(2)]) %Scales the axis (when the axis has a colour).

set(gca, 'XTick', []); %Set the tick values (next to the images or graph) to empty, so they don't show.
xlabel([]); %We set the labels and axis as empty so they are not shown.

set(gca, 'YTick', [])
ylabel([]);

linkdata on %The figure is linked to the variables in the workspace, so it will automatically update.

for ii = 1:size(GameField,1) %This draws the whole gamefield essentially. For each
%place in the Gamefield, we place a text label (or number) with its information. What the value is supposed to say.
    for j = 1:size(GameField,2)
        textlabel = sprintf('%i', GameField(ii,j));
        text(j,ii,textlabel,'HorizontalAlignment','center','fontsize',30); %Making it clearly, also centering it in the "boxes".
    end
end

end


%The functions below describe  the controls and continuation of the game.

function [direction] = ContinueBoard(GameField,score,high_score,wins)
%This function keeps the game running. It defines direction, which is the direction
%of the swipe that will be done. The swiping function is defined below.
%This function allows the input of the direction in which is being swiped.
%As input, it firstly the GameField. The GameField is the matrix on
%which the game takes place. The score is that is being updated continuously.
%The GameField and score are defined in the function StartingBoard and continuously
%updated. Wins is the amount of wins that the user has gotten. This way, the
%wins are saved and tracked.

clc %So the previous gamefield is deleted and the new one is shown. This is
%why we also do not have an error message. Nothing will happen until a correct
%direction key is entered.(Scroll down for direction input!)
disp(GameField) %Display the gaming field, with the score and high score that are tracked.
disp(' ')
disp(['Score: ',num2str(score)])
disp(['High score: ',num2str(high_score)])
disp(' ')
winvariable = find(GameField==2048, 1); %Searches whether there is a 2048 in the game.
%(Find returns where this variable is, but we don't need this to show so we used the
%semicolon to suppress this.) If so, the player won and the next if loop saves that win.

if isempty(winvariable) ~= 1 && wins == 0 %If x is not empty... and wins is 0 like described earlier in code...
    disp('WINNER WINNER CHICKEN DINNER! You can continue if you want(:') %This message is shown when the winner won.
    wins = 1;
end

%The input of the reaction is asked.
direction = input('Type "end game" to quit the game. Otherwise, enter the direction of the next swipe!\n a: LEFT, w: UP, d: RIGHT, s: DOWN\n\n','s');
%ASWD so 's', because input is a string.

%We chose to interpet the assignment saying "you should use keyboard keys
%for input" as using w/a/s/d. It was not specified to use arrows. Besides, using
%the press of arrow keys was difficult to figure out. We tried! These directions
%are used in the SwipingControl and later when swipes are checked whether possible..

end

function [GameField] = RandomPosition(GameField)
%When a swipe is done, a new random number must be added to the matrix or
%GameField. This function does that by using the input GameField
%(the matrix with the numbers/"blocks"). It then adds the value for the new
%position. We could have combined this function with CreateRandomNumber,
%but we chose to keep them apart, as we use values that come from that
%function precisely in the game code.

notzero = find(GameField==0); %This searches for whether there are zeros,
%or empty spaces in the game field/matrix. It then saves it in notzero. A similar variable/definition
%will be used in other functions. We choose to define it again to make sure
%each function uses the correct variables and defintions.

randomposition = randi(length(notzero)); %This variable
%generates a random position where the value is placed. Because it must
%be on a position that is empty, and not on top of a number/"block" that
%already exists.

randomvalue = randi(10); %We use this again to define a random value that will be assigned a random spot on the game field.
        if randomvalue < 9 %When this random number is smaller than 9, the random value added to the matrix is 2.
%We do it like this so more 2's are added than 4 (like explained before)
            randomvalue = 2;
        else
            randomvalue = 4;
        end
        
GameField(notzero(randomposition))= randomvalue; %Random position gets the randomvalue as value.

end

function [GameField, score] = SwipingControl(N,GameField,score,direction)
%This function is the core of the controls. It codes for the swiping and
%merging of blocks. It also updates the score when blocks are added. The
%input N is the size of the gamefield (the rows and columns of the matrix),
%defined in the game code. The GameField is the matrix on which the game takes place.
%The score is that is being updates continuously. The GameField and score are
%defined in the function StartingBoard and continuously updates, hence also
%being the output of this function. The direction is defined in the function
%ContinueBoard, which indicates the direction in which is being swiped.
%It consists of 4 nested if-loops (for each direction) with 3 for-loops, each
%for an important part: swiping, merging and updating score, new swipe.
%Each direction is defined separately, because trying to define them
%together confused the loops together and would not work.
    
    %% For the left-swipe.
%All the swipes work the same way, so we won't post the explanation for each
%of them. Only to indicate the swipe.

if direction == 'a' %Takes the input from the direction function.
    for ii = 1:N
        notzero = find(GameField(ii,:)~=0); %Search for all the elements in the code that are not 0 and save them in the variable notzero.
        if length(notzero) ~= N  %The shift of numbers or "blocks" will only happen if there are zeros.
            GameField(ii,:) = [GameField(ii,notzero), zeros(1,N-length(notzero))]; %All the elements that are not 0 are shifted to the left, the direction given.
        end
    end
    
    for ii = 1:N
        for j = 1:N-1
            if GameField(ii,j) == GameField(ii,j+1) %If two elements that are no zero have the same value, they will be added together.
                GameField(ii,j) = 2*GameField(ii,j); %The adding.
                GameField(ii,j+1) = 0;  %The value that was just added is removed from the original spot, so this spot becomes empty and one "block" with the added numbers remains.
                score = score+GameField(ii,j)/2; %The score that is being tracked is updated.
            end
        end
    end
    
    for ii = 1:N  %If the numbers are added together, zeros can remain between "blocks" in game field. So, there will be another swipe in the same direction. 
        notzero = find(GameField(ii,:)~=0);
        if length(notzero) ~= N
            GameField(ii,:) = [GameField(ii,notzero), zeros(1,N-length(notzero))];
        end
    end
    
    GameField = RandomPosition(GameField); %We use the function RandomPosition to add a new 2 or 4 at a random position. After the swipe is executed, this is done immediately.
end

    %% For the up-swipe.
    
if direction == 'w'
    for ii = 1:N
        notzero = find(GameField(:,ii)~=0);
        if length(notzero) ~= N
            GameField(:,ii) = vertcat(GameField(notzero,ii), zeros(N-length(notzero),1)); %Move elements up.
        end
    end
    
    for ii = 1:N-1
        for j = 1:N
            if GameField(ii,j) == GameField(ii+1,j)
                GameField(ii,j) = 2*GameField(ii,j);
                GameField(ii+1,j) = 0;
                score = score+GameField(ii,j)/2;
            end
        end
    end
 
    for ii = 1:N
        notzero = find(GameField(:,ii)~=0);
        if length(notzero) ~= N
            GameField(:,ii) = vertcat(GameField(notzero,ii), zeros(N-length(notzero),1));
        end
    end
    
    GameField = RandomPosition(GameField);
end

    %% For the right-swipe.
if direction == 'd' %Takes the input from the direction function.
    for ii = 1:N
        notzero = find(GameField(ii,:)~=0);
        if length(notzero) ~= N
            GameField(ii,:) = [zeros(1,N-length(notzero)), GameField(ii,notzero)]; %Move elements right.
        end
    end

    for ii = 1:N
        for j = N:-1:2
            if GameField(ii,j) == GameField(ii,j-1)
                GameField(ii,j-1) = 2*GameField(ii,j);
                GameField(ii,j) = 0;
                score = score+GameField(ii,j-1)/2;
            end
        end
    end

    for ii = 1:N
        notzero = find(GameField(ii,:)~=0);
        if length(notzero) ~= N
            GameField(ii,:) = [zeros(1,N-length(notzero)) GameField(ii,notzero)];
        end
    end
    
    GameField = RandomPosition(GameField);
end

    %% For the down-swipe.
if direction == 's'  %Takes the input from the direction function.

    for ii = 1:N
        notzero = find(GameField(:,ii)~=0);
        if length(notzero) ~= N
            GameField(:,ii) = vertcat(zeros(N-length(notzero),1), GameField(notzero,ii)); %Moves elements down.
        end
    end
    
    for ii = N:-1:2
        for j = 1:N
            if GameField(ii,j) == GameField(ii-1,j)
                GameField(ii-1,j) = 2*GameField(ii,j);
                GameField(ii,j) = 0;
                score = score+GameField(ii-1,j)/2;
            end
        end
    end

    for ii = 1:N
        notzero = find(GameField(:,ii)~=0);
        if length(notzero) ~= N
            GameField(:,ii) = vertcat(zeros(N-length(notzero),1), GameField(notzero,ii));
        end
    end
    
    GameField = RandomPosition(GameField);
end

end
