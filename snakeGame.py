# importing libraries
import pygame
import time
import random
import snakeMaths
import decisionRewardCalculator

from snake import snakeObject
from food import food

#this class instantiates a snake object and a food object, and constructs the game around them.
class snakeGame:

    #Game setup

    # Window size
    window_x = 100 #this is the size of the window in pixels, NOT the number of cells
    window_y = 100

    #Game colors
    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)

    snake = snakeObject(window_x, window_y)

    #speed of the snake movement - faster for training
    gameSpeed = 200

    food = food(xLimit = window_x, yLimit = window_y)

    #current score is a running placeholder for the score while the game is ongoing. Resets to 0 
    currentScore = 0

    #max Score will hold the max score achieved across all games 
    maxScore = 0

    trainingGameCounter = 1 #this will track how many games were played during training

    def __init__(self):
        self.game_window = pygame.display.set_mode((self.window_x, self.window_y))
        return


    def startGame(self, model, isTraining, dataGatheringIterationLimit):
        #isTraining will dictate whether to pass the decision params to training or not.


        # Initialising pygame
        pygame.init()

        # Re-init the game window
        self.game_window = pygame.display.set_mode((self.window_x, self.window_y))

        # Initialise game window
        pygame.display.set_caption('Reinf Learning attempt - Snake')
        #game_window = pygame.display.set_mode((self.window_x, self.window_y))

        #control FPS for visibility
        fps = pygame.time.Clock()

        if isTraining:
            self.gameSpeed = 30000 #faster runthrough for data gathering
        else:
            self.gameSpeed = 100

        print("starting new game...")

        iterationCounter = 0

        #infinite loop that continues until the game is over
        while True:

            iterationCounter += 1
            #if the modelContainer is in Data Gathering mode, we need to stop the game after a certain set of iterations
            if isTraining and iterationCounter > dataGatheringIterationLimit:
                print('Data Gathering complete. Ending the training game...')
                break #break the infinite loop after the set number of iterations is reached

            #reset the isGrowing variable, which will only be set to true when the snake reaches the food
            isGrowing = False

            #.all() is used for numpy to correctly compare the 2 2D arrays
            if((self.snake.getSnakeHead() == self.food.getFoodLocation()).all()):
                #snake has reached the food
                isGrowing = True #while moving, the snake body queue will not pop
                self.food.respawn(self.snake.getSnakeBody()) #move the food immediately to another random location
                self.currentScore += 1 #increment score by 1 

                if(self.currentScore > self.maxScore):
                    self.maxScore = self.currentScore
                    print('New Score Record:', self.maxScore)
                    print('Training games played:', self.trainingGameCounter)

            else:
                isGrowing = False

            #move the snake to the next position with/without growing it
            self.snake.move(isGrowing)

            #if(self.isGameOver()):
            if(snakeMaths.isGameOver(self.snake.getSnakeHead(),self.snake.getSnakeBody(),self.window_x,self.window_y)):


                if not(isTraining):
                    #break the infinite loop and end the game while running the trained model

                    self.endGame()
                else:
                    #continue to train the model if the snake is training, after resetting the snake.
                    self.resetGame()
        

            #setup the background
            self.game_window.fill(self.black) #recolor the cells that were covered by the snake earlier.

            #print(len(self.snake.getSnakeBody()))
            #draw the current snake and food position, if the snake is NOT training
            for pos in self.snake.getSnakeBody():
                pygame.draw.rect(self.game_window, self.green,pygame.Rect(pos[0], pos[1], 10, 10))
             
            pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.food.getFoodLocation()[0], self.food.getFoodLocation()[1], 10, 10))
            #show score on screen
            self.displayCurrentScore(self.white, 'times new roman', 20)
            # Refresh game screen
            pygame.display.update()
            # Frame Per Second /Refresh Rate as per required game speed
            fps.tick(self.gameSpeed)

            rewardOfDecision = decisionRewardCalculator.getRewardFromDirection(self.window_x,self.window_y, self.snake.getSnakeBody(), self.food.getFoodLocation(), self.snake.getSnakeHead())

            #get the next decision from the model
            modelDecision = model.predictNextStep(self.window_x,self.window_y,self.snake.getSnakeHead(), self.snake.getSnakeBody(), self.snake.getSnakeDirection(), self.food.getFoodLocation(), rewardOfDecision, isTraining)
            '''
                The model would need the following inputs during training and test
            '''

            self.snake.changeDirection(modelDecision)

        self.quitGame()
        #return the final score once the While loop breaks and the game is over.
        return self.currentScore #unreachable code


    #this is an internal function that will close the pygame instance once the snake loses
    def endGame(self):

        #end the game
        pygame.quit()
        print("Game Over. Final Score is ", self.currentScore)
        return self.currentScore

    #this will be called when we want to stop gathering data. A new game will be started to test the trained model.
    def quitGame(self):
        pygame.quit()

    #while in training mode, reset the game upon snake death, whilst the training continues...
    def resetGame(self):
        #print("Training...: Final score is ", self.currentScore)

        self.trainingGameCounter += 1

        self.snake.resetSnake(self.window_x,self.window_y)
        self.currentScore = 0

    # displaying Score function
    def displayCurrentScore(self, color, font, size):
  
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)
    
        # create the display surface object 
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.currentScore), True, color)

        #games_surface = score_font.render('Games : ' + str(self.trainingGameCounter), True, color)
        #games_surface = score_font.render('Length : ' + str(len(self.snake.getSnakeBody())), True, color)

        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        #games_rect = games_surface.get_rect()
    
        # displaying text
        self.game_window.blit(score_surface, score_rect)
        #self.game_window.blit(games_surface, games_rect)

    def getWindowX(self):
        return self.window_x

    def getWindowY(self):
        return self.window_y
