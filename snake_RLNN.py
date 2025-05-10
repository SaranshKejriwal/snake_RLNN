
from snakeGame import snakeGame
from modelContainer import modelContainer


#It is important to note that the neural network will not choose between Up, Down, Left, Right (because predicting Down while the snake is moving up does nothing). 
#It will only predict left/right/stay, RELATIVE to the current direction of movement, thereby always returning an actionable prediction

#neural network model is only instantiated here and passed to the snakeGame object. 
#All subsequent training and test occur between the model and the game objects

model = modelContainer()

model.trainModel()

model.testModel()

#model.startGame()


