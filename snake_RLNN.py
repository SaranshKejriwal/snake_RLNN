
from snakeGame import snakeGame
from modelContainer import modelContainer
from trainingDataContainer import trainingDataContainer

#It is important to note that the neural network will not choose between Up, Down, Left, Right (because predicting Down while the snake is moving up does nothing). 
#It will only predict left/right/stay, RELATIVE to the current direction of movement, thereby always returning an actionable prediction

#neural network model is only instantiated here and passed to the snakeGame object. 
#All subsequent training and test occur between the model and the game objects

dataContainer = trainingDataContainer()

#a neural network cannot be trained on one example at a time. We need to prepare thousands of example that should be sent to the training loop AT ONCE, for the network to converge at all.
model = modelContainer(dataContainer, 5000)#gather 5000 examples of what the deterministic algorithm would've done

#model.startGatheringData()

#once the model has found enough examples
#model.trainModel(10000) #modelContainer already has an instance of trainingDataContainer, so no arguments are required. We're training the model over 10000 iterations

print ('Model Training Complete.')

model.testModel()



