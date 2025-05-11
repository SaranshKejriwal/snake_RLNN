

from sre_parse import State


class trainingDataContainer:

    #initialize lists to hold all the 
    stateData = []
    decisionData = []

    def __init__(self):
        pass


    def addDatapoint(self, stateVector, idealDecision):
        self.stateData.append(stateVector)
        self.decisionData.append(idealDecision)

    def getStateVectorsData(self):
        return self.stateData

    def getDecisionsData(self):
        return self.decisionData