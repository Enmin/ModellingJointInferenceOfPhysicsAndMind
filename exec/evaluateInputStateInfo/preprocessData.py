import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import numpy as np
import pandas as pd
import state
import envMujoco as env
import reward
from trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle
from preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from collections import OrderedDict
import pickle


class RemoveNoiseFromState:

    def __init__(self, noiseIndex):
        self.noiseIndex = noiseIndex

    def __call__(self, trajectoryState):
        state = [np.delete(state, self.noiseIndex) for state in trajectoryState]
        return np.asarray(state).flatten()


class ProcessTrajectoryForNN:

    def __init__(self, agentId, actionToOneHot, removeNoiseFromState):
        self.agentId = agentId
        self.actionToOneHot = actionToOneHot
        self.removeNoiseFromState = removeNoiseFromState

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (self.removeNoiseFromState(state), self.actionToOneHot(actions[self.agentId]), value)
        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        return processedTrajectory


class AddToBoundDistanceForTrajectory:

    def __init__(self, stateIndex, statePosIndex, stateDim, numAgent, xBoundary, yBoundary):
        self.stateIndex = stateIndex
        self.statePosIndex = statePosIndex
        self.numAgnet = numAgent
        self.stateDim = stateDim
        self.xBound = xBoundary
        self.yBound = yBoundary

    def __call__(self, trajectory):
        newTrajectory = []
        for step in trajectory:
            newStep = list(step)
            state = newStep[self.stateIndex]
            agentState = [state[index:index+self.stateDim] for index in range(self.numAgnet)]
            distance = [[abs(state[0]-self.xBound[0]), abs(state[0]-self.xBound[1]), abs(state[1]-self.yBound[0]), abs(state[1]-self.yBound[1])] for state in agentState]
            newAgentState = [np.concatenate([agentState[index], distance[index]]) for index in range(self.numAgnet)]
            newState = np.concatenate(newAgentState)
            newStep[self.stateIndex] = newState
            newTrajectory.append(newStep)
        return newTrajectory


class PreProcessTrajectories:

    def __init__(self, addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                 processTrajectoryForNN):
        self.addValuesToTrajectory = addValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.processTrajectoryForNN = processTrajectoryForNN

    def __call__(self, trajectories, addInfoForTrajectory):
        trajectoriesWithValues = [
            self.addValuesToTrajectory(trajectory)
            for trajectory in trajectories
        ]
        filteredTrajectories = [
            self.removeTerminalTupleFromTrajectory(trajectory)
            for trajectory in trajectoriesWithValues
        ]
        processedTrajectories = [
            self.processTrajectoryForNN(trajectory)
            for trajectory in filteredTrajectories
        ]
        augmentedTrajectories = [
            addInfoForTrajectory(trajectory)
            for trajectory in processedTrajectories
        ]
        allDataPoints = [
            dataPoint for trajectory in augmentedTrajectories
            for dataPoint in trajectory
        ]
        trainData = [list(varBatch) for varBatch in zip(*allDataPoints)]

        return trainData


class GenerateTrainingData:

    def __init__(self, getSavePathForData, preProcessTrajectories):
        self.getSavePathForData = getSavePathForData
        self.preProcessTrajectories = preProcessTrajectories

    def __call__(self, df, trajectories):
        trainingDataType = df.index.get_level_values('trainingDataType')[0]
        numOfStateSpace = df.index.get_level_values('numOfStateSpace')[0]
        extraInfo = df.index.get_level_values('extraInfo')[0]
        extraStateNum, addInfoFunctionName, addInfoFunction = extraInfo
        trainingData = self.preProcessTrajectories(trajectories, addInfoFunction)
        parameters = {
            "numOfStateSpace": numOfStateSpace,
            "extraInfo": (extraStateNum, addInfoFunctionName),
            "trainingDataType": trainingDataType
        }
        dataSavePath = self.getSavePathForData(parameters)
        with open(dataSavePath, 'wb') as f:
            pickle.dump(trainingData, f)
        return pd.Series({"dataSet": 'done'})


def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data',
                           'evaluateInputStateInfo')
    trajectoryDir = os.path.join(dataDir, "trainingTrajectories")
    trajectoryParameter = OrderedDict()
    trajectoryParameter['killzoneRadius'] = 2
    trajectoryParameter['maxRunningSteps'] = 25
    trajectoryParameter['numSimulations'] = 100
    trajectoryParameter['numTrajectories'] = 6000
    trajectoryParameter['qPosInitNoise'] = 9.7
    trajectoryParameter['qVelInitNoise'] = 8
    trajectoryParameter['rolloutHeuristicWeight'] = -0.1
    getTrajectorySavePath = GetSavePath(trajectoryDir, ".pickle",
                                        trajectoryParameter)
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectories = loadTrajectories({})
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7),
                   (0, -10), (7, -7)]
    actionToOneHot = lambda action: np.asarray([
        1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
        for index in range(len(actionSpace))
    ])
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][
        actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(
        getTerminalActionFromTrajectory)
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = state.GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = state.GetAgentPosFromState(wolfId, xPosIndex)
    playAlivePenalty = 0.05
    playDeathBonus = -1
    playKillzoneRadius = 2
    playIsTerminal = env.IsTerminal(playKillzoneRadius, getWolfXPos,
                                    getSheepXPos)
    playReward = reward.RewardFunctionCompete(playAlivePenalty, playDeathBonus,
                                              playIsTerminal)
    qPosIndex = [0, 1]
    removeNoiseFromState = RemoveNoiseFromState(qPosIndex)
    processTrajectoryForNN = ProcessTrajectoryForNN(sheepId, actionToOneHot,
                                                    removeNoiseFromState)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)
    preProcessTrajectories = PreProcessTrajectories(
        addValuesToTrajectory, removeTerminalTupleFromTrajectory,
        processTrajectoryForNN)

    trainingDataDir = os.path.join(dataDir, "trainingData")
    if not os.path.exists(trainingDataDir):
        os.mkdir(trainingDataDir)
    getSavePathForData = GetSavePath(trainingDataDir, '.pickle')

    # extra info funtion
    stateIndex = 0
    statePosIndex = [0, 1]
    stateDim = 4
    numAgent = 2
    xBound = [-10, 10]
    yBound = [-10, 10]
    addToBoundDistanceForTrajectory = AddToBoundDistanceForTrajectory(
        stateIndex, statePosIndex, stateDim, numAgent, xBound, yBound)

    # split & apply
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionLabel']
    independentVariables['extraInfo'] = [(8,'toBoundDistance',addToBoundDistanceForTrajectory)]
    independentVariables['numOfStateSpace'] = [8]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    generateTrainingData = GenerateTrainingData(getSavePathForData,
                                                preProcessTrajectories)
    toSplitFrame.groupby(levelNames).apply(generateTrainingData, trajectories)


if __name__ == '__main__':
    main()
