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
import envMujoco as env
import reward
import state
import policyValueResNet as net

from collections import OrderedDict
from trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle
from augmentData import GenerateSymmetricData, GenerateSymmetricDistribution, GenerateSymmetricState, CalibrateState
from preprocessData import PreProcessTrajectories, ProcessTrajectoryForNN, RemoveNoiseFromState
from preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from dataTools import createSymmetricVector
import trainTools
import random

def dictToFileName(parameters):
    sortedParameters = sorted(parameters.items())
    nameValueStringPairs = [
        parameter[0] + '=' + str(parameter[1])
        for parameter in sortedParameters
    ]
    modelName = '_'.join(nameValueStringPairs).replace(" ", "")
    return modelName


class GenerateTrainedModel:

    def __init__(self, numTrials, getSavePathForModel, generateModel, generateTrain, generateSampleBatch, generateReporter):
        self.numTrials = numTrials
        self.getSavePathForModel = getSavePathForModel
        self.generateTrain = generateTrain
        self.generateModel = generateModel
        self.generateSampleBatch = generateSampleBatch
        self.generateReporter = generateReporter

    def __call__(self, df, trainData, trainingSteps):
        batchSize = df.index.get_level_values('batchSize')[0]
        neuronsPerLayer = df.index.get_level_values('neuronsPerLayer')[0]
        sharedLayers = df.index.get_level_values('sharedLayers')[0]
        actionLayers = df.index.get_level_values('actionLayers')[0]
        valueLayers = df.index.get_level_values('valueLayers')[0]
        augment = df.index.get_level_values('augment')[0]
        initialization = df.index.get_level_values('initialization')[0]
        blockSize = df.index.get_level_values('blockSize')[0]
        indexLevelNames = df.index.names
        parameters = {
            levelName: df.index.get_level_values(levelName)[0]
            for levelName in indexLevelNames
        }
        sample = self.generateSampleBatch(augment)
        for step in trainingSteps:
            parameters.update({'trainingStep': step})
            saveModelDir = self.getSavePathForModel(parameters)
            modelName = dictToFileName(parameters)
            modelPath = os.path.join(saveModelDir, modelName)
            model = self.generateModel([neuronsPerLayer] * sharedLayers,
                                       [neuronsPerLayer] * actionLayers,
                                       [neuronsPerLayer] * valueLayers,
                                       blockSize, initialization)
            if not os.path.exists(saveModelDir):
                reporter = self.generateReporter(step)
                train = self.generateTrain(step, batchSize, sample, reporter)
                trainedModel = train(model, trainData)
                net.saveVariables(trainedModel, modelPath)
        return pd.Series({"train": 'done'})


class SampleDataWithAugmentation:
    def __init__(self, generateSymmetricData,  augmented, trainData):
        self.generateSymmetricData = generateSymmetricData
        self.augmented = augmented
        self.data = list(zip(*trainData))
        self.augmentData = None

    def __call__(self, data, batchSize):
        if self.augmented and self.augmentData is None:
            self.augmentData = [self.generateSymmetricData(data) for data in self.data]
        if self.augmented and self.augmentData is not None:
            normalBatch = [list(varBatch) for varBatch in random.sample(self.augmentData, batchSize)]
            # augmentedBatch = [self.generateSymmetricData(data) for data in normalBatch]
            reducedBatch = [random.sample(batch, 1)[0] for batch in normalBatch]
            finalBatch = [list(batch) for batch in zip(*reducedBatch)]
            return finalBatch
        else:
            return net.sampleData(self.data, batchSize)


def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data', 'evaluateAugmentationWithinMujoco')

    # get train data
    trajectoryDir = os.path.join(dataDir, "trainingTrajectories")
    trajectoryParameter = OrderedDict()
    trajectoryParameter['killzoneRadius'] = 2
    trajectoryParameter['maxRunningSteps'] = 25
    trajectoryParameter['numSimulations'] = 100
    trajectoryParameter['qPosInitNoise'] = 9.7
    trajectoryParameter['numTrajectories'] = 6000
    trajectoryParameter['qVelInitNoise'] = 8
    trajectoryParameter['rolloutHeuristicWeight'] = -0.1
    getTrajectorySavePath = GetSavePath(trajectoryDir, ".pickle", trajectoryParameter)
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectories = loadTrajectories({})
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7),
                   (0, -10), (7, -7)]
    actionToOneHot = lambda action: np.asarray(
        [1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
         for index in range(len(actionSpace))])
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
    playIsTerminal = env.IsTerminal(playKillzoneRadius, getWolfXPos, getSheepXPos)
    playReward = reward.RewardFunctionCompete(playAlivePenalty, playDeathBonus,
                                       playIsTerminal)
    processTrajectoryForNN = ProcessTrajectoryForNN(sheepId, actionToOneHot)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory,
                                                    removeTerminalTupleFromTrajectory,
                                                    processTrajectoryForNN)
    trainData = preProcessTrajectories(trajectories)

    # generate NN model
    numOfStateSpace = 12
    numOfActionSpace = 8
    regularizationFactor = 1e-4
    generateModel = net.GenerateModel(numOfStateSpace, numOfActionSpace, regularizationFactor)

    # generate NN train
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    trainTerminalController = trainTools.TrainTerminalController(
        lossHistorySize, lossChangeThreshold)
    initActionCoefficient = (1, 1)
    initValueCoefficient = (1, 1)
    coefficientController = trainTools.CoefficientCotroller(
        initActionCoefficient, initValueCoefficient)
    reportInterval = 1000
    generateReporter = lambda trainingStep: trainTools.TrainReporter(reportInterval,trainingStep)
    decayRate = 1
    decayStep = 1
    initLearningRate = 1e-4
    learningRateModifier = trainTools.LearningRateModifier(
        initLearningRate, decayRate, decayStep)
    numOfAgent = 2
    stateDim = 6
    symmetries = [
        np.array([1, 1]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([-1, 1])
    ]
    mujocoXBoundary = [-10, 10]
    mujocoYBoundary = [-10, 10]
    round = lambda state: np.round(state, 10)
    calibrateState = CalibrateState(mujocoXBoundary, mujocoYBoundary, round)
    xPosIndex = [0, 1]
    qPosIndex = [2, 3]
    velIndex = [4, 5]
    generateSymmetricState = GenerateSymmetricState(numOfAgent,stateDim,qPosIndex,xPosIndex,velIndex,createSymmetricVector,calibrateState)
    generateSymmetricDistribution = GenerateSymmetricDistribution(actionSpace,createSymmetricVector)
    generateSymmetricData = GenerateSymmetricData(symmetries,generateSymmetricState,generateSymmetricDistribution)
    generateSampleData = lambda augment: SampleDataWithAugmentation(generateSymmetricData, augment, trainData)
    generateTrain = lambda trainingStep, batchSize, sampleData, reporter: net.Train(
        trainingStep, batchSize, sampleData, learningRateModifier,
        trainTerminalController, coefficientController, reporter)

    # split
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionLabel']
    independentVariables['batchSize'] = [64]
    independentVariables['augment'] = [True, False]
    independentVariables['neuronsPerLayer'] = [256]
    independentVariables['sharedLayers'] = [17]
    independentVariables['actionLayers'] = [1]
    independentVariables['valueLayers'] = [1]
    independentVariables['initialization'] = ['glorot', 'uniform']
    independentVariables['blockSize'] = [2]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    trainedModelDir = os.path.join(dataDir, "trainedModel")
    if not os.path.exists(trainedModelDir):
        os.mkdir(trainedModelDir)
    getModelSavePath= GetSavePath(trainedModelDir, "")

    numTrials = 100
    generateTrainingOutput = GenerateTrainedModel(numTrials,
                                                  getModelSavePath,
                                                  generateModel, generateTrain,
                                                  generateSampleData,
                                                  generateReporter)
    maxTrainingStep = 200001
    saveInterval = 20000
    trainingSteps = [step for step in range(0, maxTrainingStep, saveInterval)]
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput,
                                                      trainData, trainingSteps)


if __name__ == "__main__":
    main()
