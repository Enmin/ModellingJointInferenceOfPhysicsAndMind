import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))

import numpy as np
import itertools
from constrainedChasingEscapingEnv.envMujoco import IsTerminal
from constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle
from neuralNetwork.policyValueNet import TrainWithCustomizedFetches, sampleData
from neuralNetwork.trainTools import PrepareFetches, SavingTrainReporter, LearningRateModifier
import neuralNetwork.visualizeNN as visualizeNN
from constrainedChasingEscapingEnv.state import GetAgentPosFromState
from preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory


class ProcessTrajectoryForNN:
    def __init__(self, actionToOneHot, agentId):
        self.actionToOneHot = actionToOneHot
        self.agentId = agentId

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        return processedTrajectory


class PreProcessTrajectories:
    def __init__(self, addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN):
        self.addValuesToTrajectory = addValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.processTrajectoryForNN = processTrajectoryForNN

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        processedTrajectories = [self.processTrajectoryForNN(trajectory) for trajectory in filteredTrajectories]
        allDataPoints = [dataPoint for trajectory in processedTrajectories for dataPoint in trajectory]
        trainData = [list(varBatch) for varBatch in zip(*allDataPoints)]
        return trainData


def main(initializationPair, nnStructure):
    # trajectories
    dataDir = os.path.join(os.pardir, os.pardir, 'data', 'compareNetInitialization')
    trajectorySaveDirectory = os.path.join(dataDir, 'trainingTrajectories')

    numTrajectoriesInDataSet = 6000
    numSimulations = 100
    killzoneRadius = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    rolloutHeuristicWeight = -0.1
    maxRunningSteps = 25
    trajectorySaveParameters = {'numTrajectories': numTrajectoriesInDataSet, 'numSimulations': numSimulations,
                                'killzoneRadius': killzoneRadius, 'qPosInitNoise': qPosInitNoise,
                                'qVelInitNoise': qVelInitNoise, 'rolloutHeuristicWeight': rolloutHeuristicWeight,
                                'maxRunningSteps': maxRunningSteps}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)

    # pre-process
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAlivePenalty = 0.05
    playDeathBonus = -1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getWolfXPos, getSheepXPos)
    playReward = RewardFunctionCompete(playAlivePenalty, playDeathBonus, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
                                                for index in range(len(actionSpace))])
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForNN(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                    processTrajectoryForNN)

    # NN
    initializationName, GenerateModel = initializationPair
    numStateSpace = 12
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    sharedWidths, actionLayerWidths, valueLayerWidths = nnStructure
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    model = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # train
    learningRate = 1e-4
    learningRateDecayRate = 1
    learningRateDecayInterval = 5000
    lossCoefs = (1, 1)
    miniBatchSize = 64
    numTrainingTrajectories = 6000
    numTrainIterations = 500000
    trainCheckpointInterval = 2000
    NNFixedParameters = {'structure': nnStructure, 'lossCoefs': lossCoefs,
                         'learningRate': learningRate, 'miniBatch': miniBatchSize,
                         'numTrajectories': numTrainingTrajectories, 'initialization': initializationName}
    NNModelSaveDirectory = os.path.join(dataDir, 'trainedNNs')
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    varsInReport = ['weightGradient', 'biasGradient']
    sectionNameToDepth = {"shared": len(sharedWidths), "action": len(actionLayerWidths) + 1}
    layerIndices = visualizeNN.indexLayers(sectionNameToDepth)
    allKeys = model.graph.get_all_collection_keys()
    findKey = visualizeNN.FindKey(allKeys)
    prepareFetches = PrepareFetches(varsInReport, layerIndices, findKey)
    trainReporter = SavingTrainReporter(trainCheckpointInterval, getModelSavePath)

    terminalController = lambda evalDict, stepNum: False
    coefficientController = lambda evalDict: lossCoefs
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecayRate, learningRateDecayInterval)
    train = TrainWithCustomizedFetches(numTrainIterations, miniBatchSize, sampleData, learningRateModifier,
                                       terminalController, coefficientController, trainReporter, prepareFetches)

    # save untrained model
    modelSavePath = getModelSavePath({'trainSteps': 0})
    saver = model.graph.get_collection_ref('saver')[0]
    saver.save(model, modelSavePath)
    print(f'Model saved in {modelSavePath}')

    allTrajectories = loadTrajectories({})
    trajectories = allTrajectories[0:numTrainingTrajectories]
    trainData = preProcessTrajectories(trajectories)
    print(f'Trajectories# = {numTrainingTrajectories}, dataPoints# = {len(trainData[0])}')
    train(model, trainData)


if __name__ == '__main__':
    from GenerateModel import UniformInitializationModel, NomralInitializationModel, GlorotNormalInitializationModel, HeNormalInitializationModel
    initializationPairs = [#('uniform', UniformInitializationModel),
            #('normal', NomralInitializationModel),
            ('glorot', GlorotNormalInitializationModel),
            ('he',HeNormalInitializationModel)]
    nnStrucutre = [#((128,)*8, (128,), (128,)),
                   ((128,)*16, (128,), (128,))]
    for pair, structure in itertools.product(initializationPairs, nnStrucutre):
        main(pair, structure)