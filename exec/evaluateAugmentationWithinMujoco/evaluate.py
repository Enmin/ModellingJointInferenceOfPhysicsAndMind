import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))

from episode import SampleTrajectory, chooseGreedyAction
from constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniform, TransitionFunction
from evaluationFunctions import GenerateInitQPosUniform
from pylab import plt
from trajectoriesSaveLoad import GetSavePath
from collections import OrderedDict
from reward import RewardFunctionCompete
from preProcessing import AccumulateRewards
import policyValueResNet as net
import mujoco_py as mujoco
import state
import numpy as np
import pandas as pd
import pickle

def dictToFileName(parameters):
    sortedParameters = sorted(parameters.items())
    nameValueStringPairs = [
        parameter[0] + '=' + str(parameter[1])
        for parameter in sortedParameters
    ]
    modelName = '_'.join(nameValueStringPairs).replace(" ", "")
    return modelName

class ModifyEscaperInputState:
    def __init__(self, index):
        self.removeIndex = index

    def __call__(self, worldState):
        state = [np.delete(state, self.removeIndex) for state in worldState]
        return np.asarray(state).flatten()

class PreparePolicy:
    def __init__(self, modifyEscaperInputState):
        self.modifyEscaperInputState = modifyEscaperInputState

    def __call__(self, chaserPolicy, escaperPolicy):
        policy = lambda state: [escaperPolicy(np.asarray(state).flatten()), chaserPolicy(state)]
        return policy

class EvaluateEscaperPerformance:
    def __init__(self, chaserPolicy, allSampleTrajectory, measure, generateEscaperModel, generateEscaperPolicy, preparePolicy, getModelSavePath):
        self.chaserPolicy = chaserPolicy
        self.allSampleTrajectory = allSampleTrajectory
        self.generateEscaperModel = generateEscaperModel
        self.generateEscaperPolicy = generateEscaperPolicy
        self.preparePolicy = preparePolicy
        self.getModelSavePath = getModelSavePath
        self.measure = measure

    def __call__(self, df):
        neuronsPerLayer = df.index.get_level_values('neuronsPerLayer')[0]
        sharedLayers = df.index.get_level_values('sharedLayers')[0]
        actionLayers = df.index.get_level_values('actionLayers')[0]
        valueLayers = df.index.get_level_values('valueLayers')[0]
        indexLevelNames = df.index.names
        parameters = {
            levelName: df.index.get_level_values(levelName)[0]
            for levelName in indexLevelNames
        }
        saveModelDir = self.getModelSavePath(parameters)
        modelName = dictToFileName(parameters)
        modelPath = os.path.join(saveModelDir, modelName)
        escaperModel = self.generateEscaperModel([neuronsPerLayer] * sharedLayers,
                                        [neuronsPerLayer] * actionLayers,
                                        [neuronsPerLayer] * valueLayers)
        if os.path.exists(saveModelDir):
            net.restoreVariables(escaperModel, modelPath)
        else:
            print(modelPath)
            exit()
            return pd.Series({"mean": None})
        escaperPolicy = self.generateEscaperPolicy(escaperModel)
        policy = self.preparePolicy(self.chaserPolicy, escaperPolicy)
        trajectories = [sampleTraj(policy) for sampleTraj in self.allSampleTrajectory]
        reward = np.mean([self.measure(trajectory) for trajectory in trajectories])
        return pd.Series({"mean": reward})


def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data','evaluateAugmentationWithinMujoco')

    # generate policy
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    chaserSavedModelDir = os.path.join(dataDir, 'wolfNNModels')
    chaserModelName = 'killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999'
    chaserNumStateSpace = 12
    numActionSpace = 8
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = net.GenerateModel(chaserNumStateSpace, numActionSpace, regularizationFactor)
    chaserModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    net.restoreVariables(chaserModel, os.path.join(chaserSavedModelDir, chaserModelName))
    approximateWolfPolicy = net.ApproximatePolicy(chaserModel, actionSpace)
    chaserPolicy = lambda state: approximateWolfPolicy(state)

    # mujoco
    physicsDynamicsPath = os.path.join(os.pardir, os.pardir, 'env', 'xmls','twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = state.GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = state.GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal,numSimulationFrames)

    # sampleTrajectory
    qPosInitNoise = 0
    qVelInitNoise = 0
    numAgent = 2
    getResetFromInitQPosDummy = lambda qPosInit: ResetUniform(physicsSimulation,qPosInit,(0, 0, 0, 0),numAgent)
    generateQPosInit = GenerateInitQPosUniform(-9.7, 9.7, isTerminal,getResetFromInitQPosDummy)
    numTrials = 1000
    allQPosInit = [generateQPosInit() for _ in range(numTrials)]
    allQVelInit = np.random.uniform(-8, 8, (numTrials, 4))
    getResetFromSampleIndex = lambda sampleIndex: ResetUniform(
        physicsSimulation, allQPosInit[sampleIndex],
        allQVelInit[sampleIndex], numAgent, qPosInitNoise,
        qVelInitNoise)
    maxRunningSteps = 25
    getSampleTrajectory = lambda sampleIndex: SampleTrajectory(
        maxRunningSteps, transit, isTerminal,
        getResetFromSampleIndex(sampleIndex),
        chooseGreedyAction)
    allSampleTrajectory = [getSampleTrajectory(sampleIndex) for sampleIndex in
                           range(numTrials)]

    # statistic reward function
    alivePenalty = 0.05
    deathBonus = -1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measure = lambda trajectory: accumulateRewards(trajectory)[0]

    # split
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionLabel']
    independentVariables['batchSize'] = [64]
    independentVariables['augment'] = [False, True]
    independentVariables['trainingStep'] = [num for num in range(0, 200001, 20000)]
    independentVariables['neuronsPerLayer'] = [256]
    independentVariables['sharedLayers'] = [17]
    independentVariables['actionLayers'] = [1]
    independentVariables['valueLayers'] = [1]
    independentVariables['initialization'] = ['uniform', 'glorot']
    independentVariables['blockSize'] = [2]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    trainedModelDir = os.path.join(dataDir, "trainedModel")
    if not os.path.exists(trainedModelDir):
        os.mkdir(trainedModelDir)
    getModelSavePath = GetSavePath(trainedModelDir, "")
    escaperNumStateSpace = 12
    generateEscaperModel = net.GenerateModel(escaperNumStateSpace, numActionSpace, regularizationFactor)
    generateEscaperPolicy = lambda model: net.ApproximatePolicy(model, actionSpace)
    qPosIndex = [0, 1]
    modifyEscaperInputState = ModifyEscaperInputState(qPosIndex)
    preparePolicy = PreparePolicy(modifyEscaperInputState)
    evaluate = EvaluateEscaperPerformance(chaserPolicy, allSampleTrajectory, measure, generateEscaperModel, generateEscaperPolicy, preparePolicy, getModelSavePath)
    statDF = toSplitFrame.groupby(levelNames).apply(evaluate)
    with open(os.path.join(dataDir, 'evaluate.pkl'), 'wb') as f:
        # statDF = pickle.load(f)
        pickle.dump(statDF, f)

    # plotbatchSize
    xStatistic = "trainingStep"
    yStatistic = "mean"
    lineStatistic = "augment"
    rowSubplotStatistic = "initialization"
    colSubplotStatistic = "blockSize"
    subplotStatistic = [rowSubplotStatistic, colSubplotStatistic]
    figsize = (12, 10)
    figure = plt.figure(figsize=figsize)
    subplotNum = len(statDF.groupby(subplotStatistic))
    numOfPlot = 1
    ylimTop = max(statDF[yStatistic])
    ylimBot = min(statDF[yStatistic]) - 1
    for subplotKey, subPlotDF in statDF.groupby(subplotStatistic):
        for linekey, lineDF in subPlotDF.groupby(lineStatistic):
            ax = figure.add_subplot(1, subplotNum, numOfPlot)
            plotDF = lineDF.reset_index()
            plotDF.plot(x=xStatistic,
                        y=yStatistic,
                        ax=ax,
                        label=linekey,
                        title="{}".format(subplotKey))
            plt.ylim(bottom=ylimBot, top=ylimTop)
        numOfPlot += 1
    plt.legend(loc='best')
    plt.subplots_adjust(wspace=0.4)
    plt.suptitle("{} vs {} episode length".format(xStatistic, yStatistic))
    figureName = "effect_augmentation_on_NNPerformance.png"
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == '__main__':
    main()