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
from evaluateByStateDimension.preprocessData import ZeroValueInState
import policyValueNet as net
import mujoco_py as mujoco
import state
import numpy as np
import pandas as pd
import pickle


def dictToFileName(parameters):
    sortedParameters = sorted(parameters.items())
    nameValueStringPairs = [
        parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters
    ]
    modelName = '_'.join(nameValueStringPairs).replace(" ", "")
    return modelName


class PreparePolicy:

    def __init__(self):
        return

    def __call__(self, chaserPolicy, escaperPolicy):
        policy = lambda state: [
            escaperPolicy(state),
            chaserPolicy(state)
        ]
        return policy


class EvaluateEscaperPerformance:

    def __init__(self, chaserPolicy, allSampleTrajectory, measure,
                 getGenerateEscaperModel, generateEscaperPolicy,
                 getPreparePolicy, getModelSavePath):
        self.chaserPolicy = chaserPolicy
        self.allSampleTrajectory = allSampleTrajectory
        self.getGenerateEscaperModel = getGenerateEscaperModel
        self.generateEscaperPolicy = generateEscaperPolicy
        self.getPreparePolicy = getPreparePolicy
        self.getModelSavePath = getModelSavePath
        self.measure = measure

    def __call__(self, df):
        structure = df.index.get_level_values('structure')[0]
        shared, action, value = structure
        indexLevelNames = df.index.names
        parameters = {
            levelName: df.index.get_level_values(levelName)[0]
            for levelName in indexLevelNames
        }
        saveModelPath = self.getModelSavePath(parameters)
        generateEscaperModel = self.getGenerateEscaperModel(12)
        escaperModel = generateEscaperModel(shared, action, value)
        net.restoreVariables(escaperModel, saveModelPath)
        preparePolicy = self.getPreparePolicy()
        escaperPolicy = self.generateEscaperPolicy(escaperModel)
        policy = preparePolicy(self.chaserPolicy, escaperPolicy)
        trajectories = [
            sampleTraj(policy) for sampleTraj in self.allSampleTrajectory
        ]
        reward = np.mean(
            [self.measure(trajectory) for trajectory in trajectories])
        return pd.Series({"mean": reward})


def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data',
                           'compareNetStructures')

    # generate policy
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7),
                   (0, -10), (7, -7)]
    chaserSavedModelDir = os.path.join(dataDir, 'wolfNNModels')
    chaserModelName = 'killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999'
    chaserNumStateSpace = 12
    numActionSpace = 8
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = net.GenerateModel(chaserNumStateSpace, numActionSpace,
                                      regularizationFactor)
    chaserModel = generateModel(sharedWidths, actionLayerWidths,
                                valueLayerWidths)
    net.restoreVariables(chaserModel,
                         os.path.join(chaserSavedModelDir, chaserModelName))
    approximateWolfPolicy = net.ApproximatePolicy(chaserModel, actionSpace)
    chaserPolicy = lambda state: approximateWolfPolicy(state)

    # mujoco
    physicsDynamicsPath = os.path.join(os.pardir, os.pardir, 'env', 'xmls',
                                       'twoAgents.xml')
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
    transit = TransitionFunction(physicsSimulation, isTerminal,
                                 numSimulationFrames)

    # sampleTrajectory
    qPosInitNoise = 0
    qVelInitNoise = 0
    numAgent = 2
    getResetFromInitQPosDummy = lambda qPosInit: ResetUniform(
        physicsSimulation, qPosInit, (0, 0, 0, 0), numAgent)
    generateQPosInit = GenerateInitQPosUniform(-9.7, 9.7, isTerminal,
                                               getResetFromInitQPosDummy)
    numTrials = 100
    allQPosInit = [generateQPosInit() for _ in range(numTrials)]
    allQVelInit = np.random.uniform(-8, 8, (numTrials, 4))
    getResetFromSampleIndex = lambda sampleIndex: ResetUniform(
        physicsSimulation, allQPosInit[sampleIndex], allQVelInit[sampleIndex],
        numAgent, qPosInitNoise, qVelInitNoise)
    maxRunningSteps = 25
    getSampleTrajectory = lambda sampleIndex: SampleTrajectory(
        maxRunningSteps, transit, isTerminal,
        getResetFromSampleIndex(sampleIndex), chooseGreedyAction)
    allSampleTrajectory = [
        getSampleTrajectory(sampleIndex) for sampleIndex in range(numTrials)
    ]

    # statistic reward function
    alivePenalty = 0.05
    deathBonus = -1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measure = lambda trajectory: accumulateRewards(trajectory)[0]

    # split
    structureList = [((128,128,128,128,128,128,128,128,),(128,),(128,)),
                    # ((16, 16, 32, 32, 64, 64, 128, 128,), (128,), (128,)),
                    # ((128,128,64,64,32,32,16,16,),(16,),(16,)),
                    # ((128,128,64,64,64,64,64,64),(64,),(64,)),
                    # ((128, 128, 64, 64, 32, 32, 32, 32), (32,), (32,)),
                    # ((128, 128, 128, 128, 64, 64, 64, 64), (64,), (64,)),]
                    ((256, 256, 128, 128, 64, 64, 32, 32), (32,), (32,)),
                    ((256, 256, 128, 128, 64, 64, 32, 32), (16,), (16,)),]
                    # ((64, 64, 64, 128, 128, 64, 64, 64), (64,), (64,)),
                    # ((32, 64, 64, 128, 128, 64, 64, 32), (32,), (32,)),]
    independentVariables = OrderedDict()
    independentVariables['numTrajectories'] = [6000]
    independentVariables['miniBatch'] = [64]
    independentVariables['trainSteps'] = [
        num for num in range(0, 500001, 50000)
    ]
    independentVariables['regularization'] = [1e-4]
    independentVariables['lrDecay'] = [1]
    independentVariables['lrDecayInterval'] = [5000]
    independentVariables['lossCoefs'] = [(1,1)]
    independentVariables['learningRate'] = [1e-4]
    independentVariables['structure'] = structureList

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    trainedModelDir = os.path.join(dataDir, "trainedNNs")
    if not os.path.exists(trainedModelDir):
        os.mkdir(trainedModelDir)
    getModelSavePath = GetSavePath(trainedModelDir, "")
    getGenerateEscaperModel = lambda numStateSpace: net.GenerateModel(
        numStateSpace, numActionSpace, regularizationFactor)
    generateEscaperPolicy = lambda model: net.ApproximatePolicy(
        model, actionSpace)
    qPosIndex = [0, 1]
    zeroIndex = [2, 3, 6, 7]
    zeroValueInState = ZeroValueInState(zeroIndex)
    evaluate = EvaluateEscaperPerformance(chaserPolicy, allSampleTrajectory,
                                          measure, getGenerateEscaperModel,
                                          generateEscaperPolicy, PreparePolicy,
                                          getModelSavePath)
    statDF = toSplitFrame.groupby(levelNames).apply(evaluate)
    with open(os.path.join(dataDir, "evaluate.pkl"), 'wb') as f:
        pickle.dump(statDF, f)
        # statDF = pickle.load(f)
    # print(statDF)
    # exit()
    # plotbatchSize
    xStatistic = "trainSteps"
    yStatistic = "mean"
    lineStatistic = "structure"
    subplotStatistic = "learningRate"
    figsize = (12, 10)
    figure = plt.figure(figsize=figsize)
    subplotNum = len(statDF.groupby(subplotStatistic))
    numOfPlot = 1
    ylimTop = max(statDF[yStatistic])
    ylimBot = min(statDF[yStatistic]) - 0.2
    for subplotKey, subPlotDF in statDF.groupby(subplotStatistic):
        for linekey, lineDF in subPlotDF.groupby(lineStatistic):
            ax = figure.add_subplot(1, subplotNum, numOfPlot)
            plotDF = lineDF.reset_index()
            plotDF.plot(x=xStatistic,
                        y=yStatistic,
                        ax=ax,
                        label=linekey,
                        title="{}:{}".format(subplotStatistic, subplotKey))
            plt.ylim(bottom=ylimBot, top=ylimTop)
        numOfPlot += 1
    plt.legend(loc='best')
    plt.subplots_adjust(hspace=0.4, wspace=0.6)
    plt.suptitle("batchSize:64, trajectory:6000")
    figureName = "DownSampling-rewards.png"
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == '__main__':
    main()
