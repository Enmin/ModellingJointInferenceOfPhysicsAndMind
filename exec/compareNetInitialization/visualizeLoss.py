import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import pickle
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
import src.neuralNetwork.visualizeNN as visualizeNN
from exec.trajectoriesSaveLoad import conditionDfFromParametersDict, GetSavePath


class ReadReport:
    def __init__(self, keyToRead, getModelSavePath):
        self.keyToRead = keyToRead
        self.getModelSavePath = getModelSavePath

    def __call__(self, oneConditionDf):
        nnStructure = oneConditionDf.index.get_level_values('nnStructure')[0]
        learningRate = oneConditionDf.index.get_level_values('learningRate')[0]
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        initialization = oneConditionDf.index.get_level_values('initialization')[0]
        pathParams = {'structure': nnStructure, 'learningRate': learningRate, 'trainSteps': trainSteps, 'initialization': initialization}
        reportSavePath = self.getModelSavePath(pathParams) + ".report.pickle"
        with open(reportSavePath, 'rb') as f:
            reportDict = pickle.load(f)
        if self.keyToRead == 'variables':
            varDict = reportDict['variables']
            flattenIndexToData = lambda indexToData: {index: data.flatten() for index, data in indexToData.items()}
            flattenedVarDict = {varName: flattenIndexToData(dataDict) for varName, dataDict in varDict.items()}
            reportDict['variables'] = flattenedVarDict
        return pd.Series(reportDict[self.keyToRead])


def drawLossLine(ax, reportDf, evalName):
    for struct, structGrp in reportDf.groupby('initialization'):
        structGrp.index = structGrp.index.droplevel('initialization')
        structGrp.plot(ax=ax, y=evalName, label=struct,
                       marker='o', markersize=2)


def main():
    manipulatedVariables = OrderedDict()
    dataDir = os.path.join(os.pardir, os.pardir, 'data', 'compareNetInitialization')
    nnStructures = [((128,)*16,(128,),(128,))]
    manipulatedVariables['nnStructure'] = nnStructures
    manipulatedVariables['learningRate'] = [1e-4]
    manipulatedVariables['initialization'] = ['normal', 'he', 'glorot', 'uniform']
    manipulatedVariables['trainSteps'] = [1] + list(range(2000, 20000, 2000))
    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    learningRateDecayRate = 1
    learningRateDecayInterval = 5000
    regularizationFactor = 1e-4
    lossCoefs = (1, 1)
    miniBatchSize = 64
    numTrainingTrajectories = 6000
    NNFixedParameters = {'lossCoefs': lossCoefs, 'miniBatch': miniBatchSize,
                         'numTrajectories': numTrainingTrajectories}
    NNModelSaveDirectory = os.path.join(dataDir, 'trainedNNs')
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    keyInReport = 'evaluations'
    readReport = ReadReport(keyInReport, getModelSavePath)
    levelNames = list(manipulatedVariables.keys())
    reportDf = toSplitFrame.groupby(levelNames).apply(readReport)
    print(reportDf)

    evalNames = ['loss', 'actionLoss', 'valueLoss']
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f'#trajectories={numTrainingTrajectories}, lossCoefs={lossCoefs}, regFactor={regularizationFactor}, '
                 f'lrDecay={learningRateDecayRate}, decayInterval={learningRateDecayInterval}')
    nRows = len(evalNames)
    nCols = len(manipulatedVariables['learningRate'])
    subplotIndex = 1
    axs = []
    for evalName in evalNames:
        for lr, lrGrp in reportDf.groupby('learningRate'):
            lrGrp.index = lrGrp.index.droplevel(['learningRate','nnStructure'])
            ax = fig.add_subplot(nRows, nCols, subplotIndex)
            axs.append(ax)
            subplotIndex += 1
            ax.text(0.01, 0.99, f'lr={lr:.0E}, {evalName}', ha='left', va='top', transform=ax.transAxes, fontdict={'size': 8})
            ax.set_ylabel(evalName)
            drawLossLine(ax, lrGrp, evalName)
            plt.legend(loc='best', prop={'size': 6})
            plt.ylim(bottom=0, top=5)
    # visualizeNN.syncLimits(axs)
    plt.savefig(os.path.join(dataDir, 'loss.png'))


if __name__ == '__main__':
    main()
