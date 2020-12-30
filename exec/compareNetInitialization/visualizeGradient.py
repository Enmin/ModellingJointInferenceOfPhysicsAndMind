import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
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
        lr = oneConditionDf.index.get_level_values('learningRate')[0]
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        initialization = oneConditionDf.index.get_level_values('initialization')[0]
        pathParams = {'structure': nnStructure, 'learningRate': lr, 'trainSteps': trainSteps, 'initialization': initialization}
        reportSavePath = self.getModelSavePath(pathParams) + '.report.pickle'
        with open(reportSavePath, 'rb') as f:
            reportDict = pickle.load(f)
        if self.keyToRead == 'variables':
            varDict = reportDict['variables']
            flattenIndexToData = lambda indexToData: {index: data.flatten() for index, data in indexToData.items()}
            flattenedVarDict = {varName: flattenIndexToData(dataDict) for varName, dataDict in varDict.items()}
            reportDict['variables'] = flattenedVarDict
        return pd.Series(reportDict[self.keyToRead])


def normGradient(oneConditionDf, gradName):
    nnStructure = oneConditionDf.index.get_level_values('nnStructure')[0]
    lr = oneConditionDf.index.get_level_values('learningRate')[0]
    trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
    initialization = oneConditionDf.index.get_level_values('initialization')[0]
    layerIndexToGrad = oneConditionDf[gradName].values[0]
    layerIndexToNorm = {layerIndex: np.linalg.norm(grad) for layerIndex, grad in layerIndexToGrad.items()}
    return pd.Series({f'{gradName}Norm': layerIndexToNorm})


def main():
    manipulatedVariables = OrderedDict()
    nnStructures = [((128,)*16,(128,),(128,))]
    manipulatedVariables['nnStructure'] = nnStructures
    manipulatedVariables['learningRate'] = [1e-4]
    manipulatedVariables['initialization'] = ['uniform', 'normal', 'he', 'glorot']
    manipulatedVariables['trainSteps'] = [1] + list(range(2000, 500000, 2000))
    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    regularizationFactor = 1e-4
    learningRateDecayRate = 1
    learningRateDecayInterval = 5000
    lossCoefs = (1, 1)
    miniBatchSize = 64
    numTrainingTrajectories = 6000
    NNFixedParameters = {'lossCoefs': lossCoefs, 'miniBatch': miniBatchSize,
                         'numTrajectories': numTrainingTrajectories,}
    dataDir = os.path.join(os.pardir, os.pardir, 'data', 'compareNetInitialization')
    NNModelSaveDirectory = os.path.join(dataDir, 'trainedNNs')
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    keyInReport = 'variables'
    readReport = ReadReport(keyInReport, getModelSavePath)
    levelNames = list(manipulatedVariables.keys())
    reportDf = toSplitFrame.groupby(levelNames).apply(readReport)
    print(reportDf)
    gradToPlot = 'weightGradient'
    reportDfWithNorm = reportDf.groupby(levelNames).apply(normGradient, gradToPlot)
    print(reportDfWithNorm)

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f'#trajectories={numTrainingTrajectories}, lossCoefs={lossCoefs}, regFactor={regularizationFactor}, '
                 f'lrDecay={learningRateDecayRate}, decayInterval={learningRateDecayInterval}')
    nRows = len(manipulatedVariables['initialization'])
    nCols = len(manipulatedVariables['learningRate'])
    subplotIndex = 1
    axs = []
    sectionNameToDepth = {"shared": 16,
                          "action": 1}
    layerIndices = visualizeNN.indexLayers(sectionNameToDepth)
    layerIndicesToPlot = layerIndices[1:-1]
    if len(layerIndicesToPlot) > 8:
        layerIndicesToPlot = layerIndicesToPlot[::2]
    for struct, structGrp in reportDfWithNorm.groupby('initialization'):
        structGrp.index = structGrp.index.droplevel(['initialization','nnStructure'])
        for lr, lrGrp in structGrp.groupby('learningRate'):
            lrGrp.index = lrGrp.index.droplevel('learningRate')
            trainStepsList = lrGrp.index.tolist()
            ax = fig.add_subplot(nRows, nCols, subplotIndex)
            ax.text(0, 1, f'lr= {lr}\
                    initialization= {struct}',
                    ha='left', va='bottom', transform=ax.transAxes, fontdict={'size': 8})
            ax.set_ylabel(f'{gradToPlot}Norm')
            ax.set_yscale('log')
            axs.append(ax)
            subplotIndex += 1
            for layerIndex in layerIndicesToPlot:
                norms = [lrGrp[f'{gradToPlot}Norm'][trainSteps][layerIndex] for trainSteps in trainStepsList]
                ax.plot(trainStepsList, norms, label=layerIndex)
            plt.legend(loc='best')
    visualizeNN.syncLimits(axs)

    nnStructureToPlot = nnStructures[0]
    sharedWidths, actionLayerWidths, valueLayerWidths = nnStructureToPlot
    sectionNameToDepth = {"shared": len(sharedWidths), "action": len(actionLayerWidths) + 1}
    layerIndices = visualizeNN.indexLayers(sectionNameToDepth)
    useAbs = True
    useLog = True
    histBase = 1e-10
    binCount = 50
    varNameToPlot = 'weightGradient'

    # histogram
    histOn = False
    if histOn:
        getDfData = lambda trainStepsDf, trainSteps: np.concatenate(list(trainStepsDf[varNameToPlot][nnStructureToPlot]
                                                                         [trainSteps].values()))
        rawDataList = [getDfData(trainStepsDf, trainSteps) for trainSteps, trainStepsDf in reportDf.groupby('trainSteps')]
        rawAllData = np.concatenate(rawDataList)
        allData = np.abs(rawAllData) if useAbs else rawAllData
        _, bins = visualizeNN.logHist(allData, binCount, histBase) if useLog else np.histogram(allData, bins=binCount)
        plotHist = visualizeNN.PlotHist(useAbs, useLog, histBase, bins)

        plotRows = len(manipulatedVariables['trainSteps'])
        plotCols = len(layerIndices)
        histFig = plt.figure()
        histFig.suptitle(f"Histograms of {varNameToPlot}, structure = {nnStructureToPlot}")
        histGS = histFig.add_gridspec(plotRows, plotCols)
        axs = []
        for trainSteps, trainStepsDf in reportDf.groupby('trainSteps'):
            layerIndexToData = trainStepsDf[varNameToPlot][nnStructureToPlot][trainSteps]
            for layerIndex, data in layerIndexToData.items():
                ax = histFig.add_subplot(histGS[manipulatedVariables['trainSteps'].index(trainSteps),
                                                layerIndices.index(layerIndex)])
                axs.append(ax)
                ax.text(0, 1, f"step#{trainSteps}\n{layerIndex[0]}/{layerIndex[1]}",
                        ha='left', va='top', transform=ax.transAxes, fontdict={'size': 8})
                plotHist(ax, data)
        visualizeNN.syncLimits(axs)

    # bar
    barOn = False
    if barOn:
        plotBars = visualizeNN.PlotBars(useAbs, useLog)

        barFig = plt.figure()
        barFig.suptitle(f"Bar plots of {varNameToPlot}, structure: {nnStructureToPlot}")
        figRowNum = len(manipulatedVariables['trainSteps'])
        figColNum = 1
        subplotIndex = 1
        axs = []
        for trainSteps, trainStepsDf in reportDf.groupby('trainSteps'):
            layerIndexToData = trainStepsDf[varNameToPlot][nnStructureToPlot][trainSteps]
            rawDataList = list(layerIndexToData.values())
            dataList = [np.abs(data) for data in rawDataList] if useAbs else rawDataList
            indexToStr = lambda sectionName, layerIndex: f"{sectionName}/{layerIndex}"
            labelList = [indexToStr(section, layer) for section, layer in layerIndices]
            ax = barFig.add_subplot(figRowNum, figColNum, subplotIndex)
            axs.append(ax)
            subplotIndex += 1
            ax.text(0, 1, f"step#{trainSteps}", ha='left', va='top', transform=ax.transAxes, fontdict={'size': 8})
            statsOnPlot = [(np.mean(data), np.std(data), np.min(data), np.max(data)) for data in dataList]
            means, stds, mins, maxs = [np.array(stats) for stats in zip(*statsOnPlot)]
            plotBars(ax, means, stds, mins, maxs, labelList)
        visualizeNN.syncLimits(axs)

    plt.savefig(os.path.join(dataDir, 'gradient.png'))


if __name__ == '__main__':
    main()
