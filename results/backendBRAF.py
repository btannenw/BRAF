#!/usr/bin/env python

#  Author:  Ben Tannenwald
##  Date:   June 16, 2020
##  Purpose: Class to load data, make datasets, make tree, make forest, evaluate predictions, etc


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

#####################################
#####    Preprocessing Block    #####
#####################################

def plotOneVariable(_df, varName='', bins=10):
    """ just like what it sounds like"""
    
    plt.hist( _df[varName], bins=bins)
    plt.xlabel(varName)
    plt.show()
    
    return

def calculateCorrelation(_df, varName1='', varName2='', threshold=0.0):
    """ calculate correlation between two variables"""
    
    r = np.corrcoef( _df[varName1], _df[varName2])
    #print(r)
    p = r[0,1]
    
    if np.abs(p) > threshold:
        print("Pearson R ({},{}) = {:.3f}".format(varName1, varName2, r[0,1]))
    
    return

def minMaxScaleVariables(df):
    """return dataframe with variables scaled to [0,1] """
    _df = df.copy()
    
    for variable in _df.columns.to_list():
        if variable == 'Outcome':
            continue
            
        minVal = np.min(_df[variable])
        maxVal = np.max(_df[variable])
        print(variable, minVal, maxVal)
        _df[variable] = _df[variable].apply(lambda x: (x-minVal)/maxVal)
        
    return _df

def standardScaleVariables(df):
    """return dataframe with variables scaled to [0,1] """
    _df = df.copy()
    
    for variable in _df.columns.to_list():
        if variable == 'Outcome':
            continue
            
        meanVal = np.mean(_df[variable])
        stdVal = np.std(_df[variable])
        #print(variable, meanVal, stdVal)
        _df[variable] = _df[variable].apply(lambda x: (x-meanVal)/stdVal)
        
    return _df

def returnTestTrainSplit(_df, testingFraction=0.2):
    """takes in dataset, returns testing and training sets split by outcome"""
    
    # Outcome == 1
    _df_outcome1 = _df[_df.Outcome == 1]
    _trainLength = int((1-testingFraction)*len(_df_outcome1))
    
    _train_outcome1 = _df_outcome1[:_trainLength]
    _test_outcome1  = _df_outcome1[_trainLength:]
    print(len(_df_outcome1), _trainLength, len(_train_outcome1), len(_test_outcome1))

    # Outcome == 0
    _df_outcome0 = _df[_df.Outcome == 0]
    _trainLength = int((1-testingFraction)*len(_df_outcome0))
    
    _train_outcome0 = _df_outcome0[:_trainLength]
    _test_outcome0  = _df_outcome0[_trainLength:]
    print(len(_df_outcome0), _trainLength, len(_train_outcome0), len(_test_outcome0))

    return _train_outcome1, _test_outcome1, _train_outcome0, _test_outcome0


def cleanAndImpute( _df):
    """ impute values using simple pdf from histogram. WARNING: this ignores correlations between variables"""

    _df_clean = _df.copy()
    for variable in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' ]:
        _df_clean[variable] = _df_clean[variable].apply( lambda x: imputeFromCDF(x, variable, _df))

    return _df_clean

def imputeFromCDF(x, variable, _df_raw):
    """ generate cdf from input distibution and return random value pulled from cdf if x==0"""
 
    # return immediately if we don't need to impute                             
    if x != 0:
        return x

    # Else, create variable distribution                                        
    _dist = _df_raw.copy()[_df_raw[variable]!=0][variable]
    hist, bins = np.histogram(_dist, bins=25)

    # Create CDF                                                                
    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]

    # generate random and find place in CDF                                     
    rand = np.random.rand()
    rand_bin = np.searchsorted(cdf, rand)
    rand_cdf = bin_midpoints[rand_bin]

    #print(rand, rand_bin, rand_cdf)                                            

    return rand_cdf


#####################################
#####     Critical Set Block    #####
#####################################

def returnDistance(testRow, seedRow):
    """ calculate distance between test point and seed point, then return"""
    
    distance = 0
    
    for variable in testRow.keys().to_list():
        distance += (testRow[variable]-seedRow[variable])**2
    
    return np.sqrt(distance)

def applyNeighborDistance(seedRow, df):
    """add column of distance to original point"""
    _df = df.copy()
    
    _df['neighborDistance'] = _df.apply( lambda x: returnDistance(x, seedRow), axis=1)
    
    return _df
    

def buildCriticalSet( _train0, _train1, k=5):
    """ build critical set using k-nearest neighbor algorithm"""
    
    _Tc = _train1.copy()
    _T = _train0.copy()
    _T = _T.append(_train1.copy())

    for iRow in range(0, len(_T)):
        if _T.iloc[iRow].Outcome == 1:
            # calculate distance for selected rows and apply as new column
            _tempT = applyNeighborDistance( _T.iloc[iRow], _T)
            # keep the top 6 (includes self + top5)
            _tempTop6 = _tempT.nsmallest(k+1, 'neighborDistance')
            _tempTop6 = _tempTop6.drop(columns=['neighborDistance'])
            #print(iRow, _tempTop6)
            
            # add top to critical set
            _Tc = _Tc.append(_tempTop6)

            # drop duplicates
            _Tc.drop_duplicates(keep='first', inplace=True)
                  
    return _Tc


#####################################
#####    Random Forest Block    #####
#####################################

def trainRandomForest(_train, max_depth, min_size, nTrees, sample_size=1.0, name=''):
    """build random forest given user hyperparameters"""

    _trees = []
    if name != '':
        print("Training {} Forest".format(name))

    for i in np.arange(0, nTrees):
                
        # randomly shuffle data with repeats so each tree has 'unique' inputs
        _trainSample = _train.sample(frac=sample_size, replace=True)
        
        # select subset of columns for each tree. this could be more compact
        _treeOutcome = _trainSample.Outcome
        _treeData = _trainSample.drop(columns=['Outcome'])
        _nVars = int(np.ceil(np.sqrt(len(_treeData.columns))))
        _treeData = _treeData.sample(n=_nVars, axis=1)
        _treeData['Outcome']= _treeOutcome
        
        _tree = buildTree( _treeData, max_depth, min_size)
        #printTree(_tree)
        _trees.append( _tree)

    return _trees


def evaluateForest(_name, _forest, _df):
    """evaluate forest"""

    _df_pred = makeBaggingPredictions(_df, _forest)
    
    acc, prec, rec = simpleMetrics(_df_pred, _name)
    
    return acc, prec, rec


def buildTree( _df, max_depth, min_size):
    """ build a single decision tree """
    
    _tree = getBestSplit( _df)
    fillTree( _tree, max_depth, min_size, 1)
    
    return _tree


def fillTree(node, max_depth, min_size, depth):
    """Create child splits for a node or make terminal"""

    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if len(left)==0 or len(right)==0:
        _merge = left.copy()
        _merge = _merge.append(right.copy())
        node['left'] = node['right'] = terminalNode(_merge)
        return
    
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = terminalNode(left), terminalNode(right)
        return
    
    # process left child
    if len(left) <= min_size:
        node['left'] = terminalNode(left)
    else:
        node['left'] = getBestSplit(left)
        fillTree(node['left'], max_depth, min_size, depth+1)
        
    # process right child
    if len(right) <= min_size:
        node['right'] = terminalNode(right)
    else:
        node['right'] = getBestSplit(right)
        fillTree(node['right'], max_depth, min_size, depth+1)
        
    return


def getBestSplit( _df):
    """ get best split point in dataset """
    
    _best_variable = -1
    _best_split = -1
    _best_gini = 9999
    _best_groups = []
    
    # iterate over features to find best split
    for variable in _df.columns.to_list():
        if variable == 'Outcome':
            continue
        
        _splitVals = set(np.sort(_df[variable].values))
        for _split in _splitVals:
            _groups = list( (_df[ _df[variable] < _split], _df[ _df[variable] >= _split] ))
            _classes = _df.Outcome.value_counts().keys().to_list()
    
            _gini = giniIndex( _groups, _classes)
            if _gini < _best_gini:
                _best_variable = variable
                _best_split = _split
                _best_gini = _gini
                _best_groups = _groups

    #print("Best Split using {}, cut: {:.3f}, gini: {:.3f}, n_left: {}, n_right: {}".format(_best_variable, _best_split, _best_gini, len(_best_groups[0]), len(_best_groups[1])))   
 
    return {'variable':_best_variable, 'split':_best_split, 'groups':_best_groups}


def giniIndex(group_dfs, classes):
    # count all samples at split point
    n_instances = float(sum([len(group_df) for group_df in group_dfs]))
    
    # sum weighted Gini index for each group
    gini = 0.0
    for group_df in group_dfs:
        size = float(len(group_df))
    
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = len( group_df[ group_df.Outcome == class_val] ) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
        
    return gini 


def terminalNode(_df):
    """ return value of most common outcome in a node. this is node prediction"""
    return _df.Outcome.value_counts().idxmax()


def printTree(node, depth=0):
    """print decision tree"""
    
    if isinstance(node, dict):
        print('{} [{} < {:.3f}]'.format(depth, node['variable'], (node['split']+1) ))
        printTree(node['left'], depth+1)
        printTree(node['right'], depth+1)
    else:
        print('{} [{}]'.format(depth, node))
        
        
def predict(row, node):
    """make a prediction for a single row given a decision tree"""

    if row[node['variable']] < node['split']:
        if isinstance(node['left'], dict):
            return predict(row, node['left'])
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(row, node['right'])
        else:
            return node['right']


def baggingPredict(row, trees, prob=False):
    """make a prediction for a single row given bagged trees"""

    # calculate predictions of each tree in forest
    _treePredictions = [ predict(row, tree) for tree in trees]
    
    # *** A. Nominal: forest prediction is majority vote of constituent trees
    _forestPrediction = max( set(_treePredictions), key=_treePredictions.count)

    # *** B. Probability: forest prediction is sum of tree votes / total trees (~probabalistic)
    if prob==True:
        _forestPrediction = sum(_treePredictions) / len(_treePredictions)
    
    return _forestPrediction


def makePredictions(_df, tree):
    """add column of predictions"""
    
    _df = _df.copy()
    _df['PredictedOutcome'] = _df.apply( lambda x: predict(x, tree), axis=1)
        
    return _df


def makeBaggingPredictions(_df, trees):
    """add column for bagged prediction"""
    
    _df = _df.copy()
    _df['PredictedOutcome'] = _df.apply( lambda x: baggingPredict(x, trees), axis=1)
    _df['PredictedOutcomeProb'] = _df.apply( lambda x: baggingPredict(x, trees, prob=True), axis=1)
    
    return _df

def simpleMetrics(_df, name=''):
    """return accuracy, precision, recall"""
    
    #print( _df.PredictedOutcome.value_counts())
    
    _correct = len( _df[_df.Outcome == _df.PredictedOutcome])
    _total = len(_df)
    _truePositive  = len( _df[ (_df.PredictedOutcome == 1) & (_df.Outcome==1)])
    _falsePositive = len( _df[ (_df.PredictedOutcome == 1) & (_df.Outcome==0)])
    _trueNegative  = len( _df[ (_df.PredictedOutcome == 0) & (_df.Outcome==0)])
    _falseNegative = len( _df[ (_df.PredictedOutcome == 0) & (_df.Outcome==1)])
    
    _accuracy  = 100 * ( _correct / _total)
    _precision = 100 * ( _truePositive / (_truePositive + _falsePositive)) 
    _recall    = 100 * ( _truePositive / (_truePositive + _falseNegative))
    
    #if name!='':
    #    print("{} Results".format(name))
    #print( "{:.2f}% Accuracy, {:.2f}% Precision, {:.2f}% Recall".format( _accuracy, _precision, _recall))

    return _accuracy, _precision, _recall


def drawPredictionPlot(_df, title='', saveDir='', bins=10):
    """draw prediction plot"""

    weights0 = np.ones_like(_df[_df.Outcome==0].PredictedOutcomeProb) / len (_df[_df.Outcome==0])
    weights1 = np.ones_like(_df[_df.Outcome==1].PredictedOutcomeProb) / len (_df[_df.Outcome==1])

    plt.hist( _df[ _df.Outcome==0].PredictedOutcomeProb, bins=bins, alpha=0.5, weights=weights0, label='Outcome = 0')
    plt.hist( _df[ _df.Outcome==1].PredictedOutcomeProb, bins=bins, alpha=0.5, weights=weights1, label='Outcome = 1')
    
    plt.ylabel('Normalized Entries')
    plt.xlabel('Diabetes Prediction')
    if title != '':
        plt.title(title)
    
    plt.legend(loc='upper right')
    
    # store figure copy for later saving
    fig = plt.gcf()
    
    # draw interactively
    plt.show()
    
    #save an image file
    if(saveDir != ''):    
        fig.savefig( saveDir + '/' + title+'_probScore.png', bbox_inches='tight' )
    
    #close out
    plt.close(fig)

    return


def plotPRC(name, _df, curve_color='blue', drawPlot=True):
    """ plot Precision-Recall curve, calculate AUPRC, etc"""
    
    rand = len( _df[_df.Outcome==1]) / len(_df)
    plt.plot([0, 1], [rand, rand], color="black", linestyle='--')

    plt.title("{} Precision-Recall curve".format(name))                                                                                                       
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(rand-0.05, 1)
    plt.tick_params(axis="both", direction="in")

    total_True  = len(_df[ _df.Outcome==1])
    recall = []
    precision = []
        
    nSteps = 50
    for cut in np.linspace( min(_df.PredictedOutcomeProb), 1, nSteps):
        truePositive  = len(_df[ (_df.Outcome==1) & (_df.PredictedOutcomeProb >= cut) ])
        falsePositive = len(_df[ (_df.Outcome==0) & (_df.PredictedOutcomeProb >= cut) ])

        if truePositive+falsePositive == 0: # protection for zero
            continue
        else:
            precision.append( truePositive / (truePositive+falsePositive) )
            
        recall.append( truePositive / total_True )
        
    auprc = np.sum(precision)/nSteps
    label = "{} (AUPRC={:.3f})".format(name, auprc)

    if drawPlot:
        plt.plot(recall, precision, label=label, color=curve_color) # signal eff vs background eff
        plt.legend(loc="center left")
    
    return auprc


def plotROC(name, _df, curve_color='blue', drawPlot=True):
    """ plot ROC curve, calculate AUC, etc"""
    
    plt.plot( [[0, 0], [1, 1]], color="black", linestyle="--")
    plt.title("{} ROC curve".format(name))                                                                                                       
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis="both", direction="in")

    total_True  = len(_df[ _df.Outcome==1])
    total_False = len(_df[ _df.Outcome==0])
    fpr = []
    tpr = []
    
    nSteps = 50
    for cut in np.linspace( 0, 1, nSteps):

        fpr.append( 1 - len(_df[ (_df.Outcome==0) & (_df.PredictedOutcomeProb < cut) ]) / total_False )
        tpr.append( len(_df[ (_df.Outcome==1) & (_df.PredictedOutcomeProb >= cut) ]) / total_True )
        
    auc = np.sum(tpr)/nSteps
    label = "{} (AUC={:.3f})".format(name, auc)

    if drawPlot:
        plt.plot(fpr, tpr, label=label, color=curve_color) # signal eff vs background eff
        plt.legend(loc="lower right")
    
    return auc

def compareCurves( _nom_preds, _crit_preds, _braf_preds, iFold, curve='AUC', name='train', savePlot=True):
    """ either save plot with overlaid curves or just return the auc/auprc values"""

    if curve == 'AUC':
         _nom  = plotROC('Nominal', _nom_preds, 'blue', savePlot)
         _crit = plotROC('Critical-Only', _crit_preds, 'red', savePlot)
         _braf = plotROC('BRAF', _braf_preds, 'green', savePlot)
    elif curve == 'AUPRC':
         _nom  = plotPRC('Nominal', _nom_preds, 'blue', savePlot)
         _crit = plotPRC('Critical-Only', _crit_preds, 'red', savePlot)
         _braf = plotPRC('BRAF', _braf_preds, 'green', savePlot)

    # make directory if needed
    outDir = 'fold_{}'.format(iFold)
    if ( not os.path.exists( outDir )):
        #print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(args.outputDir))
        os.system("mkdir {0}".format( outDir ))

    #store fig
    fig = plt.gcf()
    
    # draw interactively
    if name=='train':
        plt.show()
    
    # save plot if asked
    if savePlot:
        #save image
        fig.savefig( outDir + '/' + name + '_' + curve+'_fold{}.png'.format(iFold), bbox_inches='tight' )
    
    #close out
    plt.close(fig)

    return _nom, _crit, _braf


#####################################
#####        k-Fold Block       #####
#####################################

def kfoldResultsDict():
    """helper function to create dictionary for storing k-fold results"""

    _results = dict( nominal_train={}, critical_train={}, BRAF_train={}, nominal={}, critical={}, BRAF={} )
    for cat in _results:
        _results[cat] = dict( accuracy=[], precision=[], recall=[], auc=[], auprc=[])

    return _results


def storeFoldResults(res, name, acc, prec, recall, auc, auprc, iFold, save=True):
    """helper function to store results and return"""
    
    res[name]['accuracy'].append( acc )
    res[name]['precision'].append( prec )
    res[name]['recall'].append( recall )
    res[name]['auc'].append( auc )
    res[name]['auprc'].append( auprc )

    if save:
        # make directory if needed
        outDir = 'fold_{}'.format(iFold)
        if ( not os.path.exists( outDir )):
            #print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(args.outputDir))
            os.system("mkdir {0}".format( outDir ))

        outFile = open('{}/{}_results.txt'.format(outDir, name), 'w')
        outFile.write('accuracy: {:.2f}\n'.format(acc))
        outFile.write('precision: {:.2f}\n'.format(prec))
        outFile.write('recall: {:.2f}\n'.format(recall))
        outFile.write('auc: {:.3f}\n'.format(auc))
        outFile.write('auprc: {:.3f}\n'.format(auprc))
        outFile.close()
        
    return

def compareAcrossFolds(_results, metric):
    """helper function to compare nominal/critical/BRAF across folds"""
    
    for cat in ['nominal_train', 'critical_train', 'BRAF_train', 'nominal', 'critical', 'BRAF']:
        _mean = np.mean(_results[cat][metric])
        _std = np.std(_results[cat][metric])
        print('{} ({}): {:.2f} +/- {:.2f}'.format(metric, cat, _mean, _std))

    return
