import math
from datetime import datetime
import mariadb
import mysql.connector
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, \
    PowerTransformer, Normalizer
import time
import numpy as np
from sys import platform
import multiprocessing as mp
from sklearn.utils.validation import column_or_1d
# from pai4sk import RandomForestClassifier
# from pai4sk import DecisionTreeClassifier
from pai4sk import SupportVectorMachine
# from pai4sk import LogisticRegression
from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression


def getConnection():
    connection = mariadb.connect(
        pool_name="store",
        pool_size=32,
        host="store.usr.user.hu",
        user="mki",
        password="pwd"
    )
    return connection


def getScalers():
    scalers = {'StandardScaler': StandardScaler(),
               'MinMaxScaler': MinMaxScaler(),
               'MaxAbsScaler': MaxAbsScaler(),
               'RobustScaler': RobustScaler(),
               'QuantileTransformer-Normal': QuantileTransformer(output_distribution='normal'),
               'QuantileTransformer-Uniform': QuantileTransformer(output_distribution='uniform'),
               'Normalizer': Normalizer(),
               'NoScaler': None
               }
    return scalers


def getModels():
    models = {
        # 'Logistic Regression GPU': LogisticRegression(use_gpu=True, device_ids=[0, 1]),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1),
        # 'DecisionTree GPU': DecisionTreeClassifier(use_gpu=True, use_histograms=True),
        'DecisionTree': DecisionTreeClassifier(),
        'GaussianNB': GaussianNB(),
        'SupportVectorMachine GPU': SupportVectorMachine(use_gpu=True, device_ids=[0, 1]),
        # 'Random Forest GPU': RandomForestClassifier(use_gpu=True, gpu_ids=[0, 1], use_histograms=True),
        'Random Forest': RandomForestClassifier(n_jobs=-1),
        # 'MLP': MLPClassifier(),
        'Light GBM': LGBMClassifier(n_jobs=-1),
        # 'XGBoost': XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    }
    return models


def getRandomSamplers():
    samplers = {
        'UnderSampler': RandomUnderSampler(sampling_strategy=0.1),
        'OverSampler': RandomOverSampler(sampling_strategy=0.1),
    }
    return samplers


def getAllRecordsFromDatabase(databaseName):
    start = time.time()
    connection = getConnection()
    cursor = connection.cursor()
    sql_use_Query = "USE " + databaseName
    cursor.execute(sql_use_Query)
    # sql_select_Query = "select * from transaction order by timestamp"
    sql_select_Query = "select * from transaction"
    cursor.execute(sql_select_Query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    numpy_array = np.array(result)
    end = time.time()
    elapsedTime = end - start
    print(f'{databaseName} beolvasva, betöltési idő: {elapsedTime}, rekordszám: {numpy_array.shape}')
    return numpy_array[:, :]


# def getDatabaseNames():
#     databaseNames = ["card_100000_1_1_i_a_parallel"]
#     return databaseNames


def calculateF(beta, precision, recall):
    temp = beta * beta * precision + recall
    if temp != 0:
        f_beta = (1 + beta) * (1 + beta) * precision * recall / temp
    else:
        f_beta = 0
    return f_beta


def saveMetrics(begin_time, statisticalDatabaseName, trainDatabaseName, testDatabaseName, samplerName, scalerName,
                featureSelectorName, modelName, processTimeOfSample, processTimeOfScale, processTimeOfFeatureSelection,
                processTimeOfModelFit, selectedFeatureNumber, predictedLabels,
                testLabels):
    try:
        connection = getConnection()
        cursor = connection.cursor()
        sqlUSEQuery = "USE " + statisticalDatabaseName
        cursor.execute(sqlUSEQuery)

        confusionMatrix = confusion_matrix(testLabels, predictedLabels)
        print(f"Confusion Matrix: {confusionMatrix}")
        TN = int(confusionMatrix[0][0])
        FP = int(confusionMatrix[0][1])
        FN = int(confusionMatrix[1][0])
        TP = int(confusionMatrix[1][1])
        temp = TP + FN
        sensitivity = 0
        if temp != 0:
            sensitivity = TP / (TP + FN)
        temp = TN + FP
        specificity = 0
        if temp != 0:
            specificity = TN / (TN + FP)
        accuracy = float(accuracy_score(testLabels, predictedLabels))
        balanced_accuracy = float(balanced_accuracy_score(testLabels, predictedLabels))
        precision = 0
        temp = TP + FP
        if temp != 0:
            precision = TP / (TP + FP)
        recall = float(recall_score(testLabels, predictedLabels))
        temp = TP + FN
        PPV = 0
        if temp != 0:
            PPV = TP / (TP + FN)
        temp = TN + FN
        NPV = 0
        if temp != 0:
            NPV = TN / (TN + FN)
        temp = FN + TP
        FNR = 0
        if temp != 0:
            FNR = FN / (FN + TP)
        temp = FP + TN
        FPR = 0
        if temp != 0:
            FPR = FP / (FP + TN)
        FDR = 0
        temp = FP + TP
        if temp != 0:
            FDR = FP / (FP + TP)
        temp = FN + TN
        FOR = 0
        if temp != 0:
            FOR = FN / (FN + TN)
        f1 = float(f1_score(testLabels, predictedLabels))
        f_05 = calculateF(0.5, precision, recall)
        f2 = calculateF(2, precision, recall)
        temp = math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN)
        MCC = 0
        if temp != 0:
            MCC = (TP * TN - FP * FN) / temp
        ROCAUC = float(roc_auc_score(testLabels, predictedLabels))
        Youdens_statistic = sensitivity + specificity - 1

        sql_insert_Query = "INSERt INTO metrics (train_database_name,test_database_name,sampler_name, scaler_name,feature_selector_name,model_name,samppling_time, scaling_time,feature_selection_time, model_fit_time, " \
                           "selected_feature_number,TP,FP,TN,FN,sensitivity,specificity,accuracy,balanced_accuracy,prec,recall,PPV,NPV,FNR,FPR,FDR,F_OR,f1,f_05,f2,MCC,ROCAUC,Youdens_statistic, begin_time) VALUES" \
                           "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        values = (
            trainDatabaseName, testDatabaseName, samplerName, scalerName, featureSelectorName, modelName,
            processTimeOfSample,
            processTimeOfScale, processTimeOfFeatureSelection, processTimeOfModelFit, selectedFeatureNumber,
            TP, FP, TN, FN, sensitivity, specificity, accuracy, balanced_accuracy, precision, recall, PPV, NPV, FNR,
            FPR, FDR, FOR, f1, f_05, f2, MCC, ROCAUC, Youdens_statistic, begin_time)
        cursor.execute(sql_insert_Query, values)
        connection.commit()
    except Exception as e:
        print(type(e, e))
        print(
            f'Hiba a mutatószámok képzésekor vagy adatbázisba íráskor, adatbásis neve {testDatabaseName}; sampler: {samplerName}; skálázó: {scalerName}; feature selector: {featureSelectorName}, model: {modelName}')
    finally:
        cursor.close()
        connection.close()


def createStatisticalDatabase(statisticalDatabaseName):
    connection = getConnection()
    sqlCreateSchemaScript = "CREATE DATABASE IF NOT EXISTS " + statisticalDatabaseName
    cursor = connection.cursor()
    cursor.execute(sqlCreateSchemaScript)
    connection.commit()
    sqlUseScript = "USE " + statisticalDatabaseName
    cursor.execute(sqlUseScript)
    file = open("SQL create table metrics.txt", "r")
    sqlCreataTableScript = file.read()
    cursor.execute(sqlCreataTableScript)
    connection.commit()
    cursor.close()
    connection.close()


def getSvdModels(length):
    models = dict()
    for i in range(1, length):
        models[i] = TruncatedSVD(n_components=i)
    return models


def getOptimizedSVDFeatureNumber(features, binaries):
    models = getSvdModels(features.shape[1])
    maxIndex = 0
    for index in models.keys():
        svd = models.get(index)
        svd.fit(features, binaries)
        variancies = svd.explained_variance_ratio_
        # print(f'index: {index}; variancies: {variancies}')
        if svd.explained_variance_ratio_.sum() > 0.995:
            maxIndex = index
            break
    print(f'SVD feature number: {maxIndex}')
    return maxIndex


if __name__ == '__main__':
    print(f'Platform: {platform}')
    cpuCount = mp.cpu_count()
    print(f'Cpu count: {cpuCount}')
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    min = now.strftime("%M")
    sec = now.strftime("%S")
    statisticalDatabaseName = "statistic_" + year + "_" + month + "_" + day + "_cross_validated"
    # statisticalDatabaseName = "statistic_" + year + "_" + month + "_" + day + "_" + hour + "_" + min + "_" + sec + "_ac922"
    createStatisticalDatabase(statisticalDatabaseName)
    trainDatabaseName = "card_100000_1_1_i_a_parallel"
    testDatabaseNames = ["card_100000_1_2_i_a_parallel", "card_100000_1_3_i_a_parallel"]
    availableScalers = getScalers()
    availableRandomSamplers = getRandomSamplers()
    availableModels = getModels()
    featureSelectorNames = ['RFE', 'PCA', 'SVD', 'noSelector']
    # featureSelectorNames = ['PCA']
    trainDataSet = getAllRecordsFromDatabase(trainDatabaseName)
    arraySize = trainDataSet.shape[0]
    transactionTypes = trainDataSet[:, 2:3]
    amounts = trainDataSet[:, 4:5]
    vendorCodes = trainDataSet[:, 8:9]
    aggregates = trainDataSet[:, 49:119]
    s = np.column_stack((transactionTypes, amounts, vendorCodes, aggregates))
    trainFeatures = np.column_stack((transactionTypes, amounts, vendorCodes, aggregates))
    trainLabels = trainDataSet[:, 119:]
    for testDatabaseName in testDatabaseNames:
        testDataSet = getAllRecordsFromDatabase(testDatabaseName)
        testTransactionTypes = testDataSet[:arraySize, 2:3]
        testAmounts = testDataSet[:arraySize, 4:5]
        testVendorCodes = testDataSet[:arraySize, 8:9]
        testAggregates = testDataSet[:arraySize, 49:119]
        testFeatures = np.column_stack((testTransactionTypes, testAmounts, testVendorCodes, testAggregates))
        testLabels = testDataSet[:arraySize, 119:]
        for samplerName in availableRandomSamplers.keys():
            if samplerName != 'NoSampler':
                currentSampler = availableRandomSamplers.get(samplerName)
                print(f'Sampler: {currentSampler}')
                startOfSample = time.time()
                sampledFeatures, sampledLabels = currentSampler.fit_resample(trainFeatures, trainLabels)
                endOfSample = time.time()
                processTimeOfSample = endOfSample - startOfSample
                print(f'Sample process time: {processTimeOfSample}')
            else:
                sampledFeatures = trainFeatures
                sampledLabels = trainLabels
                processTimeOfSample = 0

            for scalerName in availableScalers.keys():
                scaledDatabaseName = testDatabaseName + scalerName
                if scalerName != 'NoScaler':
                    currentScaler = availableScalers.get(scalerName)
                    print(f'Scaler: {currentScaler}')
                    startOfScale = time.time()
                    scaledTrainFeatures = currentScaler.fit_transform(sampledFeatures)
                    scaledTestFeatures = currentScaler.fit_transform(testFeatures)
                    endOfScale = time.time()
                    processTimeOfScale = endOfScale - startOfScale
                    print(f'Scale process time: {processTimeOfScale}')
                else:
                    scaledTrainFeatures = sampledFeatures
                    scaledTestFeatures = testFeatures
                    processTimeOfScale = 0

                selectedFeatureNumber = 0
                featureSelector = None
                processTimeOfFeatureSelection = 0
                for featureSelectorName in featureSelectorNames:
                    if featureSelectorName == 'RFE':
                        start = time.time()
                        featureSelector = RFECV(estimator=RandomForestClassifier(n_jobs=-1), n_jobs=-1)
                        # featureSelector = RFECV(XGBClassifier(tree_method='gpu_hist', gpu_id=0), n_jobs=-1)
                        featureSelector.fit(scaledTrainFeatures, sampledLabels)
                        selectedFeatureNumber = featureSelector.n_features_
                        selectedFeatureNumber = int(selectedFeatureNumber)
                        reducedTrainFeatures = featureSelector.transform(scaledTrainFeatures)
                        reducedTestFeatures = featureSelector.transform(scaledTestFeatures)
                        end = time.time()
                        processTimeOfFeatureSelection = end - start
                    elif featureSelectorName == 'PCA':
                        start = time.time()
                        featureSelector = PCA(n_components=0.95, svd_solver='full')
                        selectedFeatureNumber = 0
                        featureSelector.fit(scaledTrainFeatures)
                        reducedTrainFeatures = featureSelector.transform(scaledTrainFeatures)
                        reducedTestFeatures = featureSelector.transform(scaledTestFeatures)
                        end = time.time()
                        processTimeOfFeatureSelection = end - start
                    elif featureSelectorName == 'SVD':
                        start = time.time()
                        svdFeatureNumber = getOptimizedSVDFeatureNumber(scaledTrainFeatures, trainLabels)
                        featureSelector = TruncatedSVD(n_components=svdFeatureNumber)
                        selectedFeatureNumber = featureSelector.n_components
                        featureSelector.fit(scaledTrainFeatures)
                        reducedTrainFeatures = featureSelector.transform(scaledTrainFeatures)
                        reducedTestFeatures = featureSelector.transform(scaledTestFeatures)
                        end = time.time()
                        processTimeOfFeatureSelection = end - start
                    else:
                        reducedTrainFeatures = scaledTrainFeatures
                        reducedTestFeatures = scaledTestFeatures
                        processTimeOfFeatureSelection = 0
                        selectedFeatureNumber = scaledTrainFeatures.shape[1]
                    print(f'Feature selection process time: {processTimeOfFeatureSelection}')

                    for modelName in availableModels.keys():
                        currentModel = availableModels.get(modelName)
                        print(f'model: {currentModel}')
                        startOfModelFit = time.time()
                        modifiedTrainLabels = column_or_1d(sampledLabels)
                        currentModel.fit(reducedTrainFeatures, modifiedTrainLabels)
                        endOfModelFit = time.time()
                        processTimeOfModelFit = endOfModelFit - startOfModelFit
                        print(f'Model fit process time: {processTimeOfModelFit}')

                        predictedLabels = currentModel.predict(reducedTestFeatures)
                        saveMetrics(now, statisticalDatabaseName, trainDatabaseName, testDatabaseName, samplerName,
                                    scalerName,
                                    featureSelectorName, modelName, processTimeOfSample, processTimeOfScale,
                                    processTimeOfFeatureSelection,
                                    processTimeOfModelFit, selectedFeatureNumber, predictedLabels, testLabels)

                # for samplerName in availableUnderAndOverSamplers:
                #     currentSampler = availableUnderAndOverSamplers.get(samplerName)
                #     print(f'Sampler: {currentSampler}')
                #     startOfSample = time.time()
                #     sampledFeatures, sampledLabels = currentSampler.fit_resample(trainFeatures, trainLabels)
                #     endOfSample = time.time()
                #     processTimeOfSample = endOfSample - startOfSample
                #     print(f'Sample process time: {processTimeOfSample}')
                #     for modelName in availableRandomModels.keys():
                #         currentModel = availableRandomModels.get(modelName)
                #         print(f'model: {currentModel}')
                #         startOfModelFit = time.time()
                #         modifiedTrainLabels = column_or_1d(trainLabels)
                #         currentModel.fit(trainFeatures, modifiedTrainLabels)
                #         endOfModelFit = time.time()
                #         processTimeOfModelFit = endOfModelFit - startOfModelFit
                #         print(f'Model fit process time: {processTimeOfModelFit}')
                #         predictedLabels = currentModel.predict(testFeatures)
                #         saveMetrics(statisticalDatabaseName, databaseName, scalerName, samplerName, modelName,
                #                     processTimeOfScale,
                #                     processTimeOfPCA, pcaComponentNumber, processTimeOfModelFit, processTimeOfSample,
                #                     predictedLabels,
                #                     testLabels)
