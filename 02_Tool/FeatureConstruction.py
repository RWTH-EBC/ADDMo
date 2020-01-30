import pandas as pd

from openpyxl import load_workbook
from Functions.PlotFcn import *
import sys
from sklearn.model_selection import train_test_split

#from GlobalVariables import *
import SharedVariables as SV
import os

#Todo: get the resolution of the data


#Cross, Cloud and Autocorrelation plots
def cross_auto_cloud_correlation_plotting(LagsToBePlotted, Data):
    # set folder for sensor and resolution
    CorrelationResultsFolder = "%s/%s" % (SV.ResultsFolder, "cross_auto_cloud_plotting")
    # check if directory exists, if not, create it
    if not os.path.exists(CorrelationResultsFolder):
        os.makedirs(CorrelationResultsFolder)

    # plot simple dependency on influence factors
    Labels = [column for column in Data.columns]
    for i in range(1, len(Labels)):
        plot_x_y(Data[Labels[i]], Data[SV.NameOfSignal], CorrelationResultsFolder)

    # plot autocorrelation
    plot_acf(Data[SV.NameOfSignal], CorrelationResultsFolder, lags=LagsToBePlotted)

    # make an array for correlation coefficients
    corrCoeffMax = np.zeros(len(Labels))
    DictCoerrCoeff = dict()

    # plot cross correlation between signal to be forecasted and exogenous input
    for i in range(1, len(Labels)):
        plot_crosscorr(Data[SV.NameOfSignal], Data[Labels[i]], CorrelationResultsFolder, lags=LagsToBePlotted)
        corrCoeff = plt.xcorr(Data[SV.NameOfSignal], Data[Labels[i]], maxlags=LagsToBePlotted, normed=True)
        # max value over the values of the correlation coefficient
        DictCoerrCoeff[Data[Labels[i]].name] = np.amax(corrCoeff[1])
        corrCoeffDf = pd.DataFrame(DictCoerrCoeff, index=[Data[SV.NameOfSignal].name], columns=Labels)

    # save cross correlations in an excel file
    ExcelFile = "%s/CrossCorrelationCoefficients.xlsx" % (CorrelationResultsFolder)
    writer = pd.ExcelWriter(ExcelFile)
    corrCoeffDf.to_excel(writer, sheet_name=SV.NameOfExperiment, float_format='%.2f')
    writer.save()

#Manual creation of lags of the Features(Signal excluded)
def manual_featurelag_create(FeatureLag, Data, Data_AllSamples):
    if len(FeatureLag) != len(list(Data)):
        sys.exit("Your FeatureLag Array has to have as many Arrays as Columns of your input data(Index excluded)")
    for i in range(0, len(FeatureLag)):
        NameOfFeature = SV.nameofmeter(Data, i) #get the name of the meter in column i
        if NameOfFeature != SV.NameOfSignal: #making sure this method does not produce OwnLags
            Xauto = Data_AllSamples[NameOfFeature]  # copy of the Feature to use for shifting
            for lag in range(0, len(FeatureLag[i])):
                FeatureLagName = NameOfFeature + "_lag" + str(FeatureLag[i][lag]) #create a column name per lag
                DataShift = Xauto.shift(periods=FeatureLag[i][lag]).fillna(method='bfill')  # shift with the respective values with the respective lag
                Data = pd.concat([Data, DataShift.rename(FeatureLagName)], axis=1, join="inner")  # joining the dataframes just for the selected period
    return (Data)

#Manual creation of lags of the Signal(OwnLags)
def man_ownlag_create(OwnLag, Data, Data_AllSamples):
    # is there a relevant autocorrelation for the signal to be predicted?
    Xauto = Data_AllSamples[SV.NameOfSignal]  # copy of Y to use for shifting
    for lag in range(0,len(OwnLag)):
        OwnLagName = SV.NameOfSignal + "_lag_" + str(OwnLag[lag]) #create a column name per lag
        DataShift = Xauto.shift(periods=OwnLag[lag]).fillna(method='bfill')  # shift with the lag to be considered
        Data = pd.concat([Data, DataShift.rename(OwnLagName)], axis=1, join="inner")  # joining the dataframes just for the selected period
    return (Data)

#Automatic creation of difference data through building the delta value of t - (t-1)
def difference_create(FeaturesDifference, Data):
    if FeaturesDifference == True: #for all features a derivative series should be constructed
        (X, Y) = SV.split_signal_and_features(Data)
    else: #if certain features are selected
        (X, Y) = SV.split_signal_and_features(Data)
        X = X[X.columns[FeaturesDifference]] # select only those columns of which a difference shall be created
    for column in X:  # loop through all columns
        DifferenceName = "Delta_" + column #construct a new name for the difference data
        Difference = X[column].diff() #build difference of respective feature
        Difference = Difference.fillna(0) #fill up first row with 0 (since the first row has no value before and hence should be 0)
        Difference = SV.post_scaler(Difference, SV.StandardScaling, SV.RobustScaling)
        Data = pd.concat([Data, Difference.rename(columns={0:DifferenceName})], axis=1, join="inner")  # joining the dataframes just for the selected period
    return Data

#automatic timeseries ownlag construction used in Feature Selection (Disadvantage: Only finds local optimum of the amount of Ownlags)
def automatic_timeseries_ownlag_constructor(Data, Data_AllSamples, MinOwnLag, Estimator, params, MinIncrease):
    print("Auto timeseries-ownlag constructor START")
    Xauto = Data_AllSamples[SV.NameOfSignal]  # copy of Y to use for shifting
    (X, Y) = SV.split_signal_and_features(Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
    Result_dic = Estimator(X_train, y_train, X_test, y_test, *params)  # score will be done over hold out 0.25 percent of data
    Score = Result_dic["score"]  # get the score  #get initial score
    Score_i = Score
    i = MinOwnLag - 1 #just for easier looping
    while True: #loop as long as a new ownlag increases the accuracy
        i += 1
        Score = Score_i #set score equal to the new and better score_i
        OwnLagName = SV.NameOfSignal + "_lag_" + str(i)  # create a column name per lag
        DataShift = Xauto.shift(periods=i).fillna(method='bfill')  # shift with the lag to be considered
        Data = pd.concat([Data, DataShift.rename(OwnLagName)], axis=1, join="inner")  #joining the dataframes just for the selected period
        (X, Y) = SV.split_signal_and_features(Data)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        Result_dic = Estimator(X_train, y_train, X_test, y_test,*params)  # score will be done over hold out 0.25 percent of data
        Score_i = Result_dic["score"] #get the score with the additional ownlag
        if not (Score+MinIncrease) <= Score_i:
            break
    Data = Data.drop([OwnLagName], axis=1) #drop the last ownlag that was not improving the score
    return Data

def auto_featurelag_constructor(Data, Data_AllSamples, MinFeatureLag, MaxFeatureLag, Estimator, params, MinIncrease):
    print("auto featurelag constructor START")
    #wrapper gets default accuracy
    (X, Y) = SV.split_signal_and_features(Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
    Result_dic = Estimator(X_train, y_train, X_test, y_test, *params)  # score will be done over hold out 0.25 percent of data
    Score = Result_dic["score"]  # get the score  #get initial score
    Columns = list(Data)
    Data_c = Data #copy of Data that is never changed
    for i in range(0, len(Columns)):
        NameOfFeature = SV.nameofmeter(Data, i) #get the name of the meter in column i
        if NameOfFeature != SV.NameOfSignal: #making sure this method does not produce OwnLags
            Xauto = Data_AllSamples[NameOfFeature]  # copy of the Feature to use for shifting
            Score_best= (-100) #set initial very bad value
            for lag in range(MinFeatureLag, (MaxFeatureLag+1)):
                Score_1 = Score_best
                FeatureLagName = NameOfFeature + "_lag" + str(lag) #create a column name per lag
                DataShift = Xauto.shift(periods=lag).fillna(method='bfill')  # shift with the respective values with the respective lag
                Data_i = pd.concat([Data_c, DataShift.rename(FeatureLagName)], axis=1, join="inner")  # joining the dataframes just for the selected period
                #wrapper gets accuracy with respective feature lag
                (X_i, Y) = SV.split_signal_and_features(Data_i)
                X_train_i, X_test_i, y_train, y_test = train_test_split(X_i, Y, test_size=0.25)
                Result_dic = Estimator(X_train_i, y_train, X_test_i, y_test,*params)  # score will be done over hold out 0.25 percent of data
                Score_2 = Result_dic["score"]  # get the score with the additional featurelag
                #check whether score is higher than previous or not
                if Score_2 > Score_1:
                    DataShift_best = DataShift
                    FeatureLagName_best = FeatureLagName
                    Score_best = Score_2
            if Score_best > (Score+MinIncrease): #if best featurelag of respective feature is better than initial score: add it
                Data = pd.concat([Data, DataShift_best.rename(FeatureLagName_best)], axis=1, join="inner")#add best lag of respective feature to the dataframe
                print("Added feature lag = %s" %(FeatureLagName_best))
    return (Data)


#Main#
def main():
    print("FeatureConstruction")

    Data = pd.read_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_PeriodSelection" + '.pickle'))
    Data_AllSamples = pd.read_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_Preprocessing" + '.pickle'))


    Datas=[Data] #also for not making e.g. featurelagcreate create lags of differences; Data needs to be in for the case no feature construction is done
    if SV.Cross_auto_cloud_correlation_plotting == True:
        cross_auto_cloud_correlation_plotting(LagsToBePlotted = SV.LagsToBePlotted, Data = Data)
    if SV.DifferenceCreate == True:
        _Data = difference_create(SV.FeaturesDifference, Data)
        Datas.append(_Data.drop(list(Data),axis=1)) #make sure only the added features are appended to Datas
    if SV.ManFeaturelagCreate == True:
        _Data = manual_featurelag_create(SV.FeatureLag, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))
    if SV.AutoFeaturelagCreate == True:
        _Data = auto_featurelag_constructor(Data, Data_AllSamples, SV.MinFeatureLag, SV.MaxFeatureLag, SV.EstimatorWrapper, SV.WrapperParams, SV.MinIncrease)
        Datas.append(_Data.drop(list(Data), axis=1))
    if SV.ManOwnlagCreate == True:
        _Data = man_ownlag_create(SV.OwnLag, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))

    DataF = pd.concat(Datas, axis=1, join="inner")  #joining the datas produced by all feature construction methods

    #Save dataframe to pickle
    DataF.to_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_FeatureConstruction" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file
    ExcelFile = os.path.join(SV.ResultsFolder, "ProcessedInputData_%s.xlsx"%(SV.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    DataF.to_excel(writer, sheet_name="FeatureConstruction")
    writer.save()
    writer.close()