from matplotlib import style
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import statsmodels.api as sm
import numpy as np
import SharedVariables as SV

style.use('fivethirtyeight')


def plot_TimeSeries(df, unitOfMeasure, savePath, Scaled, column):
    column = SV.del_unsupported_os_characters(column)
    #fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    df.plot(ax=ax1, color='k', label=df.name, lw=0.5)
    plt.xlabel('time')
    plt.legend(loc='best')
    if Scaled == True:
        plt.ylabel("Scaled")
        plt.tight_layout()
        SavePath_pdf = "%s/Scaled_%s.pdf" % (savePath, column)
        SavePath_jpg = "%s/Scaled_%s.jpg" % (savePath, column)
    else:
        plt.ylabel(unitOfMeasure)
        plt.tight_layout()
        SavePath_pdf = "%s/Raw_%s.pdf" % (savePath, column)
        SavePath_jpg = "%s/Raw_%s.jpg" % (savePath, column)
    plt.savefig(SavePath_pdf)
    plt.savefig(SavePath_jpg)
    plt.close()


def plot_x_y(x, y, savePath, xlim=False, ylim=False):
    #fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(x.values, y.values, marker='o', linestyle='None', label=y.name + 'vs' + x.name)
    if xlim:
        ax1.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.legend(loc='best')
    plt.tight_layout()
    SavePath_pdf = "%s/CloudPlot_%s.pdf" % (savePath, SV.del_unsupported_os_characters(x.name))
    SavePath_jpg = "%s/CloudPlot_%s.jpg" % (savePath, SV.del_unsupported_os_characters(x.name))
    plt.savefig(SavePath_pdf)
    plt.savefig(SavePath_jpg)
    plt.close()


def plot_crosscorr(x, y, savePath, xlim=False, ylim=False, lags=50, level=0.5):
    #fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.xcorr(x, y, usevlines=True, maxlags=lags, normed=True, lw=0.5)  # marker = 'o', markersize = 5)
    #    ax1.vlines(lags, [0], ax1.line2d[0].get_ydata())
    ax1.grid(True)
    ax1.axhline(0, color='black', lw=2)
    ax1.fill_between(ax1.xaxis.get_data_interval(), 0.0, level, alpha=0.25)
    ax1.set_ylim(0.0, 1.0)
    label = "Crosscorrelation: " + x.name + " vs " + y.name
    ax1.set_title(label)
    plt.xlabel('lags')
    plt.ylabel('coefficient value')
    #plt.tight_layout()
    SavePath_pdf = "%s/Crosscorelation_%s.pdf" % (savePath, SV.del_unsupported_os_characters(y.name))
    SavePath_jpg = "%s/Crosscorelation_%s.jpg" % (savePath, SV.del_unsupported_os_characters(y.name))
    plt.savefig(SavePath_pdf)
    plt.savefig(SavePath_jpg)
    plt.close()


def plot_acf(x, savePath, lags=50):
    #fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    sm.graphics.tsa.plot_acf(x, lags=lags, ax=ax1)
    label = "Autocorrelation: " + x.name
    ax1.set_title(label)
    plt.xlabel('lags')
    plt.ylabel('coefficient value')
    plt.tight_layout()
    SavePath_pdf = "%s/Autocorrelation.pdf" % (savePath)
    SavePath_jpg = "%s/Autocorrelation.jpg" % (savePath)
    plt.savefig(SavePath_pdf)
    plt.savefig(SavePath_jpg)
    plt.close()


def plot_acf_diff(x, savePath, lags=50):
    #fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    sm.graphics.tsa.plot_acf(np.diff(x), lags=lags, ax=ax1)
    label = "Autocorrelation: diff(" + x.name + ")"
    ax1.set_title(label)
    plt.xlabel('lags')
    plt.ylabel('coefficient value')
    plt.tight_layout()
    SavePath_pdf = "%s/Autocorrelation.pdf" % (savePath)
    SavePath_jpg = "%s/Autocorrelation.jpg" % (savePath)
    plt.savefig(SavePath_pdf)
    plt.savefig(SavePath_jpg)
    plt.close()

'''
def plot_predict_measured(prediction, measurement, Score, StartDatePredict, SavePath, nameOfSignal, BlackBox, NameOfSubTest):
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    measurement.plot(ax=ax1, color='k', label='Measurement', lw=0.5)
    labelX = 'Prediction(Score = %.3f)' %Score
    prediction.plot(ax=ax1, color='r', label=labelX, lw=0.5)
    plt.ylabel(nameOfSignal)
    plt.xlabel('time')
    plt.legend(loc='best')
    # plt.show()
    ax1.axvline(StartDatePredict, color='k', linestyle='--')
    plt.tight_layout()
    SavePath = "%s/Prediction_%s_%s" % (SavePath, BlackBox, NameOfSubTest)
    SavePath_pdf = "%s.pdf" % (SavePath)
    SavePath_jpg = "%s.jpg" % (SavePath)
    plt.savefig(SavePath_pdf, transparent=True)
    plt.savefig(SavePath_jpg, transparent=True)
    plt.close()
'''
'''
def plot_predict_exogeneous(prediction, measurement, Xtest, nameOfSignal, ExogInputs, ExogVector, SavePath,
                            EndingFileNames):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False);
    measurement.plot(ax=ax1, color='k', label='Measurement');
    prediction.plot(ax=ax1, color='r', label='Prediction')
    ax1.set_ylabel(nameOfSignal);
    Xtest[ExogInputs[ExogVector[0]]].plot(ax=ax2, color='k', label=Xtest[ExogInputs[ExogVector[0]]].name)
    ax2.set_ylabel(Xtest[ExogInputs[ExogVector[0]]].name);
    plt.legend(loc='best')
    plt.xlabel('time of year')
    plt.tight_layout()
    SavePath = "%s/PredictionAndExog_%s" % (SavePath, EndingFileNames)
    SavePath_pdf = "%s.pdf" % (SavePath)
    SavePath_jpg = "%s.jpg" % (SavePath)
    plt.savefig(SavePath_pdf)
    plt.savefig(SavePath_jpg)
    plt.close()
'''
'''
def plot_Residues(prediction, measurement, Score, SavePath, nameOfSignal, BlackBox, NameOfSubTest):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False);
    measurement.plot(ax=ax1, color='k', label='Measurement', lw=0.5)
    labelX = 'Prediction(Score = %.3f)' %Score
    prediction.plot(ax=ax1, color='r', label=labelX, lw=0.5)
    ax1.set_ylabel(nameOfSignal);
    plt.legend(loc='best')

    Residues = measurement - prediction;
    Residues.plot(ax=ax2, color='k', label='Measurement - Prediction', lw=0.5)
    ax2.set_ylabel('Residuals')
    plt.legend(loc='best')

    plt.xlabel('time')
    plt.tight_layout()
    SavePath = "%s/Residuals_%s_%s" % (SavePath, BlackBox, NameOfSubTest)
    SavePath_pdf = "%s.pdf" % (SavePath)
    SavePath_jpg = "%s.jpg" % (SavePath)
    plt.savefig(SavePath_pdf, transparent=True)
    plt.savefig(SavePath_jpg, transparent=True)
    plt.close()
'''

def plot_predict_measured(prediction, measurement, MAE, R2, StartDatePredict, SavePath, nameOfSignal, BlackBox, NameOfSubTest):
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'small',
              #'figure.figsize': (15, 5),
              'axes.labelsize': 'small',
              'axes.titlesize': 'small',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small'}
    pylab.rcParams.update(params)
    #fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    measurement.plot(ax=ax1, color='k', label='Measurement', lw=0.5)
    labelX = 'Prediction(MAE = %.3f; Score = %.2f)' %(MAE, R2)
    prediction.plot(ax=ax1, color='r', label=labelX, lw=0.5)
    plt.ylabel("%s" %(nameOfSignal))
    plt.xlabel('Time')
    plt.legend(fontsize="small" ,loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.23), fancybox=True, framealpha=0.5, labelspacing=0.1)
    # plt.show()
    #ax1.axvline(StartDatePredict, color='k', linestyle='--')
    plt.tight_layout(rect=[-0.02, -0.03, 1.02, 0.90])
    SavePath = "%s/Prediction_%s" % (SavePath, BlackBox)
    SavePath_pdf = "%s.pdf" % (SavePath)
    SavePath_jpg = "%s.jpg" % (SavePath)
    plt.savefig(SavePath_pdf, transparent=True)
    plt.savefig(SavePath_jpg, transparent=True)
    plt.close()


def plot_Residues(prediction, measurement, MAE, R2, SavePath, nameOfSignal, BlackBox, NameOfSubTest):
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'small',
              #'figure.figsize': (15, 5),
              'axes.labelsize': 'small',
              'axes.titlesize': 'small',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small'}
    pylab.rcParams.update(params)


    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    measurement.plot(ax=ax1, color='k', label='Measurement', lw=0.5)
    labelX = 'Prediction(MAE = %.3f; Score = %.2f)' % (MAE, R2)
    prediction.plot(ax=ax1, color='r', label=labelX, lw=0.5)
    ax1.set_ylabel("%s" %(nameOfSignal))
    #ax1.legend(loc="best", framealpha=0.5)
    #ax1.legend(loc=3, ncol=2, framealpha=0.5)
    ax1.legend(fontsize="small" ,loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.23), fancybox=True, framealpha=0.5, labelspacing=0.1)

    Residues = measurement - prediction
    Residues.plot(ax=ax2, color='k', label='Measurement - Prediction', lw=0.5)
    ax2.set_ylabel('Residuals')
    #ax2.legend(loc='best', framealpha=0.5, fontsize="small")
    ax2.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.25), fancybox=True, framealpha=0.5, labelspacing=0.1)

    plt.xlabel('Time')
    #plt.tight_layout()
    plt.tight_layout(rect=[-0.02, -0.03, 1.02, 0.97])
    SavePath = "%s/Residuals_%s" % (SavePath, BlackBox)
    SavePath_pdf = "%s.pdf" % (SavePath)
    SavePath_jpg = "%s.jpg" % (SavePath)
    plt.savefig(SavePath_pdf, transparent=True)
    plt.savefig(SavePath_jpg, transparent=True)
    plt.close()


#Style used for the master thesis
def PAPER_plot_Residues(prediction, measurement, R2, SavePath, nameOfSignal, unit, xUnit, BlackBox, NameOfSubTest):
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'small',
              #'figure.figsize': (15, 5),
              'axes.labelsize': 'small',
              'axes.titlesize': 'small',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small'}
    pylab.rcParams.update(params)


    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    measurement.plot(ax=ax1, color='k', label='Measurement', lw=0.5)
    labelX = 'Prediction(MAE = %.2f)' %R2
    prediction.plot(ax=ax1, color='r', label=labelX, lw=0.5)
    ax1.set_ylabel("%s [%s]" %(nameOfSignal, unit))
    #ax1.legend(loc="best", framealpha=0.5)
    #ax1.legend(loc=3, ncol=2, framealpha=0.5)
    ax1.legend(fontsize="small" ,loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.23), fancybox=True, framealpha=0.5, labelspacing=0.1)

    Residues = measurement - prediction
    Residues.plot(ax=ax2, color='k', label='Measurement - Prediction', lw=0.5)
    ax2.set_ylabel('Residuals [%s]' %unit)
    ax2.legend(loc='best', framealpha=0.5, fontsize="small")
    #ax2.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.25), fancybox=True, framealpha=0.5, labelspacing=0.1)

    plt.xlabel('Time [%s]' %xUnit)
    #plt.tight_layout()
    plt.tight_layout(rect=[-0.02, -0.03, 1.02, 0.97])
    SavePath = "%s/Residuals_PAPER_%s" % (SavePath, BlackBox)
    SavePath_pdf = "%s.pdf" % (SavePath)
    SavePath_jpg = "%s.jpg" % (SavePath)
    plt.savefig(SavePath_pdf, transparent=True)
    plt.savefig(SavePath_jpg, transparent=True)
    plt.close()