from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np




"""
df = pd.read_excel("C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Messdaten.xlsx")
Time = df['Time']
bat= df['P_bat_control [W]']
batarray=bat.array
x1=batarray[3]
x2=batarray[4]

print(df.head())
"""
def averageData(data1, data2, data3):

    df1= pd.read_excel(data1)
    df2= pd.read_excel(data2)
    df3= pd.read_excel(data3)
    Time = df1['Time']
    bat1 = df1['P_bat_control [W]'].array
    bat2 = df2['P_bat_control [W]'].array
    bat3 = df3['P_bat_control [W]'].array

    #print(bat1)
    avData=[]
    for i in range(480):
        temp = (bat1[i]+bat2[i]+bat3[i])/3
        avData.append(temp)
    #combine time + avData

    avgseries=pd.Series(avData,index=Time)
    return avgseries
def messDatas(Messdaten):
    messdata = pd.read_excel(Messdaten)
    Time=messdata['Time']
    pbat= messdata['P_bat_control [W]'].array
    messseries=pd.Series(pbat,Time)

    return messseries
def plot_predict(messdata, ann,rf,svr,Nummer):
    import matplotlib.pylab as pylab

    params = {'legend.fontsize': 'small',
              # 'figure.figsize': (15, 5),
              'axes.labelsize': 'small',
              'axes.titlesize': 'small',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small'}
    pylab.rcParams.update(params)
    #Messdata = pd.read_excel(data)
    #series=Messdata.ix[:,0]
    #x= data['Time']
    #y= data['P_bat_control [W]']
    #ax1=plt.subplot2grid((1, 1), (0, 0))
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=False);
    messdata.plot(ax=ax1 ,color ='k', label= 'Messdaten', lw= 1.5)
    ann.plot(ax=ax1 ,color ='r', label= 'ANN', lw= 1)
    rf.plot(ax=ax1, color='g', label='RF', lw=1)
    svr.plot(ax=ax1, color='b', label='SVR',linestyle='dashed', lw=1)
    plt.ylabel("P_bat_control[W] in W")
    plt.xlabel('Time')
    plt.legend(fontsize="small", loc="best", ncol=1, bbox_to_anchor=(0.1, 0.2), fancybox=True, framealpha=0.5,
               labelspacing=0.1)
    #plt.show()
    SavePath = "Plot3short"


    SavePath_pdf = "%s.pdf" % (SavePath)
    SavePath_svg = "%s.svg" % (SavePath)
    plt.savefig(SavePath_svg, format="svg")
    plt.savefig(SavePath_pdf, format="pdf")
    plt.close()


if __name__ == '__main__':
    Messdaten= "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Messdaten_short.xlsx"
    #messdata = pd.read_excel(Messdaten)
    ANN_1= "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_ANNKPI2(0.5).1_short.xlsx"
    ANN_2= "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_ANNKPI2(0.5).2_short.xlsx"
    ANN_3 = "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_ANNKPI2(0.5).3_short.xlsx"

    RF_1 = "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_RFBIC.1_short.xlsx"
    RF_2 = "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_RFBIC.2_short.xlsx"
    RF_3 = "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_RFBIC.3_short.xlsx"

    SVR_1 = "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_NuSVRAIC.1_short.xlsx"
    SVR_2 = "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_NuSVRAIC.2_short.xlsx"
    SVR_3 = "C:/Users/pm/Documents/RWTH/7.Semester/Bachelorarbeit/Python/addmo-automated-ml-regression/04_Results/predict/Predict_NuSVRAIC.3_short.xlsx"
    nummer='1'
    messdata=messDatas(Messdaten)
    Annavg = averageData(ANN_1,ANN_2, ANN_3)
    Rfavg= averageData(RF_1,RF_2,RF_3)
    Svravg= averageData(SVR_1,SVR_2,SVR_3)
    plot_predict(messdata, Annavg,Rfavg,Svravg,nummer)














