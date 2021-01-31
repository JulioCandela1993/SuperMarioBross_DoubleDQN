# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
#os.chdir("C:/Users/Julio/Documents/MasterDegree/BDMA/Classes/CentraleSupelec/DeepLearning/Project/Test")


## First Experiment (Double DQN vs DQN)

def plotComparissonChart(name_doubleDQN, name_DQN, title):
    path_doubleDQN = name_doubleDQN+ "/DQN.csv"
    path_DQN = name_DQN+ "/DQN.csv"

    doubleDQN_pd = pd.read_csv(path_doubleDQN,header = None)
    DQN_pd = pd.read_csv(path_DQN,header = None)

    pd_data = doubleDQN_pd
    pd_data.columns = [name_doubleDQN]
    pd_data[name_DQN] = DQN_pd

    pd_data = pd_data.reset_index()
    bines = [(i) * 50 for i in list(range(101))]
    bines_lab = [(i) * 50 for i in list(range(100))]
    pd_data['binned'] = pd.cut(pd_data['index'], bins=bines, labels=bines_lab)
    pd_data = pd_data.fillna(0)
    datag = pd_data.groupby(['binned']).mean()
    datag = datag.reset_index()
    dataq_min = pd_data.groupby(['binned']).quantile(.10)
    dataq_min = dataq_min.reset_index()
    dataq_max = pd_data.groupby(['binned']).quantile(.90)
    dataq_max = dataq_max.reset_index()

    fig, ax = plt.subplots()
    ax.plot(datag['binned'],datag[name_doubleDQN])
    ax.fill_between(datag['binned'], dataq_min[name_doubleDQN], dataq_max[name_doubleDQN], alpha=.3)
    
    ax.plot(datag['binned'],datag[name_DQN])
    ax.fill_between(datag['binned'], dataq_min[name_DQN], dataq_max[name_DQN], alpha=.3)
    
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Reward")
    
    ax.legend(('Double DQN', 'DQN') , loc='upper left')
    
    ax.set_title(title + ' - Double DQN vs DQN', fontsize=16)
    
    plt.show()



# Classic Architecture
plotComparissonChart("DoubleDQN_Classic", "DQN_Classic", "Classic")
# Deeper Architecture
plotComparissonChart("DoubleDQN_Deeper_ConvNet", "DQN_Deeper_ConvNet", "Deeper")
# Wider Architecture
plotComparissonChart("Double_DQN_Wider_ConvNet", "DQN_Wider_ConvNet", "Wider")
# MaxPooling Architecture
plotComparissonChart("DoubleDQN_MaxPooling_ConvNet", "DQN_MaxPooling_ConvNet", "MaxPooling")


## Second Experiment (Reward of all of them)

files = { 'DoubleDQN_Classic':'DoubleDQN - Classic',
         "DQN_Classic": 'DQN - Classic',
         'DoubleDQN_Deeper_ConvNet':'DoubleDQN - Deeper',
         "DQN_Deeper_ConvNet": 'DQN - Deeper',
         'Double_DQN_Wider_ConvNet':'DoubleDQN - Wider',
         "DQN_Wider_ConvNet": 'DQN - Wider',
         'DoubleDQN_MaxPooling_ConvNet':'DoubleDQN - MaxPooling',
         "DQN_MaxPooling_ConvNet": 'DQN - MaxPooling'
         
    }

test_results = pd.DataFrame(columns = ["Method", "Mean Reward", "P10_Reward",
                                           "P90_Reward","std","ymin","ymax"])
i=1
for key in files:
    path = key+ "/DQN_finaltest.csv"
    test_rewards = pd.read_csv(path,header = None)
    mean_r = float(test_rewards.mean())
    p10 = float(test_rewards.quantile(.15))
    p90 = float(test_rewards.quantile(.85))
    
    ymin = mean_r - p10
    ymax = p90 - mean_r
    
    std = float(test_rewards.std())
    test_results.loc[i] = [files[key] ,mean_r,p10,p90,std,ymin,ymax]
    i+=1


plt.bar(test_results.index, test_results["Mean Reward"]                
                , align='center', alpha=0.5
                , yerr = np.array(np.transpose(test_results[["ymin","ymax"]]))
                , error_kw=dict(lw=1, capsize=5, capthick=1))
plt.xticks(test_results.index, test_results["Method"], rotation=90)
plt.ylabel('Reward')
plt.title('Rewards after 5000 training episodes', fontsize=16)

plt.show()

test_results['P90_Reward']















