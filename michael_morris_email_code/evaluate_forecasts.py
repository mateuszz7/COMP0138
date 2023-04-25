import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
 
files = [pd.read_csv('Results/US_Forecasts_2015-16' + str(gamma) + 'days_ahead.csv', index_col=0, parse_dates=True) for gamma in range(1,29)]

model_unc = np.asarray([f['Model'] for f in files])
plt.subplot(2,2,1)
plt.plot(model_unc.mean(0))
plt.ylabel('model uncertainty')

plt.subplot(2,2,2)
plt.plot(model_unc.mean(1))
plt.ylabel('model uncertainty')



data_unc = np.asarray([f['Data'] for f in files])
plt.subplot(2,2,3)
plt.plot(data_unc.mean(0))
plt.ylabel('data uncertainty')

plt.subplot(2,2,4)
plt.plot(data_unc.mean(1))
plt.ylabel('data uncertainty')

plt.show()



model_unc = np.asarray([f['Pred'] for f in files])
[plt.plot(m) for m in model_unc]
plt.plot(files[0]['True'], color='black')

plt.plot(files[-1]['True']/files[-1]['True'].max())

num = 0
plt.plot(files[num]['Pred']-files[num]['Std'], color='red')
plt.plot(files[num]['Pred']+files[num]['Std'], color='red')
plt.plot(files[num]['True'], color='black')
plt.show()



