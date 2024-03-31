import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
predicted_value = pd.read_excel(r"C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\python_comparecurve\predicted_value.xlsx", header=None, usecols= [10])
actual_velocity = pd.read_excel(r'C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\20230912\velocity.xlsx', header=None, usecols= [10])

r2 = r2_score(actual_velocity, predicted_value)
print("R-squared:", r2)
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(actual_velocity, predicted_value, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
plt.plot(np.unique(actual_velocity), np.poly1d(np.polyfit(actual_velocity.values.flatten(), predicted_value.values.flatten(), 1))(np.unique(actual_velocity)))
ax.set_ylabel('predicted value', fontsize=14)
ax.set_xlabel('actual value', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18)
ax.text(12.2, 9.25, '100m', fontsize=12, alpha=0.5)

plt.show()