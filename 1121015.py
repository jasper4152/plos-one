import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import joblib
from sklearn.tree import plot_tree
import seaborn as sns

#def function
def squared_error_2006(params, Vmax, T, tmax, a):
    t2, t1 = params
    predicted = Vmax*(np.exp((-T+ tmax)/t2) - np.exp(-T / t1))
    return np.sum((a - predicted) ** 2)
    
def curve2006(T, Vmax, tmax, t2, t1):
    return Vmax * (np.exp((-T + tmax) / t2) - np.exp(-T / t1))

def my_eq(a, b, c, x):
    return a*np.power(x, 2)*np.exp(-b*x)-(c*x)

#Part curve2006
#Read data from Excel file into a pandas DataFrame
MaximumVelocity = pd.read_excel(r'C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\20230912\MaximumVelocity.xlsx', header=None)
Max_time = pd.read_excel(r'C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\20230912\max_time.xlsx', header=None)
averageVelocity = pd.read_excel(r'C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\20230912\averageVelocity.xlsx', header=None)
actual_velocity = pd.read_excel(r'C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\20230912\velocity.xlsx', header=None)
time = pd.read_excel(r'C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\20230912\time.xlsx', header=None)

t1 = []
t2 = []
i = 0
while i < 580:
    constraints = [{'type': 'ineq', 'fun': lambda x: 500-x[0]},  # Vmax >= 0
                {'type': 'ineq', 'fun': lambda x: x[1]-0},
                {'type': 'ineq', 'fun': lambda x: 1.5 - x[1]}  # k <= 1
                ]  # f >= 0
    t = np.array(time.loc[i])
    a = np.array(actual_velocity.loc[i])
    # Choose an initial guess for l
    t2_guess = 80
    t1_guess = 0.8
    initial_param = [t2_guess, t1_guess]
    estimated_t2_values = []  # List to store estimated k values for the current set of data points
    estimated_t1_values = []
    MV = np.full(11, MaximumVelocity.loc[i])
    MT = np.full(11, Max_time.loc[i])
    # Use the minimize function to estimate k
    results = minimize(squared_error_2006, initial_param, args=(MV, t[0:11], MT, a[0:11]), constraints= constraints)

    # Retrieve the estimated k
    estimated_t1 = results.x[1]
    estimated_t2 = results.x[0]
    estimated_t2_values.append(estimated_t2)
    estimated_t1_values.append(estimated_t1)

    # Append the list of estimated k values to the main list 'k'
    t1.append(estimated_t1_values)
    t2.append(estimated_t2_values)
    i += 1
print('t2', (np.mean(t2)))
print('t1', (np.mean(t1)))

# Machine learning
# Initialize lists to store true and predicted values
true_values2 = []
predicted_values2 = []
true_values3 = []
predicted_values3 = []

# Assuming X and y are dataframes or series
X = pd.concat([MaximumVelocity, (time.loc[:,10])], axis=1)  # Assuming they are already preprocessed
y = actual_velocity

loo = LeaveOneOut()

# Create a DecisionTreeRegressor
RF_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
Ann_model = MLPRegressor(hidden_layer_sizes=(50), activation='relu', solver='lbfgs', max_iter=5000, random_state=42)

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    RF_model.fit(X_train, y_train)

    # Make a prediction on the test set
    RF_y_pred = RF_model.predict(X_test)
    
    # Append true and predicted values for this iteration
    true_values2.append(y_test.values[0])
    predicted_values2.append(RF_y_pred[0])  # Since y_pred is a single-item array, we take the first element

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    Ann_model.fit(X_train, y_train)

    # Make a prediction on the test set
    Ann_y_pred = Ann_model.predict(X_test)
    
    # Append true and predicted values for this iteration
    true_values3.append(y_test.values[0])
    predicted_values3.append(Ann_y_pred[0])  # Since y_pred is a single-item array, we take the first element


# Convert the lists of true and predicted values to NumPy arrays
true_values2 = np.array(true_values2)
predicted_values2 = np.array(predicted_values2)
true_values3 = np.array(true_values3)
predicted_values3 = np.array(predicted_values3)
print('TF3 = ',true_values3)
print('PF3 = ',predicted_values3)

# Calculate the Mean Squared Error (MSE)
mse2 = mean_squared_error(true_values2, predicted_values2)
mse3 = mean_squared_error(true_values3, predicted_values3)
# Calculate the R-squared (R^2) score
r22 = r2_score(true_values2, predicted_values2)
r23 = r2_score(true_values3, predicted_values3)
RF_y_preds = pd.DataFrame(predicted_values2)
Ann_y_preds = pd.DataFrame(predicted_values3)
# Print the metrics

mse_expo = pd.DataFrame()
r2_expo = pd.DataFrame()
q = 0
y_expo = []

while q < 580:
    current_t2 = t2[q][0]
    current_t1 = t1[q][0]
    zero = curve2006(0, MaximumVelocity.loc[q], Max_time.loc[q], current_t2, current_t1)
    time_expo = time.loc[q]
    predicted_values_expo = []
    for t in time_expo:
        y_2006 = curve2006(t, MaximumVelocity.loc[q], Max_time.loc[q], current_t2, current_t1) - zero
        predicted_values_expo.append(y_2006)
    
    y_expo.append(predicted_values_expo)
    q += 1

# Convert y_expo to a 2D NumPy array
y_expo = np.array(y_expo).reshape(len(y_expo), -1)

mse2 = []
mse3 = []
mse_expo = []
for k in  range(0, 580):
    mse2.append(mean_squared_error(true_values2[k], predicted_values2[k]))
    mse3.append(mean_squared_error(true_values3[k], predicted_values3[k]))
    mse_expo.append(mean_squared_error(true_values3[k], y_expo[k]))
mse2_std = np.std(mse2)
mse3_std = np.std(mse3)
mse_expo_std = np.std(mse_expo)
mse2_mean = np.mean(mse2)
mse3_mean = np.mean(mse3)
mse_expo_mean = np.mean(mse_expo)
print("RF Mean Squared Error (MSE):", mse2_mean)
print("Ann Mean Squared Error (MSE):", mse3_mean)
print("expo Mean Squared Error (MSE):", mse_expo_mean)
algorithms = ['RF model', 'NN model', 'Exponential method']
x_pos = np.arange(len(algorithms))
CTEs = [mse2_mean, mse3_mean, mse_expo_mean]
error = [mse2_std, mse3_std, mse_expo_std]
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('MSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms)
ax.set_title('MSE of three method')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()

feature_names = ('Vmax','Tfinal')
plt.figure(figsize=(10, 6))
plt.title(f"Tree {41}")
plot_tree(RF_model.estimators_[40], filled=True, feature_names=feature_names)  # Replace X with your feature matrix
plt.show()

output = pd.DataFrame()
predicted_values = pd.DataFrame()
#plot graph
i = 0
while i < 580:
    number_list = []
    y_2006 = pd.DataFrame()
    x = np.linspace(0, time.loc[i,10], 400)
    x2 = time.loc[i]
    current_t2 = t2[i][0]
    current_t1 = t1[i][0]
    zero = curve2006(0, MaximumVelocity.loc[i], Max_time.loc[i], current_t2, current_t1)
    for time_full in x:
        y_2006 = y_2006.append(pd.Series(curve2006(time_full, MaximumVelocity.loc[i], Max_time.loc[i], current_t2, current_t1)-zero), ignore_index=True)
    plt.figure()
    model2 = interp1d(x2, predicted_values2[i], kind = "quadratic")
    model3 = interp1d(x2, predicted_values3[i], kind = "quadratic")
    for u in range(11):
        number = model3(x2[u])
        number_list.append(number)
    predicted_values.insert(0,column = '',value=number_list, allow_duplicates= True)
    print(predicted_values)
    X_ = np.linspace(0, x2.max(), 500)
    Y_2=model2(X_)
    Y_3=model3(X_) 
    plt.axis([0,13,0,13])
    plt.plot(X_, Y_3, label='NN', color = 'green')
    plt.scatter(time.loc[i], actual_velocity.loc[i], label = 'actual data', color = 'blue')
    string = 'MSE = %f' %(mse3[i])
    plt.text(10, 12, string, fontsize = 8,  
            bbox = dict(facecolor = 'blue', alpha = 0.5))
        # Initialize the range boundaries
    peak = max(Y_3)  # Find the peak value in Y_2
    lower_bound = peak * 0.98  # Calculate the lower bound of the range

        # Initialize a list to store the numbers in the range
    numbers_in_range = []

        # Loop through the Y_2 values
    for value in Y_3:
        if lower_bound <= value <= peak:
            numbers_in_range.append(value)

        # Find the indices of Y_2 values that are in the range (peak, peak - 0.5)
    indices_in_range = [z for z, value in enumerate(Y_3) if lower_bound <= value <= peak]

        # Get the corresponding X_ values for the indices
    x_values_in_range = [X_[z] for z in indices_in_range]

    # Print the corresponding x values
    plt.axvspan(x_values_in_range[0], x_values_in_range[-1], color='red', alpha=0.5)
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.legend(loc = 'lower right')
    plt.title(f"Velocity curve of data - Graph {i+1}")
    save_dir = r'C:\Users\ＫＩＴ\OneDrive\文件\運動大數據\python_comparecurve'
    # Save the figure to a file without displaying it
    filename = f'{save_dir}\\V(t)figure_{i}.png'
    plt.savefig(filename)
    
    # Close the current figure to free up memory
    plt.close()

    # Find the index of the peak value in Y_3 using numpy.argmax
    peak_index = np.argmax(Y_3)

    # Save the peak, corresponding x-coordinate, and range boundaries to a DataFrame
    data = {
        'Peak': [peak],
        'Peak_X_Coordinate': [X_[peak_index]],  # Use peak_index to get the corresponding x-coordinate
        'X_Value_in_Range_Start': [x_values_in_range[0]],
        'X_Value_in_Range_End': [x_values_in_range[-1]]
        }

    output = output.append(data, ignore_index=True)

    # Save the DataFrame to an Excel file
    excel_filename = f'{save_dir}\\peak_data.xlsx'
    output.to_excel(excel_filename, index=False)

    i = i+1
excel_filename_2 = f'{save_dir}\\predicted_value.xlsx'
predicted_values.to_excel(excel_filename_2,                                                                                                              index=False)
# Save the model to a file
joblib.dump(Ann_model, 'NN_model1113.pkl')

# mean plot
mean_velocity = actual_velocity.mean()
mean_time = time.mean()
predicted_values3 = pd.DataFrame(predicted_values3)
mean_prediction = predicted_values3.mean()
model_mean = interp1d(mean_time, predicted_values3.mean(axis=0), kind="quadratic")

x = np.linspace(0, mean_time[10], 580)
y = model_mean(x)
fig, ax = plt.subplots()

# Calculate the confidence interval for each point on the curve
for i in range(0, 11):
    ci = 1.96 * np.std(predicted_values3) / np.sqrt(580)
    print(ci)
    # Plot the mean curve
    ax.plot(x, y, label='Mean Curve', color='green')
    
    # Plot the confidence interval for each point on the curve
    ax.fill_between(mean_time[i], ((mean_prediction[i]) - (ci[i])), ((mean_prediction[i]) + (ci[i])), label='95% CI', color='b', alpha=0.3)

    # Scatter plot of actual mean data
    plt.scatter(mean_time, mean_velocity, label='Actual Mean Data', color='red')
    plt.title("Velocity curve of mean data and 95%CI" )
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.legend(loc='lower right')
    plt.show()