# This code implements a predictive maintenance system using machine learning. 
# It simulates sensor data from industrial equipment (temperature, vibration, and pressure) and uses logistic regression to predict whether 
# the equipment will fail (binary classification: 0 = normal, 1 = failure). 
# I'll break it down step by step, explaining what each part does, why it's needed, and how it works. 
# The code is written in Python and uses libraries like NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn. 

#Here, I used comments to explain each step of the code for better understanding.
#Also, I have taken sample values as I don't have access to real sensor data.
#dataset = 500 
# Number of samples in the simulated dataset

import numpy as np #Numpy for numerical operations , especially with arrays and matrices
import pandas as pd #Pandas for data manipulation and analysis , primarily using DataFrames(Tabular data)
import matplotlib.pyplot as plt #matplotlib is used for creating static, animated, and interactive visualizations in Python
import seaborn as sns #Seaborn is a statistical data visualization library based on matplotlib

from sklearn.model_selection import train_test_split #train_test_split is used to split the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler #StandardScaler standardizes features by removing the mean and scaling to unit variance
from sklearn.linear_model import LogisticRegression #LogisticRegression is a classifier for binary classification tasks
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay#These are used to evaluate the performance of classification


#Step 1: Simulate or Load Data
# For demonstration, we'll create a synthetic dataset
np.random.seed(42) #For reproducibility
n_samples = 500

# Simulate sensor data
temperature = np.random.normal(loc=70, scale=10, size=n_samples) #mean/normal 70C
vibration = np.random.normal(loc=5, scale=1.5, size=n_samples) #mean/normal 5 units (mm/s^2)
pressure = np.random.normal(loc=30, scale=5, size=n_samples) #mean/normal 30 PSI/bar

# Simulate failure probability based on sensor readings
failure_prob = (
    1/(1 + np.exp(-(0.15*(temperature - 70))) 
    + 0.5 /(1 + np.exp(-1.0*(vibration - 6.5))) 
    + 0.3/ (1 + np.exp(-0.5*(pressure - 30))))
)
# Simulate binary failure outcome (with noise for realism)[0 = Normal, 1 = Failure]
failure = (failure_prob + np.random.normal(0, 0.2, size=n_samples)) > 1.2
failure = failure.astype(int) #Convert boolean to integer (0 or 1)


#Step 2: Create DataFrame
#Purpose:  Generates fake data since real sensor data isn't available. It creates 500 samples of temperature (around 70°C), vibration (around 5 mm/s²), and pressure (around 30 PSI).
#Failure simulation:  Uses a logistic (sigmoid) function to calculate failure probability based on how far sensor values deviate from normal. Adds noise for realism, then thresholds at 1.2 to get binary outcomes (0 or 1).
#Why?:  Real-world data might be proprietary or unavailable, so simulation mimics it. The sigmoid makes failure more likely when sensors are abnormal (e.g., high temp increases risk).
data = pd.DataFrame({
    'Temperature': temperature,
    'Vibration': vibration,
    'Pressure': pressure,
    'Failure': failure
})


#Step 3: Exploratory Data Analysis (EDA)[visualization of data]

#Purpose: Organizes the data into a table (DataFrame) with columns for each sensor and the target (Failure).
#Why?: Pandas DataFrames are easy to manipulate and analyze, like a spreadsheet in code.

sns.pairplot(data, hue='Failure') #Pairplot to visualize relationships between features colored by failure status
plt.suptitle('Sensor Data and Failures', y=1.05)
plt.tight_layout() #Prevention of overlap
plt.show()

#Step 4: Data Preprocessing[prepare data for ML model]
X = data[['Temperature', 'Vibration', 'Pressure']] #Features
y = data['Failure'] #Target variable

#Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

#Step 6: Feature Scaling[Normalising Features for better model performance]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Step 7: Train the Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

#Step 8: Make Predictions and Evaluate the Model
y_pred = model.predict(X_test_scaled) #using y_predictions on test set using subset of x-scale

# Print classification report
print("Classification Report:") #for seeing on console
print(classification_report(y_test, y_pred)) #for displaying precision, recall, f1-score

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred) #subsets of y-subsets of y_predictions
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Failure'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show() #for displaying confusion matrix

#Step 9: Visualize Feature Importance (Coefficients)
coeffs = model.coef_[0] #Logistic regression coefficients
features = X.columns #Feature names

plt.bar(features, coeffs)#Bar chart of feature coefficients
plt.title('Feature Importance (Logistic Regression Coefficients)')#Title of the bar chart
plt.ylabel('Coefficient Value')#Y-axis label
plt.grid(True)#Grid for better readability
plt.show()#Display the bar chart

#Step 10: Conclusion [Trying live prediction with new data]

#Simulated example input (can replace these numbers)
new_input=pd.DataFrame ([[85,7.5,32]], columns=['Temperature','Vibration','Pressure']) #safe input values
new_input_scaled = scaler.transform(new_input) #Scaling the new input to give these values
prediction = model.predict(new_input_scaled) #Predicting failure status realtime

#Display the prediction result
print("\n--- Live Prediction Example ---")
print("Input Sensor Readings: Temperature=85, Vibration=7.5, Pressure=32")
if prediction[0] == 1:
    print("WARNING! Likely Machine Failure, Maintenance Required!")     
else:
    print("Normal Operation")
