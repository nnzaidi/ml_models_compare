import pandas as pd

data = pd.read_csv('real_estate.csv')

# Display the first few rows
data_head = data.head()
# print(data_head)
# print(data.info())    # The dataset consists of 414 entries and 7 columns, with no missing values

###### Data Pre-Processing ######
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

# Convert 'Transaction date' to datetime and extract year and month
data['Transaction date']  = pd.to_datetime(data['Transaction date'])
data['Transaction year']  = data['Transaction date'].dt.year
data['Transaction month'] = data['Transaction date'].dt.month

# Drop the original 'Transaction date' from the dataset
data = data.drop(columns=['Transaction date'])
# print(data.head())

# Define features and target variable
x = data.drop('House price of unit area', axis=1)   # axis=0 for dropping rows; axis=1 for dropping columns
y = data['House price of unit area']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,      # 0.2 indicates that 20% of the data is for testing
                                                    random_state=42)    # 42 ensures that the split is the same in every execution

# Scale the features
scaler = StandardScaler()   # A pre-processing technique in scikit-learn used for standardizing features by removing the mean and scaling to unit variance
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

x_train_scaled.shape        # (331, 7) -- 80% used for training
x_test_scaled.shape         # (83, 7)  -- 20% used for testing

###### Model Training and Comparison ######
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize the models
models = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(random_state=42),
    'Random Forest'     : RandomForestRegressor(random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(random_state=42)
}

# Dictionary to hold the evaluation metrics for each model
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Training the model
    model.fit(x_train_scaled,y_train)

    # Making predictions on the test set
    predictions = model.predict(x_test_scaled)

    # Calculating evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Storing the metrics
    results[name] = {'MAE'  : mae,
                     'R2'   : r2
                    }

# Convert the results to a DataFrame for better readability  
results_df = pd.DataFrame(results).T    # Transpose method
print(results_df)