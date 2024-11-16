import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib



data = pd.read_csv('./Data Set/50_Startups.csv')
print(data.isnull().sum())
print(data.describe())

data = data[(data['R&D Spend'] > 0) & (data['Marketing Spend'] > 0) & (data['Profit'] > 0)]


# Feature columns
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]

# Target column
y = data['Profit']

#20% of the data will be used for testing, and the remaining 80% will be used for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)


# Decision Tree Model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Predictions
y_pred_tree = tree_model.predict(X_test)


# Random Forest Model
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

# Predictions
y_pred_forest = forest_model.predict(X_test)

# KNN Model
knn_model = KNeighborsRegressor(n_neighbors=7);
knn_model.fit(X_train, y_train)

# Predictions
y_pred_knn = knn_model.predict(X_test)


# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
gb_model.fit(X_train,y_train)

# Predictions
y_pred_gb = gb_model.predict(X_test)

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2


# Linear Regression metrics
mae_linear, mse_linear, rmse_linear, r2_linear = evaluate_model(y_test, y_pred_linear)
print("Linear Regression - MAE:", mae_linear, "MSE:", mse_linear, "RMSE:", rmse_linear, "R2:", r2_linear, "Performance:", r2_linear*100)

# Decision Tree metrics
mae_tree, mse_tree, rmse_tree, r2_tree = evaluate_model(y_test, y_pred_tree)
print("Decision Tree - MAE:", mae_tree, "MSE:", mse_tree, "RMSE:", rmse_tree, "R2:", r2_tree, "Performance:", r2_tree*100)

# Random Forest metrics
mae_forest, mse_forest, rmse_forest, r2_forest = evaluate_model(y_test, y_pred_forest)
print("Random Forest - MAE:", mae_forest, "MSE:", mse_forest, "RMSE:", rmse_forest, "R2:", r2_forest, "Performance:",r2_forest*100)

# KNN metrics
mae_knn, mse_knn, rmse_knn, r2_knn = evaluate_model(y_test, y_pred_knn)
print("KNN - MAE:",mae_knn, "MSE:",mse_knn, "RMSE:",rmse_knn, "R2:",r2_knn, "Performance:",r2_knn*100)

# GB metrics
mae_gb, mse_gb, rmse_gb, r2_gb = evaluate_model(y_test,y_pred_gb)
print("Gradient Boosting - MAE:", mae_gb, "MSE:",mse_gb,"RMSE:",rmse_gb,"R2:", r2_gb, "Performance:", r2_gb*100)


joblib.dump(linear_model, './Train Model/finalised_model.pkl')

loaded_model = joblib.load('./Train Model/finalised_model.pkl')

predictions = loaded_model.predict(X_test)
print(predictions)
