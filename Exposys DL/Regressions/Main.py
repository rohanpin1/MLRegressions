from PackageClass import ModelImports 
from LinearRegression import LinearRegression
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from KNN import KNN
from GradientBoosting import GradientBoosting

model_libs = ModelImports.get_imports()

data = model_libs['pandas'].read_csv('./Data Set/50_Startups.csv')
print(data.isnull().sum())
print(data.describe())

data = data[(data['R&D Spend'] > 0) & (data['Marketing Spend'] > 0) & (data['Profit'] > 0)]


# Feature columns
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]

# Target column
y = data['Profit']

#20% of the data will be used for testing, and the remaining 80% will be used for training.
X_train, X_test, y_train, y_test = model_libs['train_test_split'](X, y, test_size=0.2, random_state=42)

# Linear Regression Model
y_pred_linear = LinearRegression.get_linear_regression(X_train, y_train).predict(X_test)

# Decision Tree Model
y_pred_tree = DecisionTree.get_decision_tree(X_train,y_train).predict(X_test)

# Random Forest Model
y_pred_forest = RandomForest.get_random_forest(X_train,y_train).predict(X_test)

# KNN Model
y_pred_knn = KNN.get_knn(X_train,y_train).predict(X_test)

# Gradient Boosting
y_pred_gb = GradientBoosting.get_gradientboosting(X_train,y_train).predict(X_test)

def evaluate_model(y_test, y_pred):
    mae = model_libs['mean_absolute_error'](y_test, y_pred)
    mse = model_libs['mean_squared_error'](y_test, y_pred)
    rmse = model_libs['numpy'].sqrt(mse)
    r2 = model_libs['r2_score'](y_test, y_pred)
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


model_libs['joblib'].dump(LinearRegression.get_linear_regression(X_train, y_train), './Train Model/finalised_model.pkl')

loaded_model = model_libs['joblib'].load('./Train Model/finalised_model.pkl')

predictions = loaded_model.predict(X_test)
print(predictions)
