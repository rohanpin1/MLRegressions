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


class ModelImports:
    # Storing the libraries as static methods or attributes
    np = np
    pd = pd
    train_test_split = train_test_split
    LinearRegression = LinearRegression
    DecisionTreeRegressor = DecisionTreeRegressor
    RandomForestRegressor = RandomForestRegressor
    KNeighborsRegressor = KNeighborsRegressor
    GradientBoostingRegressor = GradientBoostingRegressor
    mean_absolute_error = mean_absolute_error
    mean_squared_error = mean_squared_error
    r2_score = r2_score
    joblib = joblib

    @staticmethod
    def get_imports():
        return {
            'numpy': np,
            'pandas': pd,
            'train_test_split': train_test_split,
            'LinearRegression': LinearRegression,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'KNeighborsRegressor': KNeighborsRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'mean_absolute_error': mean_absolute_error,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score,
            'joblib': joblib
        }