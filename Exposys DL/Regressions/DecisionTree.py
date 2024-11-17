from PackageClass import ModelImports

class DecisionTree:
    @staticmethod
    def get_decision_tree(X_train,y_train):
        model_libs = ModelImports.get_imports();
        tree_model = model_libs['DecisionTreeRegressor'](random_state=42)
        tree_model.fit(X_train, y_train)
        return tree_model