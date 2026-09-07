import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix, f1_score, precision_score, recall_score,r2_score
from sklearn.model_selection import GridSearchCV    
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Extra Trees Classifier": ExtraTreesClassifier()
            }

            model_report : dict[str, float] = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # model_report = {k: v for k, v in model_report.items() if v is not None}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                model_report[model_name] = accuracy

            best_model_name = max(model_report, key=model_report.get)
            best_model_accuracy = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_accuracy < 0.6:
                raise CustomException("No best model found with accuracy greater than 0.6", sys)
            
            logging.info(f"Best Model Found: {best_model_name} with accuracy: {best_model_accuracy}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 Score of the best model: {r2_square}")  
            return r2_square  

        except Exception as e:
            raise CustomException(e, sys)

