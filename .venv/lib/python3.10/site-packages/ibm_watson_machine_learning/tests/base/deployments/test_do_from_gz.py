#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import unittest

import pandas as pd

from ibm_watson_machine_learning.tests.base.abstract.abstract_batch_deployment_test import AbstractBatchDeploymentTest


class TestDODeployment(AbstractBatchDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying
    DecisionOptimization model using compressed file.
    """
    deployment_type = "do-docplex_20.1"
    software_specification_name = "do_20.1"
    model_name = deployment_name = "do_model_from_gz"
    file_name = "do-model.tar.gz"
    IS_MODEL = True

    def get_model(self):
        return os.path.join(os.getcwd(), 'base', 'artifacts', 'decision_optimization', self.file_name)

    def create_model_props(self):
        output_data_schema = [
            {
                'id': 'stest',
                'type': 'list',
                'fields': [
                    {'name': 'age', 'type': 'float'},
                    {'name': 'sex', 'type': 'float'},
                    {'name': 'cp', 'type': 'float'},
                    {'name': 'restbp', 'type': 'float'},
                    {'name': 'chol', 'type': 'float'},
                    {'name': 'fbs', 'type': 'float'},
                    {'name': 'restecg', 'type': 'float'},
                    {'name': 'thalach', 'type': 'float'},
                    {'name': 'exang', 'type': 'float'},
                    {'name': 'oldpeak', 'type': 'float'},
                    {'name': 'slope', 'type': 'float'},
                    {'name': 'ca', 'type': 'float'},
                    {'name': 'thal', 'type': 'float'}]
            },
            {
                'id': 'teste2',
                'type': 'test',
                'fields': [
                    {'name': 'age', 'type': 'float'},
                    {'name': 'sex', 'type': 'float'},
                    {'name': 'cp', 'type': 'float'},
                    {'name': 'restbp', 'type': 'float'},
                    {'name': 'chol', 'type': 'float'},
                    {'name': 'fbs', 'type': 'float'},
                    {'name': 'restecg', 'type': 'float'},
                    {'name': 'thalach', 'type': 'float'},
                    {'name': 'exang', 'type': 'float'},
                    {'name': 'oldpeak', 'type': 'float'},
                    {'name': 'slope', 'type': 'float'},
                    {'name': 'ca', 'type': 'float'},
                    {'name': 'thal', 'type': 'float'}
                ]
            }
        ]
        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.OUTPUT_DATA_SCHEMA: output_data_schema,
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        diet_food = pd.DataFrame([
            ["Roasted Chicken", 0.84, 0, 10],
            ["Spaghetti W/ Sauce", 0.78, 0, 10],
            ["Tomato,Red,Ripe,Raw", 0.27, 0, 10],
            ["Apple,Raw,W/Skin", 0.24, 0, 10],
            ["Grapes", 0.32, 0, 10],
            ["Chocolate Chip Cookies", 0.03, 0, 10],
            ["Lowfat Milk", 0.23, 0, 10],
            ["Raisin Brn", 0.34, 0, 10],
            ["Hotdog", 0.31, 0, 10]], columns=["name", "unit_cost", "qmin", "qmax"])

        diet_food_nutrients = pd.DataFrame([
            ["Spaghetti W/ Sauce", 358.2, 80.2, 2.3, 3055.2, 11.6, 58.3, 8.2],
            ["Roasted Chicken", 277.4, 21.9, 1.8, 77.4, 0, 0, 42.2],
            ["Tomato,Red,Ripe,Raw", 25.8, 6.2, 0.6, 766.3, 1.4, 5.7, 1],
            ["Apple,Raw,W/Skin", 81.4, 9.7, 0.2, 73.1, 3.7, 21, 0.3],
            ["Grapes", 15.1, 3.4, 0.1, 24, 0.2, 4.1, 0.2],
            ["Chocolate Chip Cookies", 78.1, 6.2, 0.4, 101.8, 0, 9.3, 0.9],
            ["Lowfat Milk", 121.2, 296.7, 0.1, 500.2, 0, 11.7, 8.1],
            ["Raisin Brn", 115.1, 12.9, 16.8, 1250.2, 4, 27.9, 4],
            ["Hotdog", 242.1, 23.5, 2.3, 0, 0, 18, 10.4]
        ], columns=["Food", "Calories", "Calcium", "Iron", "Vit_A", "Dietary_Fiber", "Carbohydrates", "Protein"])

        diet_nutrients = pd.DataFrame([
            ["Calories", 2000, 2500],
            ["Calcium", 800, 1600],
            ["Iron", 10, 30],
            ["Vit_A", 5000, 50000],
            ["Dietary_Fiber", 25, 100],
            ["Carbohydrates", 0, 300],
            ["Protein", 50, 100]
        ], columns=["name", "qmin", "qmax"])

        return {
            self.wml_client.deployments.DecisionOptimizationMetaNames.INPUT_DATA: [
                {
                    "id": "diet_food.csv",
                    "values": diet_food
                },
                {
                    "id": "diet_food_nutrients.csv",
                    "values": diet_food_nutrients
                },
                {
                    "id": "diet_nutrients.csv",
                    "values": diet_nutrients
                }
            ],
            self.wml_client.deployments.DecisionOptimizationMetaNames.OUTPUT_DATA: [
                {
                    "id": ".*.csv"
                }
            ]
        }


if __name__ == "__main__":
    unittest.main()
