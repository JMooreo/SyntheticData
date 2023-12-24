from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBClassifier

from datasets.Dataset import Dataset
from evaluation.Evaluator import Evaluator
from machine_learning.MachineLearningMethod import MachineLearningMethod
from machine_learning.Preprocessing import load_data_and_clean_columns


class ConfusionMatrixEvaluator(Evaluator):

	def __init__(self, dataset: Dataset, machine_learning_method: MachineLearningMethod, train_data_path: str, test_data_path: str):
		self.dataset = dataset
		self.machine_learning_method = machine_learning_method
		self.train_data_path = train_data_path
		self.test_data_path = test_data_path
		self.dataset.load()

	def evaluate(self):
		training_data, _, _ = self.dataset.load_dataframe_and_columns(self.train_data_path)

		x_train, x_test, y_train, y_test, classes = load_data_and_clean_columns(
			training_data, self.dataset, self.test_data_path)

		if self.machine_learning_method == MachineLearningMethod.LOGISTIC_REGRESSION:
			model = LogisticRegression(max_iter=10000).fit(x_train, y_train)
		elif self.machine_learning_method == MachineLearningMethod.XGBOOST:
			model = XGBClassifier().fit(x_train, y_train)
		else:
			raise ValueError(f"Invalid machine learning method: {self.machine_learning_method}")

		y_pred = model.predict(x_test)
		# micro_f1 = f1_score(y_test, y_pred, average="micro")
		# macro_f1 = f1_score(y_test, y_pred, average="macro")
		weighted_f1 = f1_score(y_test, y_pred, average="weighted")

		return confusion_matrix(y_test, y_pred), classes, weighted_f1

