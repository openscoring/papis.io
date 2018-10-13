from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.ruleset import RuleSetClassifier

import pandas

iris_df = pandas.read_csv("csv/Iris.csv")
#print(iris_df.head(5))

iris_X = iris_df[iris_df.columns.difference(["Species"])]
iris_y = iris_df["Species"]

classifier = RuleSetClassifier([
	("X['Petal_Length'] < 2.45", "setosa"),
	("X['Petal_Width'] < 1.75", "versicolor"),
], default_score = "virginica")

pipeline = PMMLPipeline([
	("classifier", classifier)
])
pipeline.fit(iris_X, iris_y)

sklearn2pmml(pipeline, "pmml/RuleSetIris.pmml")