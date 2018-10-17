from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import CutTransformer, ExpressionTransformer
from xgboost import XGBClassifier

import pandas
import sys

audit_df = pandas.read_csv("csv/Audit.csv")
#print(audit_df.head(5))

audit_X = audit_df[audit_df.columns.difference(["Adjusted"])]
audit_y = audit_df["Adjusted"]

scalar_mapper = DataFrameMapper([
	("Education", [CategoricalDomain(), LabelBinarizer(), SelectKBest(chi2, k = 3)]),
	("Employment", [CategoricalDomain(), LabelBinarizer(), SelectKBest(chi2, k = 3)]),
	("Occupation", [CategoricalDomain(), LabelBinarizer(), SelectKBest(chi2, k = 3)]),
	("Age", [ContinuousDomain(), CutTransformer(bins = [17, 28, 37, 47, 83], labels = ["q1", "q2", "q3", "q4"]), LabelBinarizer()]),
	("Hours", ContinuousDomain()),
	("Income", ContinuousDomain()),
	(["Hours", "Income"], Alias(ExpressionTransformer("X[1] / (X[0] * 52)"), "Hourly_Income"))
])
interaction_mapper = DataFrameMapper([
	("Gender", [CategoricalDomain(), LabelBinarizer()]),
	("Marital", [CategoricalDomain(), LabelBinarizer()])
])
classifier = XGBClassifier()

pipeline = PMMLPipeline([
	("mapper", FeatureUnion([
		("scalar_mapper", scalar_mapper),
		("interaction", Pipeline([
			("interaction_mapper", interaction_mapper),
			("polynomial", PolynomialFeatures())
		]))
	])),
	("classifier", classifier)
])
pipeline.fit(audit_X, audit_y)

pipeline.configure(compact = True)
pipeline.verify(audit_X.sample(100), zeroThreshold = 1e-6, precision = 1e-6)

sklearn2pmml(pipeline, "pmml/XGBoostAudit.pmml")

if "--deploy" in sys.argv:
	from openscoring import Openscoring

	os = Openscoring("http://localhost:8080/openscoring")
	os.deployFile("XGBoostAudit", "pmml/XGBoostAudit.pmml")