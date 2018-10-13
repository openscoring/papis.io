from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import CutTransformer, ExpressionTransformer
from xgboost import XGBClassifier

from Audit import audit_X, audit_y

simple_mapper = DataFrameMapper([
	("Employment", [CategoricalDomain(), LabelBinarizer()]),
	("Marital", [CategoricalDomain(), LabelBinarizer()]),
	("Occupation", [CategoricalDomain(), LabelBinarizer()]),
	("Age", [ContinuousDomain(), CutTransformer(bins = [17, 28, 37, 47, 83], labels = ["q1", "q2", "q3", "q4"]), LabelBinarizer()]),
	("Hours", ContinuousDomain()),
	("Income", ContinuousDomain()),
	(["Hours", "Income"], Alias(ExpressionTransformer("X[1] / (X[0] * 52)"), "Hourly Income"))
])
interaction_mapper = DataFrameMapper([
	("Education", [CategoricalDomain(), LabelBinarizer()]),
	("Gender", [CategoricalDomain(), LabelBinarizer()]),
])
classifier = XGBClassifier()

pipeline = PMMLPipeline([
	("mapper", FeatureUnion([
		("simple_mapper", simple_mapper),
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