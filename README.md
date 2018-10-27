[https://github.com/openscoring/papis.io](https://github.com/openscoring/papis.io)
=========================================

[PAPIs 2018](https://www.papis.io/2018) tool demonstration: [Putting five ML models to production in five minutes](https://papis2018.sched.com/event/FnJW/putting-five-ml-models-to-production-in-five-minutes)

# Table of Contents #

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation and usage](#installation-and-usage)
    + [R](#r)
    + [Scikit-Learn](#scikit-learn)
    + [Apache Spark](#apache-spark)
    + [Openscoring](#openscoring)
- [TL;DR, Demo](#tldr-demo)
- [Demo](#demo)
    + [Initialization](#initialization)
    + [Logistic Regression in R](#logistic-regression-in-r)
    + [XGBoost classification in Scikit-Learn](#xgboost-classification-in-scikit-learn)
    + [H2O.ai Distributed Random Forest (DRF) classification in Scikit-Learn](#h2oai-distributed-random-forest-drf-classification-in-scikit-learn)
    + [Regularized (Elastic net) Logistic Regression in Apache Spark](#regularized-elastic-net-logistic-regression-in-apache-spark)
    + [Business rules classification in Scikit-Learn](#business-rules-classification-in-scikit-learn)
    + [Scoring data](#scoring-data)
- [Further reading](#further-reading)
- [Contact](#contact)

# Introduction #

The field of data science is split between two paradigms:

| | **Structured** (ML) | **Unstructured** (AI) |
| --- | --- | --- |
| Scale | Small to large | Medium to extremely large |
| Data | Relational | Images, videos, text |
| Feature type | Scalar | Array/matrix |
| Workflows | Manual, intelligent | Automated, brute-force |
| Hardware | Commodity (CPU) | Specialized (GPU, TPU) |
| Results | Explainable | "Black-box" |
| Standards | [PMML](http://dmg.org/) | [ONNX](https://onnx.ai/), [TensorFlow](https://www.tensorflow.org/) |

The domain of structured data science is based on a solid foundation (statistics), and is responsible for delivering the majority of business value today and in the foreseeable future.

Everything about data science is a lucrative and fast-growing market for software vendors. Legacy and continuation projects are typically served by proprietary/closed-source solutions. However, new projects tend to gravitate towards free- and open-source software (FOSS) solutions because of their superior functional and technical capabilities, and support options.

Dominant FOSS ML frameworks:

* [R](https://www.r-project.org/)
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [Apache Spark](https://spark.apache.org/)

On top of frameworks, there are a number of independent FOSS ML algorithm:

* [H2O.ai](https://www.h2o.ai/)
* [XGBoost](https://github.com/dmlc/xgboost)
* [LightGBM](https://github.com/Microsoft/LightGBM)

Third-party algorithms can deliver significant performance, predictivity and explainability gains over built-in algorithms.

The biggest issue with FOSS ML frameworks and algorithms is the difficulty of moving trained models "from the laboratory to the factory". There are two sides to it. First, the trained model object is functionally very tightly coupled to the original environment. Second, enterprise application programming languages such as Java, C# and SQL do not provide meaningful interoperability with R and Python.

Dominant productionalization strategies:

* Containerization.
* Translation from R/Python representation to Java/C#/SQL application code.
* Translation from R/Python representation to standardized intermediate representation.

This tool demonstration is about the third strategy. We shall 1) train models using popular FOSS ML frameworks and algorithms, 2) translate them from their native R/Scikit-Learn/Apache Spark representation to the standardized Predictive Model Markup Language (PMML) representation, and 3) deploy them as such using the Openscoring REST web service.

# Prerequisites

* Java 1.8 or newer. The Java executable (`java.exe`) must be available on system path.
* R 3.3 or newer
* Python 2.7, 3.3 or newer
* Apache Spark 2.0 or newer

# Installation and usage #

### R

The conversion is handled by the [`r2pmml`](https://github.com/jpmml/r2pmml) package.

This package is not available on CRAN. It can only be installed from its GitHub repository using the [`devtools`](https://cran.r-project.org/package=devtools) package:

```R
library("devtools")

install_git("git://github.com/jpmml/r2pmml.git")
```

The conversion functionality is available via the `r2pmml::r2pmml(obj, pmml_path)` function:

```R
library("r2pmml")

glm.obj = glm(y ~ ., data = mydata)

r2pmml(glm.obj, "MyModel.pmml")
```

### Scikit-Learn

The conversion is handled by the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package.

This package is available on PyPI. Alternatively, it can be installed from its GitHub repository:

```
$ pip install git+https://github.com/jpmml/sklearn2pmml.git
``` 

The `sklearn2pmml` package is "softly dependent" on `h2o`, `lightgbm` and `xgboost` packages. This tool demonstration needs two of them, so they must be installed separately:

```
$ pip install h2o xgboost
```

The layout of H2O.ai's MOJO files is version-dependent. The [JPMML-H2O](https://github.com/jpmml/jpmml-h2o) library, which handles the conversion of H2O.ai's MOJO files, works correctly with H2O.ai 3.16.X and 3.18.X versions; it may, or may not, work correctly with older/newer H2O.ai versions.

Installing the latest definitely supported `h2o` package version:

```
$ pip install -Iv h2o==3.18.0.11
```

The conversion functionality is available via the `sklearn2pmml.sklearn2pmml(pmml_pipeline, pmml_path)` function:

```Python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([...])

sklearn2pmml(pipeline, "MyModel.pmml")
```

The only code change required is using `sklearn2pmml.pipeline.PMMLPipeline` instead of `sklearn.pipeline.Pipeline`. The former is a direct descendant of the latter (hence providing full API compatibility), but adds behind-the-scenes metadata collection and a couple of PMML-related methods (decision engineering, model configuration and verification).

### Apache Spark

The conversion is handled by the [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library. R and Python users might feel more comfortable working with [`sparklyr2pmml`](https://github.com/jpmml/sparklyr2pmml) and [`pyspark2pmml`](https://github.com/jpmml/pyspark2pmml) packages, respectively.

End users are advised to download a JPMML-SparkML release version from its GitHub releases page: https://github.com/jpmml/jpmml-sparkml/releases

The JPMML-SparkML library is being developed and released in four parallel version lines, one for each supported Apache Spark version line:

| JPMML-SparkML | Apache Spark |
| --- | --- |
| [1.1.X](https://github.com/jpmml/jpmml-sparkml/tree/1.1.X) | 2.0.X |
| [1.2.X](https://github.com/jpmml/jpmml-sparkml/tree/1.2.X) | 2.1.X |
| [1.3.X](https://github.com/jpmml/jpmml-sparkml/tree/1.3.X) | 2.2.X |
| [1.4.X](https://github.com/jpmml/jpmml-sparkml/tree/master) | 2.3.X |

For example, if targeting Apache Spark 2.3.X, then the end user should download the latest JPMML-SparkML 1.4.X version (1.4.6 at the time of PAPIs.io 2018).

The JPMML-SparkML library should be appended to Apache Spark application classpath. For command-line applications, this can be easily done using the `--jars` option:

```
$ spark-submit --jars jpmml-sparkml-executable-${version}.jar <app jar | python file | R file>
```

The conversion functionality is available via the `org.jpmml.sparkml.PMMLBuilder` builder class:

```Java
DataFrame df = ...
Pipeline pipeline = ...

PipelineModel pipelineModel = pipeline.fit(df);

PMMLBuilder pmmlBuilder = new PMMLBuilder(df.schema(), pipelineModel);

pmmlBuilder.buildFile(new File("MyModel.pmml"));
```

### Openscoring

The [Openscoring](https://github.com/openscoring/openscoring) REST web service is a thin JAX-RS wrapper around the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

Openscoring provides a microservices-style approach for turning static PMML documents into live functions:

* Commissioning and decommissioning
* Schema querying
* Evaluation in single prediction, batch prediction and CSV prediction modes
* Metrics

End users are advised to download an Openscoring release version from its GitHub releases page: https://github.com/openscoring/openscoring/releases 

Starting up the standalone edition:

```
$ java -jar openscoring-server-executable-${version}.jar
```

By default, Openscoring binds to `localhost:8080`, using `/openscoring` as the web context root. If the startup was successful, then performing an HTTP GET query against the model collection endpoint [`model/`](http://localhost:8080/openscoring/model) should return an empty JSON array `{}`.

Further interaction is possible using HTTP toolkits such as [cURL](https://curl.haxx.se/) or [postman](https://www.getpostman.com/).

Emulating the full lifecycle of a model using cURL:

```
$ curl -X PUT --data-binary @MyModel.pmml -H "Content-type: text/xml" http://localhost:8080/openscoring/model/MyModel
$ curl -X GET http://localhost:8080/openscoring/model/MyModel
$ curl -X POST --data-binary @input.csv -H "Content-type: text/plain; charset=UTF-8" http://localhost:8080/openscoring/model/MyModel/csv > output.csv
$ curl -X DELETE http://localhost:8080/openscoring/model/MyModel
```

R and Python users might feel more comfortable working with [`openscoring-r`](https://github.com/openscoring/openscoring-r) and [`openscoring-python`](https://github.com/openscoring/openscoring-python) packages, respectively.

Emulating the full lifecycle of a model using the `openscoring-python` package:

```Python
from openscoring import Openscoring

os = Openscoring(base_url = "http://localhost:8080/openscoring")
os.deployFile("MyModel", "MyModel.pmml")
os.evaluateCsvFile("MyModel", "input.csv", "output.csv")
os.undeploy("MyModel")
```

# TL;DR, Demo #

Initialization:

```
$ java -jar openscoring-server-executable-${version}.jar
```

Training, converting and deploying models:

```
$ Rscript --vanilla GLMAudit.R --deploy
$ python XGBoostAudit.py --deploy
$ python RandomForestAudit.py --deploy
$ spark-shell --jars jpmml-sparkml-executable-${version}.jar,openscoring-client-executable-${version}.jar -i ElasticNetAudit.scala --conf spark.driver.args="--deploy"
$ python RuleSetIris.py --deploy
```

Scoring data:

```
$ curl -X POST --data-binary @csv/Audit.csv -H "Content-type: text/plain; charset=UTF-8" http://localhost:8080/openscoring/model/RandomForestAudit/csv > RandomForestAudit.csv
$ curl -X POST --data-binary @csv/Iris.csv -H "Content-type: text/plain; charset=UTF-8" http://localhost:8080/openscoring/model/RuleSetIris/csv > RuleSetIris.csv
```

# Demo #

### Initialization

Starting up Openscoring:

```
$ java -jar openscoring-server-executable-${version}.jar
```

### Logistic Regression in R

The R scipt file: [GLMAudit.R](GLMAudit.R)

All feature engineering should be done using the [model formula](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html) approach in order to make it part of the model object state (ie. can be saved and read back into memory using [`base::saveRDS(obj, path)`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/readRDS.html) and [`base::readRDS(path)`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/readRDS.html) functions).

Binning the "Age" feature using the [`base::cut(x, breaks)`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/cut.html) function:

```R
ageQuantiles = quantiles(audit$Age)

audit.formula = formula(Adjusted ~ . - Age + base::cut(Age, breaks = ageQuantiles))
```

Interacting "Gender" and "Marital" features using the `:` operator:

```R
audit.formula = formula(Adjusted ~ . + Gender:Marital)
```

Deriving an hourly income based on "Income" (annual income) and "Hours" (the number of working hours in a week) features using arithmetic operators; as a matter of caution, all inline R expressions should be surrounded with the [`base::I(x)`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/AsIs.html) function:

```R
audit.formula = formula(Adjusted ~ . + I(Income / (Hours * 52)))
```

After training, the model object is enhanced with verification data using the `r2pmml::verify(obj, newdata)` function:

```R
library("r2pmml")

audit.glm = glm(Adjusted ~ ., data = audit)

# Discard known values of the dependent variable
audit$Adjusted = NULL

audit.glm = verify(audit.glm, audit[sample(nrow(audit), 100), ])
```

Running the R script file:

```
$ Rscript --vanilla GLMAudit.R --deploy
```

The generated PMML document is saved as `pmml/GLMAudit.pmml` and deployed to Openscoring as [`model/GLMAudit`](http://localhost:8080/openscoring/model/GLMAudit).

### XGBoost classification in Scikit-Learn

The Python script file: [XGBoostAudit.py](XGBoostAudit.py)

All column-oriented feature engineering should be done using the `sklearn_pandas.DataFrameMapper` meta-transformer class:

```Python
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in [...]] +
	[([cont_column], [ContinuousDomain()]) for cont_column in [...]]
)
```

Binning the "Age" feature using the `sklearn2pmml.preprocessing.CutTransformer` transformer class:

```Python
from sklearn2pmml.preprocessing import CutTransformer

mapper = DataFrameMapper([
	("Age", [ContinuousDomain(), CutTransformer(bins = [17, 28, 37, 47, 83], labels = ["q1", "q2", "q3", "q4"]), LabelBinarizer()])
])
```

Interacting "Gender" and "Marital" features using the `sklearn.preprocessing.PolynomialFeatures` transformer class:

```Python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures

union = FeatureUnion([
	("scalar_mapper", DataFrameMapper([...])),
	("interaction_pipeline", Pipeline([
		("interaction_mapper", DataFrameMapper([
			("Gender", [CategoricalDomain(), LabelBinarizer()]),
			("Marital", [CategoricalDomain(), LabelBinarizer()])
		])),
		("polynomial_features", PolynomialFeatures())
	]))
])
```

Deriving an hourly income based on "Income" and "Hours" features using the `sklearn2pmml.preprocessing.ExpressionTransformer` transformer class:

```Python
from sklearn2pmml.decoration import Alias
from sklearn2pmml.preprocessing import ExpressionTransformer

mapper = DataFrameMapper([
	(["Hours", "Income"], Alias(ExpressionTransformer("X[1] / (X[0] * 52)"), "Hourly_Income"))
])
```

After training, the model object is re-encoded from binary splits to multi-way splits using the `PMMLPipeline.configure(**pmml_options)` method, and enhanced with verification data using the `PMMLPipeline.verify(X, precision, zeroThreshold)` method:

```Python
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([...])

pipeline.configure(compact = True)
pipeline.verify(audit_X.sample(100), zeroThreshold = 1e-6, precision = 1e-6)
```

Running the Python script file:

```
$ python XGBoostAudit.py --deploy
```

The generated PMML document is saved as `pmml/XGBoostAudit.pmml` and deployed to Openscoring as [`model/XGBoostAudit`](http://localhost:8080/openscoring/model/XGBoostAudit).


### H2O.ai Distributed Random Forest (DRF) classification in Scikit-Learn

The Python script file: [RandomForestAudit.py](RandomForestAudit.py)

H2O.ai algorithms provide full support for string categorical features. This is in stark contrast with other Python-accessible ML algorithms that require them to be binarized in one-hot-encoding fashion (eg. Scikit-Learn, XGBoost) or at least re-encoded (eg. LightGBM):

```Python
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain()]) for cat_column in [...]] +
	[([cont_column], [ContinuousDomain()]) for cont_column in [...]]
)
```

All feature engineering happens in local computer using Scikit-Learn transformer classes. The pre-processed dataset (could be a `pandas.DataFrane` or a Numpy matrix) is then uploaded to the remove computer where the H2O.ai compute engine resides using the `sklearn2pmml.preprocessing.h2o.H2OFrameCreator` meta-transformer class:

```Python
from h2o import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn2pmml.preprocessing.h2o import H2OFrameCreator

pipeline = PMMLPipeline([
	("local_mapper", DataFrameMapper([...])),
	("uploaded", H2OFrameCreator()),
	("remote_classifier", H2ORandomForestEstimator())
])
pipeline.fit(audit_X, H2OFrame(audit_y.to_frame(), column_types = ["categorical"]))
```

A `Pipeline.predict_proba(X)` method call returns a two-column matrix for binary classification problems, where the first column holds the probability of the negative ("no-event") scenario and the second column holds the probability of the positive ("event") scenario.

The Scikit-Learn framework does not support decision engineering (eg. appending transformation steps to the final estimator step) based on predicted labels or probability distributions. 

The `PMMLPipeline` class makes it possible by adding the following attributes and methods:

| Attribute | Method |
| --- | --- |
| `predict_transformer` | `predict_transform(X)` |
| `predict_proba_transformer` | `predict_proba_transform(X)` |
| `apply_transformer` | N/A |

Binning the probability of the positive scenario using the `CutTransformer` transformer class:

```Python
predict_proba_transformer = Pipeline([
	("expression", ExpressionTransformer("X[1]")),
	("cut", Alias(CutTransformer(bins = [0.0, 0.75, 0.90, 1.0], labels = ["no", "maybe", "yes"]), "Decision", prefit = True))
])

pipeline = PMMLPipeline([...], predict_proba_transformer = predict_proba_transformer)
pipeline.fit(audit_X, H2OFrame(audit_y.to_frame(), column_types = ["categorical"]))

pipeline.predict_proba_transform(audit_X)
```

Running the Python script file:

```
$ python RandomForestAudit.py --deploy
```

The generated PMML document is saved as `pmml/RandomForestAudit.pmml` and deployed to Openscoring as [`model/RandomForestAudit`](http://localhost:8080/openscoring/model/RandomForestAudit).

### Regularized (Elastic net) Logistic Regression in Apache Spark

The Scala script file: [ElasticNetAudit.scala](ElasticNetAudit.scala)

Apache Spark pipelines are much more flexible than Scikit-Learn pipelines. Specifically, they support model chains, transformations between models and after the last model. The JPMML-SparkML library should be able to convert all that into the standardized PMML representation in a fully automated way.

Binning the "Age" feature using the `org.apache.spark.ml.feature.QuantileDiscretizer` transformer class:

```Scala
val ageDiscretizer = new QuantileDiscretizer()
	.setNumBuckets(4)
	.setInputCol("Age")
	.setOutputCol("discretizedAge");
```

Interacting "Gender" and "Marital" features using the `org.apache.spark.ml.feature.Interaction` transformer class:

```Scala
val genderMaritalInteraction = new Interaction()
	.setInputCols(Array("encodedGender", "encodedMarital"))
	.setOutputCol("interactedGenderMarital");
```

Searching for the best regularization parameter using the `org.apache.spark.ml.tuning.CrossValidator` meta-estimator class:

```Scala
val logisticRegression = new LogisticRegression()
	.setElasticNetParam(0.5)
	.setFeaturesCol("vectorizedFeatures")
	.setLabelCol("indexedAdjusted");

stages += logisticRegression	

val estimator = new Pipeline().setStages(stages.toArray)
val estimatorParamMaps = new ParamGridBuilder().addGrid(logisticRegression.regParam, Array(0.05, 0.10, 0.15)).build()
val evaluator = new BinaryClassificationEvaluator().setLabelCol("indexedAdjusted")

val crossValidator = new CrossValidator()
	.setEstimator(estimator)
	.setEstimatorParamMaps(estimatorParamMaps)
	.setEvaluator(evaluator)
	.setSeed(42L);

val pipeline = new Pipeline().setStages(Array(crossValidator))
val pipelineModel = pipeline.fit(df)
```

Running the Scala script without Openscoring deployment:

```
$ spark-shell --jars jpmml-sparkml-executable-${version}.jar -i ElasticNetAudit.scala
```

The generated PMML document is saved as `pmml/ElasticNetAudit.pmml`.

Running the Scala script with Openscoring deployment:

```
$ spark-shell --jars jpmml-sparkml-executable-${version}.jar,openscoring-client-executable-${version}.jar -i ElasticNetAudit.scala --conf spark.driver.args="--deploy"
```

The generated PMML document is saved as `pmml/ElasticNetAudit.pmml` and deployed to Openscoring as [`model/ElasticNetAudit`](http://localhost:8080/openscoring/model/ElasticNetAudit).

### Business rules classification in Scikit-Learn

The Python script file: [RuleSetIris.py](RuleSetIris.py)

There are data science problems where the solution is obvious/known in advance, and the whole machine learning workflow is reduced to just writing down the function.

Generating PMML documents manually is not too difficult. However, it would be a major usability/productivity advance if end users could accomplish everything from within their favourite environment, without having to learn and do anything new.

The `sklearn2pmml` package provides the `sklearn2pmml.ruleset.RuleSetClassifier` estimator class, which allows a data record to be labeled by matching it against a collection of Python predicates (ie. boolean expressions).

Implementing a decision tree-like solution:

```Python
from sklearn2pmml.ruleset import RuleSetClassifier

classifier = RuleSetClassifier([
	("X['Petal_Length'] < 2.45", "setosa"),
	("X['Petal_Width'] < 1.75", "versicolor"),
], default_score = "virginica")
```

Running the Python script file:

```
$ python RuleSetIris.py --deploy
```

The generated PMML document is saved as `pmml/RuleSetIris.pmml` and deployed to Openscoring as [`model/RuleSetIris`](http://localhost:8080/openscoring/model/RuleSetIris).

### Scoring data

In this point, there should be five models deployed on the Openscoring:

* [`model/GLMAudit`](http://localhost:8080/openscoring/model/GLMAudit)
* [`model/XGBoostAudit`](http://localhost:8080/openscoring/model/XGBoostAudit)
* [`model/RandomForestAudit`](http://localhost:8080/openscoring/model/RandomForestAudit)
* [`model/ElasticNetAudit`](http://localhost:8080/openscoring/model/ElasticNetAudit)
* [`model/RuleSetIris`](http://localhost:8080/openscoring/model/RuleSetIris)

Scoring the [`csv/Audit.CSV`](csv/Audit.csv) input file with the `RandomForestAudit` model using cURL:

```
$ curl -X POST --data-binary @csv/Audit.csv -H "Content-type: text/plain; charset=UTF-8" http://localhost:8080/openscoring/model/RandomForestAudit/csv > RandomForestAudit.csv
```

The `RandomForestAudit.csv` results file contains five columns - the "Adjusted" target column, and "probability(0)", "probability(1)", "eval(X[1])" and "Decision" output columns. The last one holds the the outcome of our decision engineering efforts - all in all there are 154 "yes" decisions, 153 "maybe" decisions and 1592 "no" decisions.

Scoring the [`csv/Iris.csv`](csv/Iris.csv) input file with the `RuleSetIris` model using cURL:

```
$ curl -X POST --data-binary @csv/Iris.csv -H "Content-type: text/plain; charset=UTF-8" http://localhost:8080/openscoring/model/RuleSetIris/csv > RuleSetIris.csv
```

The `RuleSetIris.csv` results file contains a single "Species" target column.

# Further reading #

Presentations:

* [State of the (J)PMML art](https://www.slideshare.net/VilluRuusmann/state-of-the-jpmml-art)
* [Converting R to PMML](https://www.slideshare.net/VilluRuusmann/converting-r-to-pmml-82182483)
* [Converting Scikit-Learn to PMML](https://www.slideshare.net/VilluRuusmann/converting-scikitlearn-to-pmml)

Software:

* [Java PMML API](https://github.com/jpmml)
* [Openscoring REST API](https://github.com/openscoring)

# Contact #

Villu Ruusmann  
CTO and Founder at Openscoring OÜ, Estonia

GitHub: https://github.com/vruusmann  
LinkedIn: https://ee.linkedin.com/in/villuruusmann/  
SlideShare: https://slideshare.net/VilluRuusmann  
e-mail: villu@openscoring.io  
Skype: villu.ruusmann