import java.io.File
import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.jpmml.sparkml.PMMLBuilder

val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("csv/Audit.csv")

val stages = new ListBuffer[PipelineStage]()
val featureCols = new ListBuffer[String]()

stages += new SQLTransformer().setStatement("SELECT * FROM __THIS__")

stages += new StringIndexer().setInputCol("Adjusted").setOutputCol("indexedAdjusted")

for(contCol <- Seq("Hours", "Income")){
	featureCols += contCol
}

for(catCol <- Seq("Education", "Employment", "Gender", "Marital", "Occupation")){
	stages += new StringIndexer().setHandleInvalid("keep").setInputCol(catCol).setOutputCol("indexed" + catCol)
	stages += new OneHotEncoder().setInputCol("indexed" + catCol).setOutputCol("encoded" + catCol)
	featureCols += ("encoded" + catCol)
}

stages += new QuantileDiscretizer().setNumBuckets(4).setInputCol("Age").setOutputCol("discretizedAge")
featureCols += "discretizedAge"

stages += new Interaction().setInputCols(Array("encodedGender", "encodedMarital")).setOutputCol("interactedGenderMarital")
featureCols += "interactedGenderMarital"

stages += new VectorAssembler().setInputCols(featureCols.toArray).setOutputCol("vectorizedFeatures")

val logisticRegression = new LogisticRegression().setElasticNetParam(0.5).setFeaturesCol("vectorizedFeatures").setLabelCol("indexedAdjusted")

stages += logisticRegression

val estimator = new Pipeline().setStages(stages.toArray)
val estimatorParamMaps = new ParamGridBuilder().addGrid(logisticRegression.regParam, Array(0.05, 0.10)).build()
val evaluator = new BinaryClassificationEvaluator().setLabelCol("indexedAdjusted")

val cv = new CrossValidator().setEstimator(estimator).setEstimatorParamMaps(estimatorParamMaps).setEvaluator(evaluator).setNumFolds(2).setParallelism(4).setSeed(42L)

val pipeline = new Pipeline().setStages(Array(cv))
val pipelineModel = pipeline.fit(df)

val pmmlBuilder = new PMMLBuilder(df.schema, pipelineModel).verify(df.sample(false, 0.05))

pmmlBuilder.buildFile(new File("pmml/ElasticNetAudit.pmml"))

System.exit(0)