import java.io.File
import java.util.NoSuchElementException
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.DoubleType
import org.jpmml.sparkml.PMMLBuilder
import scala.collection.mutable.ListBuffer

var args = ""

try {
	args = sc.getConf.get("spark.driver.args")
} catch {
	case nsee: NoSuchElementException => ;
}

var df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("csv/Audit.csv")

df = df.withColumn("Hours", $"Hours".cast(DoubleType))

val stages = new ListBuffer[PipelineStage]()
val featureCols = new ListBuffer[String]()

stages += new SQLTransformer().setStatement("SELECT *, (Income / (Hours * 52)) AS Hourly_Income FROM __THIS__")

stages += new StringIndexer().setInputCol("Adjusted").setOutputCol("indexedAdjusted")

for(contCol <- Seq("Hourly_Income")){
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

if(args.contains("--deploy")){
	import javax.ws.rs.ProcessingException
	import org.openscoring.client.Deployer

	val deployer = new Deployer()
	deployer.setModel("http://localhost:8080/openscoring/model/ElasticNetAudit")
	deployer.setFile(new File("pmml/ElasticNetAudit.pmml"))

	try {
		deployer.run()
	} catch {
		case pe: ProcessingException => ;
	}
}

System.exit(0)
