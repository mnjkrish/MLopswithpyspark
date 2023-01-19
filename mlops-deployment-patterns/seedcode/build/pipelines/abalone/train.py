# mods
import os
import argparse
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import pyspark.sql.functions as f
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# create app
spark = SparkSession.builder.appName('spark-python-sagemaker-training').getOrCreate()

# schema

def main():
    
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-input-bucket', type=str)
    parser.add_argument('--s3-input-prefix', type=str)
    parser.add_argument('--repartition-num', type=int)
    args = parser.parse_args()
    
    # read dataset
    
    df = spark.read.format('parquet').options(header=True,inferSchema=True).load('s3://' + os.path.join(args.s3_input_bucket, args.s3_input_prefix))
    print(df.show(5))
    df.show(5)
    
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=30)
    featurizedData = hashingTF.transform(df)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    (train_df, validate_df) = rescaledData.randomSplit([0.85, 0.15], seed=0)
    
        ## Fitting the model
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'category_index')
    rfModel = rf.fit(train_df)
    rfPreds = rfModel.transform(validate_df)
    ## Evaluating the model
    evaluator = MulticlassClassificationEvaluator(labelCol="category_index", predictionCol="prediction", metricName="f1")
    rf_f1_score = evaluator.evaluate(rfPreds)
    print("F1 of Random Forests is = %g"% (rf_f1_score))
    
    # kill app
    spark.stop()

if __name__ == '__main__':
    main()