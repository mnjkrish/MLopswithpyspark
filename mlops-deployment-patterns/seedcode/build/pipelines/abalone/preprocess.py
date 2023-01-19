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
import sys

import boto3

# create app
spark = SparkSession.builder.appName('spark-python-sagemaker-processing').getOrCreate()

# schema
schema = StructType([StructField('category', StringType(), True), 
                     StructField('message', StringType(), True)])

def main():
    
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-input-bucket', type=str)
    parser.add_argument('--s3-input-prefix', type=str)
    parser.add_argument('--train-split-size', type=float)
    parser.add_argument('--test-split-size', type=float)
    parser.add_argument('--s3-output-bucket', type=str)
    parser.add_argument('--s3-output-prefix', type=str)
    parser.add_argument('--repartition-num', type=int)
    args = parser.parse_args()
    
    # read dataset
    print('s3://' + os.path.join(args.s3_input_bucket, args.s3_input_prefix,'spamdataset.csv'))
    print(os.getcwd())

    print("this is the dataset")
    try:
        df = spark.read.csv('s3://' + os.path.join(args.s3_input_bucket, args.s3_input_prefix, 'spamdataset.csv'),
                        header=False,
                           schema= schema)
    except Exception as e:
        print("problem : ",e)
    print("after reading")
    # drop nan
    df = df.na.drop()
    
    df = df.select("*", f.lower(f.col("message")).alias("messages")).drop("message")
    inx = StringIndexer(inputCol='category', outputCol='category_index')
    encoded = inx.fit(df)
    encoded_trans = encoded.transform(df)
    
    tokenizer = Tokenizer(inputCol="messages", outputCol="words")
    wordsData = tokenizer.transform(encoded_trans).drop("messages","category")
    
    
    
    
    
    (train_df, test_df) = wordsData.randomSplit([args.train_split_size, args.test_split_size], seed=0)
    
    # write
    (train_df
    .repartition(args.repartition_num)
    .write
    .mode('overwrite')
    .parquet(('s3://' + os.path.join(args.s3_output_bucket, args.s3_output_prefix, 'train', 'train.parquet'))))
    print("---Writing to output")
    
   
    (test_df
    .repartition(args.repartition_num)
    .write
    .mode('overwrite')
    .parquet(('s3://' + os.path.join(args.s3_output_bucket, args.s3_output_prefix, 'test', 'test.parquet'))))
    print("---Wrote the output")
    # kill app
    spark.stop()

if __name__ == '__main__':
    main()
