import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, Word2Vec, MinMaxScaler, Tokenizer, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

from msd_kMeans_short import all_tf, ignore_columns

spark = SparkSession.builder.appName('msd-kMeans-short').getOrCreate()
assert spark.version >= '3.2' # make sure we have Spark 3.2+
spark.sparkContext.setLogLevel('WARN')


######################################################################################


def main(inputs):

    input_train = inputs + "/trainSet"
    #input_test = inputs + "/testSet"

    train_df = spark.read.option("header", "true").csv(input_train, inferSchema=True).cache()
    train_df.drop("filename")
    #train_df.show(1)

    """
    Build pipeline
    """

    scaled_features = [name + '_scaled' for name in train_df.schema.names if name not in ignore_columns]
    final_assembler = VectorAssembler(
        # inputCols=[x for x in df.columns if x not in ignore], 
        inputCols=scaled_features, 
        outputCol='features')

    # Transform and assemble:
    pipeline = Pipeline(stages=[
        # Transformers
        all_tf,
        # # Assemble everything
        final_assembler,
        # KMeans(featuresCol='features', k=10)    #TODO: optmize k later
    ])

    model = pipeline.fit(train_df)

    transformed_df = model.transform(train_df).cache()
    
    index_to_try = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    evaluator = ClusteringEvaluator(featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean')

    silhouette_scores = []
    index = []

    for K in index_to_try:
        KMeans_=KMeans(featuresCol='features', k=K)
        KMeans_fit=KMeans_.fit(transformed_df)
        KMeans_transform=KMeans_fit.transform(transformed_df) 
        evaluation_score=evaluator.evaluate(KMeans_transform)
        silhouette_scores.append(evaluation_score)
        index.append(K)
    
    k_score = np.stack((index, silhouette_scores), axis=-1)
    k_dataframe = pd.DataFrame(data=k_score, columns=('K', 'Silhouette_Score'))
    print(k_dataframe)
    k_dataframe.to_csv('k_index.csv')


if __name__ == '__main__':
    inputs = sys.argv[1]
    

    # sc = spark.sparkContext

    main(inputs)