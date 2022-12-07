# A locally test code on the subset
## Setup:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
import tarfile
import h5py
import hdf5_getters
import os
from glob import glob

from pyspark.sql import SparkSession, functions, types, Row

from pyspark.sql.types import ArrayType, FloatType, StringType


# Stats:
import boto3
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import Imputer
from pyspark.ml.classification import RandomForestClassifier,GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt
import statistics
# To initialize spark
from pyspark.sql import SparkSession, functions, types, Row
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder \
     .appName("Test SparkSession") \
     .getOrCreate()
schema = types.StructType([
    types.StructField('filename', types.StringType()),
    types.StructField('artist_familiarity', types.FloatType()),
    types.StructField('artist_hotttnesss', types.FloatType()),
    types.StructField('artist_id', types.StringType()),
    types.StructField('artist_mbid', types.StringType()),
    #5
    types.StructField('artist_playmeid', types.IntegerType()),
    types.StructField('artist_7digitalid', types.IntegerType()),
    types.StructField('artist_latitude', types.FloatType()),
    types.StructField('artist_longitude', types.FloatType()),
    types.StructField('artist_location', types.StringType()),
    #10
    types.StructField('artist_name', types.StringType()),
    types.StructField('release', types.StringType()),
    types.StructField('release_7digitalid', types.IntegerType()),
    types.StructField('song_id', types.StringType()),
    types.StructField('song_hotttnesss', types.FloatType()),
    #15
    types.StructField('title', types.StringType()),
    types.StructField('track_7digitalid', types.IntegerType()),
    types.StructField('similar_artists', types.ArrayType(types.StringType())),
    types.StructField('artist_terms', types.ArrayType(types.StringType())),
    types.StructField('artist_terms_freq', types.ArrayType(types.FloatType())),
    #20
    types.StructField('artist_terms_weight', types.ArrayType(types.FloatType())),
    types.StructField('analysis_sample_rate', types.FloatType()),
    types.StructField('audio_md5', types.StringType()),
    types.StructField('danceability', types.FloatType()),
    types.StructField('duration', types.FloatType()),
    #25
    types.StructField('end_of_fade_in', types.FloatType()),
    types.StructField('energy', types.FloatType()),
    types.StructField('key', types.IntegerType()),
    types.StructField('key_confidence', types.FloatType()),
    types.StructField('loudness', types.FloatType()),
    #30
    types.StructField('mode', types.IntegerType()),
    types.StructField('mode_confidence', types.FloatType()),
    types.StructField('start_of_fade_out', types.FloatType()),
    types.StructField('tempo', types.FloatType()),
    types.StructField('time_signature', types.IntegerType()),    
    #35
    types.StructField('time_signature_confidence', types.FloatType()),
    types.StructField('track_id', types.StringType()),
    types.StructField('segments_start', types.ArrayType(types.FloatType())),
    types.StructField('segments_confidence', types.ArrayType(types.FloatType())),
    types.StructField('segments_pitches', types.ArrayType(types.ArrayType(types.FloatType()))), 
    #40
    types.StructField('segments_timbre', types.ArrayType(types.ArrayType(types.FloatType()))),
    types.StructField('segments_loudness_max', types.ArrayType(types.FloatType())),
    types.StructField('segments_loudness_max_time', types.ArrayType(types.FloatType())),
    types.StructField('segments_loudness_start', types.ArrayType(types.FloatType())),
    types.StructField('sections_start', types.ArrayType(types.FloatType())),
    #45
    types.StructField('sections_confidence', types.ArrayType(types.FloatType())),
    types.StructField('beats_start', types.ArrayType(types.FloatType())),
    types.StructField('beats_confidence', types.ArrayType(types.FloatType())),
    types.StructField('bars_start', types.ArrayType(types.FloatType())),
    types.StructField('bars_confidence', types.ArrayType(types.FloatType())),
    #50
    types.StructField('tatums_start', types.ArrayType(types.FloatType())),
    types.StructField('tatums_confidence', types.ArrayType(types.FloatType())),
    types.StructField('artist_mbtags', types.ArrayType(types.StringType())),
    types.StructField('artist_mbtags_count', types.ArrayType(types.IntegerType())),
    types.StructField('year', types.IntegerType()),
    #55
])

def two_d_array2one (tda):
    result = [0,0,0,0,0,0,0,0,0,0,0,0]
    for idx in range(0, 12):
        for l in tda:
            result[idx] += l[idx]
        result[idx] = result[idx]/len(tda)
    return result


def main():
    ## Subset files read:
    input_path = '/Users/billyshi/Documents/Code/CMPT353_Code/Project/msd_subset'
    spark_df = spark.read.json(input_path, schema=schema)
    spark_df.printSchema()
    spark_df.count()
    # Convert spark dataframe to pandas, accelerate using pyarrow
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set(
        "spark.sql.execution.arrow.pyspark.fallback.enabled", "true"
    )
    # Easier quick peek on the dataframe
    pd.set_option("display.max_columns", None)
    pandas_df = spark_df.limit(7).toPandas()
    pandas_df.head()
    array = (pandas_df['segments_timbre'][0])
    array
    ## Clean Up:
    # Remove duplicate
    duplicates_removed = spark_df.dropDuplicates()

    # Remove if year is unknown (For year prediction model)
    #filtered = duplicates_removed.filter(duplicates_removed['year'] != 0)

    duplicates_removed.count()
    cleanedup_df = duplicates_removed
    cleanedup_df = cleanedup_df.cache()
    # cleanedup_df.show(1)
    ## Stats and Modelling:
    ### kMeans:
    # columns to be selected for transformation
    stringTypes = [ 
        'filename',
        'artist_id', 
        'artist_name',
        'artist_location', 
        'title'
    ]

    stringArrTypes = [         
        'artist_terms', # TODO: Decide if want to continue with using terms/tags
        # 'similar_artists',
    ]

    # also include Int types
    floatTypes = [
        'artist_familiarity', # use to filter out low familiarity artists?
        'artist_hotttnesss',
        'artist_latitude', 'artist_longitude',
        'song_hotttnesss', 'danceability', 'energy',
        'end_of_fade_in', 'start_of_fade_out',
        'key', 'key_confidence',
        'duration', 
        'loudness', #TODO: shift +60dB (since -60dB is the zero in raw data)
        'tempo', 
        'mode', 'mode_confidence',
        'time_signature', 'time_signature_confidence',
        'year', #TODO: shift to delta since min year
    ]

    floatArrTypes = [
        # 'artist_terms_freq', 'artist_terms_weight', #TODO: VectorAssemble with 'artist_terms'
        'segments_start', 'segments_confidence',
        'segments_loudness_max', 'segments_loudness_start',
        'sections_start', 'sections_confidence',
        'beats_start', 'beats_confidence',
        'bars_start', 'bars_confidence',
        'tatums_start', 'tatums_confidence'
    ]

    # 2D float arrays
    specialArrTypes = ['segments_pitches', 'segments_timbre']
    feature_cols = stringTypes + stringArrTypes + floatTypes + floatArrTypes + specialArrTypes
    kMeans_df = cleanedup_df.select(feature_cols)
    # tda = kMeans_df.first()['segments_timbre']
    # result = [0,0,0,0,0,0,0,0,0,0,0, 0]
    # for idx in range(0, 12):
    #     for l in tda:
    #         result[idx] += l[idx]
    #     print(result[idx]/len(tda))
    #     result[idx] = result[idx]/len(tda)
    # result
    # kMeans_df = kMeans_df.withColumn(
    #     'segments_timbre_vec', 
    #     functions.transform(        # map each element (https://spark.apache.org/docs/3.2.0/api/python/reference/api/pyspark.sql.functions.transform.html)
    #         'segments_timbre',
    #         lambda x : array_to_vector(x)
    #     )
    # )
    two_d_array_converter = functions.udf(two_d_array2one, ArrayType(types.FloatType()))
    cleaned_kMeans_df = kMeans_df.select(
        kMeans_df['filename'],
        array_to_vector(two_d_array_converter(kMeans_df['segments_timbre'])).alias('segments_timbre_combined'),
        array_to_vector(two_d_array_converter(kMeans_df['segments_pitches'])).alias('segments_pitches_combined'),
    )
    #cleaned_kMeans_df.show(truncate=False)
    # Drop rows with any empty fields
    cleaned_kMeans_df = cleaned_kMeans_df.dropna("any") # Note: in 10k subset, only 2210 songs left
    rows = cleaned_kMeans_df.count() # should have less than 10k songs now
    print(f"Number of rows left: {rows}")


    """
    Build pipeline
    """
    # Transform all features into a Vector
    ignore = [] # columns to ignore (str)
    vectors = ['artist_id_vec', 'filename_vec', 'energy', 'similar_artists_vec',
        'danceability', 'segments_start']
    final_assembler = VectorAssembler(
        # inputCols=[x for x in df.columns if x not in ignore], 
        inputCols=[x for x in vectors if x not in ignore], 
        outputCol='features')

    train, test = cleaned_kMeans_df.limit(500).randomSplit([0.8, 0.2])

    # Pipeline
    pipeline = Pipeline(stages=[
        # # Transformers
        # filename_2vec,
        # artist_id_2vec,
        # similar_artists_2vec,
        # # Assemble everything
        # final_assembler,
        VectorAssembler(inputCols=['segments_timbre_vec'], outputCol='features'),
        KMeans(k=20)    #TODO: assume 20 clusters, optimize later
    ])

    model = pipeline.fit(train)
    ### Regression:


if __name__ == '__main__':

    spark = SparkSession.builder.appName('msd-kNN').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')
    #sc = spark.sparkContext

    main()