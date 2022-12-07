import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline

from pyspark.sql.types import ArrayType, FloatType, StringType

import numpy as np

from msd_schema import msd_schema


spark = SparkSession.builder.appName('msd-kMeans-ETL').getOrCreate()
assert spark.version >= '3.2' # make sure we have Spark 3.2+
spark.sparkContext.setLogLevel('WARN')

# columns to be selected for transformation
stringTypes = [ 
    'filename',
    # 'artist_id', 
    # 'artist_name',
    # 'artist_location', 
    # 'title'
]


######################################################################################

stringArrTypes = [         
    # 'artist_terms', # TODO: Decide if want to continue with using terms/tags
    # 'similar_artists',
]


######################################################################################

# also include Int types
floatTypes = [
    'artist_familiarity', # use to filter out low familiarity artists?
    'artist_hotttnesss',
    # 'artist_latitude', 'artist_longitude',    # dropped because too many N/A
    # 'song_hotttnesss', # dropped because too many N/A
    'danceability', 'energy',
    'end_of_fade_in', 'start_of_fade_out',
    'key', 'key_confidence',
    'duration', 
    'loudness', #TODO: shift +60dB (since -60dB is the zero in raw data)
    'tempo', 
    'mode', 'mode_confidence',
    'time_signature', 'time_signature_confidence',  # beats per min
    'year', #TODO: shift to delta since min year
]


######################################################################################
floatArrTypes = [
    # 'artist_terms_freq', 'artist_terms_weight', #TODO: VectorAssemble with 'artist_terms'
    'segments_start', #'segments_confidence',
    'segments_loudness_max', 'segments_loudness_max_time', 'segments_loudness_start',
    'sections_start', #'sections_confidence',
    'beats_start', #'beats_confidence',
    'bars_start', #'bars_confidence',
    'tatums_start', #'tatums_confidence'
]


######################################################################################


# 2D float arrays
specialArrTypes = ['segments_pitches', 'segments_timbre']


######################################################################################

# Select from the raw DF
feature_cols = stringTypes + stringArrTypes + floatTypes + floatArrTypes + specialArrTypes



def main(inputs, output):
    """
    Naive training with all the complete data
    """

    df = spark.read.json(inputs, schema=msd_schema)
    
    # TODO: caching here seems to break test on local machine, but caching fixes 0 row count error on the cluster
    kMeans_df = df.select(feature_cols)#.cache()     # TODO: remove the limit, maybe cache? 

    """
    Transform 2D arrays to meaningful statistics
    """
    # Mean segments_timbre by each column
    two_d_df = kMeans_df.select(
            'filename',
            functions.explode('segments_timbre').alias('segments_timbre'))
    two_d_df = two_d_df.select(
            'filename',
            two_d_df.segments_timbre[0], two_d_df.segments_timbre[1], two_d_df.segments_timbre[2], two_d_df.segments_timbre[3],
            two_d_df.segments_timbre[4], two_d_df.segments_timbre[5], two_d_df.segments_timbre[6], two_d_df.segments_timbre[7],
            two_d_df.segments_timbre[8], two_d_df.segments_timbre[9], two_d_df.segments_timbre[10], two_d_df.segments_timbre[11]
        )
    two_d_df = two_d_df.groupBy('filename').avg().cache()

    two_d_df.show(1)

    # Mean segments_pitches by each column
    two_d_df_2 = kMeans_df.select(
            'filename',
            functions.explode('segments_pitches').alias('segments_pitches'))
    two_d_df_2 = two_d_df_2.select(
            'filename',
            two_d_df_2.segments_pitches[0], two_d_df_2.segments_pitches[1], two_d_df_2.segments_pitches[2], two_d_df_2.segments_pitches[3],
            two_d_df_2.segments_pitches[4], two_d_df_2.segments_pitches[5], two_d_df_2.segments_pitches[6], two_d_df_2.segments_pitches[7],
            two_d_df_2.segments_pitches[8], two_d_df_2.segments_pitches[9], two_d_df_2.segments_pitches[10], two_d_df_2.segments_pitches[11],
        )
    two_d_df_2 = two_d_df_2.groupBy('filename').avg().cache()

    two_d_df_2.show(1)

    """
    Transform 1D Arrays to meaningful statistics
    """
    # Mean and std all the array elements
    array_mean = functions.udf(lambda x: float(np.mean(x)), FloatType())
    array_std = functions.udf(lambda x: float(np.std(x)), FloatType())

    mean_df = kMeans_df.select(
            'filename',
            array_mean('segments_start').alias('segments_start_mean'),
            array_mean('segments_loudness_max').alias('segments_loudness_max_mean'),
            array_mean('segments_loudness_max_time').alias('segments_loudness_max_time_mean'),
            array_mean('segments_loudness_start').alias('segments_loudness_start_mean'),
            array_mean('sections_start').alias('sections_start_mean'),
            array_mean('beats_start').alias('beats_start_mean'),
            array_mean('bars_start').alias('bars_start_mean'),
            array_mean('tatums_start').alias('tatums_start_mean'),

            array_std('segments_start').alias('segments_start_std'),
            array_std('segments_loudness_max').alias('segments_loudness_max_std'),
            array_std('segments_loudness_max_time').alias('segments_loudness_max_time_std'),
            array_std('segments_loudness_start').alias('segments_loudness_start_std'),
            array_std('sections_start').alias('sections_start_std'),
            array_std('beats_start').alias('beats_start_std'),
            array_std('bars_start').alias('bars_start_std'),
            array_std('tatums_start').alias('tatums_start_std')
        ).cache()

    # mean_df.show(1)
    # Drop the untransformed, original 1D and 2D array columns to make the DF smaller
    kMeans_df = kMeans_df.drop(
        # 1D original data
        'segments_start', 'segments_loudness_max', 'segments_loudness_max_time', 'segments_loudness_start',
        'sections_start', 'beats_start', 'bars_start', 'tatums_start', 
        # 2D orginal data
        'segments_timbre', 'segments_pitches')    

    kMeans_df = kMeans_df.join(two_d_df, on='filename')
    two_d_df.unpersist()
    kMeans_df = kMeans_df.join(two_d_df_2, on='filename')
    two_d_df_2.unpersist()
    kMeans_df = kMeans_df.join(mean_df, on='filename')
    mean_df.unpersist()

    # Drop rows with any empty fields
    final_df = kMeans_df.dropna("any").cache() # Note: in 10k subset, only 2210 songs left
    # final_df.show(1)
    rows = final_df.count() # should have less than 10k songs now
    print(f"Number of rows left: {rows}")


    train, test = final_df.randomSplit([0.8, 0.2], seed=123)

    output_train = output + "/trainSet"
    output_test = output + "/testSet"

    train.write.format("csv").option("header", "true").save(output_train, mode='overwrite')
    test.write.format("csv").option("header", "true").save(output_test, mode='overwrite')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    

    #sc = spark.sparkContext

    main(inputs, output)