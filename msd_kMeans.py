import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.feature import VectorAssembler, Word2Vec, MinMaxScaler, Tokenizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql.types import ArrayType, FloatType, StringType

import numpy as np

from msd_schema import msd_schema

def main(inputs, output):
    """
    Naive training with all the complete data
    """
    df = spark.read.json(inputs, schema=msd_schema)

    # columns to be selected for transformation
    stringTypes = [ 
        'filename',
        'artist_id', 
        'artist_name',
        # 'artist_location', 
        'title'
    ]
    """ Transformers for string type columns """
    artist_id_tf = Pipeline(stages=[
        Tokenizer(inputCol='artist_id', outputCol='artist_id_words'),
        Word2Vec(inputCol='artist_id_words', outputCol='artist_id_vec', minCount=1),
        MinMaxScaler(inputCol='artist_id_vec', outputCol='artist_id_scaled')
    ])
    artist_name_tf = Pipeline(stages=[
        Tokenizer(inputCol='artist_name', outputCol='artist_name_words'),
        Word2Vec(inputCol='artist_name_words', outputCol='artist_name_vec', minCount=1),
        MinMaxScaler(inputCol='artist_name_vec', outputCol='artist_name_scaled')
    ])
    title_tf = Pipeline(stages=[
        Tokenizer(inputCol='title', outputCol='title_words'),
        Word2Vec(inputCol='title_words', outputCol='title_vec', minCount=1),
        MinMaxScaler(inputCol='title_vec', outputCol='title_scaled')
    ])

    stringTypes_cols = ['artist_id_scaled', 'artist_name_scaled', 'title_scaled']

    ######################################################################################

    stringArrTypes = [         
        # 'artist_terms', # TODO: Decide if want to continue with using terms/tags
        'similar_artists',
    ]

    """ Transformers for string array type columns """
    similar_artists_tf = Pipeline(stages=[
        Word2Vec(inputCol='similar_artists', outputCol='similar_artists_vec'),
        MinMaxScaler(inputCol='similar_artists_vec', outputCol='similar_artists_scaled')
    ])

    stringArrTypes_cols = ['similar_artists_scaled']

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
    """ Transformers for float/int type columns """
    artist_familiarity_tf = Pipeline(stages=[
        VectorAssembler(inputCols=['artist_familiarity'], outputCol='artist_familiarity_vec'),
        MinMaxScaler(inputCol='artist_familiarity_vec', outputCol='artist_familiarity_scaled')
    ])
    artist_hotttnesss_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['artist_hotttnesss'], outputCol='artist_hotttnesss_vec'),
        MinMaxScaler(inputCol='artist_hotttnesss_vec', outputCol='artist_hotttnesss_scaled')
    ])
    # artist_latitude_tf = MinMaxScaler(inputCol='artist_latitude', outputCol='artist_latitude_scaled')
    # artist_longitude_tf = MinMaxScaler(inputCol='artist_longitude', outputCol='artist_longitude_scaled')
    danceability_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['danceability'], outputCol='danceability_vec'),
        MinMaxScaler(inputCol='danceability_vec', outputCol='danceability_scaled')
    ])
    energy_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['energy'], outputCol='energy_vec'),
        MinMaxScaler(inputCol='energy_vec', outputCol='energy_scaled')
    ])
    end_of_fade_in_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['end_of_fade_in'], outputCol='end_of_fade_in_vec'),
        MinMaxScaler(inputCol='end_of_fade_in_vec', outputCol='end_of_fade_in_scaled')
    ])
    start_of_fade_out_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['start_of_fade_out'], outputCol='start_of_fade_out_vec'),
        MinMaxScaler(inputCol='start_of_fade_out_vec', outputCol='start_of_fade_out_scaled')
    ])
    key_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['key'], outputCol='key_vec'),
        MinMaxScaler(inputCol='key_vec', outputCol='key_scaled')
    ])
    key_confidence_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['key_confidence'], outputCol='key_confidence_vec'),
        MinMaxScaler(inputCol='key_confidence_vec', outputCol='key_confidence_scaled')
    ])
    duration_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['duration'], outputCol='duration_vec'),
        MinMaxScaler(inputCol='duration_vec', outputCol='duration_scaled')
    ])
    loudness_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['loudness'], outputCol='loudness_vec'),
        MinMaxScaler(inputCol='loudness_vec', outputCol='loudness_scaled')
    ])
    tempo_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['tempo'], outputCol='tempo_vec'),
        MinMaxScaler(inputCol='tempo_vec', outputCol='tempo_scaled')
    ])
    mode_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['mode'], outputCol='mode_vec'),
        MinMaxScaler(inputCol='mode_vec', outputCol='mode_scaled')
    ])
    time_signature_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['time_signature'], outputCol='time_signature_vec'),
        MinMaxScaler(inputCol='time_signature_vec', outputCol='time_signature_scaled')
    ])
    year_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['year'], outputCol='year_vec'),
        MinMaxScaler(inputCol='year_vec', outputCol='year_scaled')
    ])
    

    floatTypes_cols = ['artist_familiarity_scaled', 'artist_hotttnesss_scaled', 'danceability_scaled', 'energy_scaled',
        'end_of_fade_in_scaled', 'start_of_fade_out_scaled', 'key_scaled', 'key_confidence_scaled', 'duration_scaled',
        'loudness_scaled', 'tempo_scaled', 'mode_scaled', 'time_signature_scaled', 'year_scaled']

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



    # 2D float arrays
    specialArrTypes = ['segments_pitches', 'segments_timbre']

    # Select from the raw DF
    feature_cols = stringTypes + stringArrTypes + floatTypes + floatArrTypes + specialArrTypes

    kMeans_df = df.limit(1000).select(feature_cols)

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
    two_d_df = two_d_df.groupBy('filename').avg()


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
    two_d_df_2 = two_d_df_2.groupBy('filename').avg()

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
            array_std('tatums_start').alias('tatums_start_std'),)
    segments_start_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_start_mean'], outputCol='segments_start_mean_vec'),
        MinMaxScaler(inputCol='segments_start_mean_vec', outputCol='segments_start_mean_scaled')
    ])
    segments_loudness_max_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_loudness_max_mean'], outputCol='segments_loudness_max_mean_vec'),
        MinMaxScaler(inputCol='segments_loudness_max_mean_vec', outputCol='segments_loudness_max_mean_scaled')
    ])
    segments_loudness_max_time_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_loudness_max_time_mean'], outputCol='segments_loudness_max_time_mean_vec'),
        MinMaxScaler(inputCol='segments_loudness_max_time_mean_vec', outputCol='segments_loudness_max_time_mean_scaled')
    ])
    segments_loudness_start_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_loudness_start_mean'], outputCol='segments_loudness_start_mean_vec'),
        MinMaxScaler(inputCol='segments_loudness_start_mean_vec', outputCol='segments_loudness_start_mean_scaled')
    ])
    sections_start_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['sections_start_mean'], outputCol='sections_start_mean_vec'),
        MinMaxScaler(inputCol='sections_start_mean_vec', outputCol='sections_start_mean_scaled')
    ])
    beats_start_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['beats_start_mean'], outputCol='beats_start_mean_vec'),
        MinMaxScaler(inputCol='beats_start_mean_vec', outputCol='beats_start_mean_scaled')
    ])
    bars_start_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['bars_start_mean'], outputCol='bars_start_mean_vec'),
        MinMaxScaler(inputCol='bars_start_mean_vec', outputCol='bars_start_mean_scaled')
    ])
    tatums_start_mean_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['tatums_start_mean'], outputCol='tatums_start_mean_vec'),
        MinMaxScaler(inputCol='tatums_start_mean_vec', outputCol='tatums_start_mean_scaled')
    ])
    segments_start_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_start_std'], outputCol='segments_start_std_vec'),
        MinMaxScaler(inputCol='segments_start_std_vec', outputCol='segments_start_std_scaled')
    ])
    segments_loudness_max_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_loudness_max_std'], outputCol='segments_loudness_max_std_vec'),
        MinMaxScaler(inputCol='segments_loudness_max_std_vec', outputCol='segments_loudness_max_std_scaled')
    ])
    segments_loudness_max_time_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_loudness_max_time_std'], outputCol='segments_loudness_max_time_std_vec'),
        MinMaxScaler(inputCol='segments_loudness_max_time_std_vec', outputCol='segments_loudness_max_time_std_scaled')
    ])
    segments_loudness_start_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['segments_loudness_start_std'], outputCol='segments_loudness_start_std_vec'),
        MinMaxScaler(inputCol='segments_loudness_start_std_vec', outputCol='segments_loudness_start_std_scaled')
    ])
    sections_start_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['sections_start_std'], outputCol='sections_start_std_vec'),
        MinMaxScaler(inputCol='sections_start_std_vec', outputCol='sections_start_std_scaled')
    ])
    beats_start_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['beats_start_std'], outputCol='beats_start_std_vec'),
        MinMaxScaler(inputCol='beats_start_std_vec', outputCol='beats_start_std_scaled')
    ])
    bars_start_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['bars_start_std'], outputCol='bars_start_std_vec'),
        MinMaxScaler(inputCol='bars_start_std_vec', outputCol='bars_start_std_scaled')
    ])
    tatums_start_std_tf = Pipeline(stages=[ 
        VectorAssembler(inputCols=['tatums_start_std'], outputCol='tatums_start_std_vec'),
        MinMaxScaler(inputCol='tatums_start_std_vec', outputCol='tatums_start_std_scaled')
    ])


    one_d_df = two_d_df.join(two_d_df_2,on='filename')
    kMeans_df = kMeans_df.join(one_d_df, on='filename')
    kMeans_df = kMeans_df.join(mean_df, on='filename')

    # Drop rows with any empty fields
    final_df = kMeans_df.dropna("any") # Note: in 10k subset, only 2210 songs left
    # final_df.show()
    rows = final_df.count() # should have less than 10k songs now
    print(f"Number of rows left: {rows}")

    """
    Build pipeline
    """
    # Transform all features into a Vector
    # ignore = [] # columns to ignore (str)
    # scaled_features = stringTypes_cols + stringArrTypes_cols + floatTypes_cols
    stringTypes_ignore = ['artist_id_words', 'artist_id_vec', 
        'artist_name_words', 'artist_name_vec',
        'title_words', 'title_vec']
    stringArrTypes_ignore = ['similar_artists_vec']
    floatTypes_ignore = ['artist_familiarity_vec', 'artist_hotttnesss_vec', 'danceability_vec', 'energy_vec',
        'end_of_fade_in_vec', 'start_of_fade_out_vec', 'key_vec', 'key_confidence_vec', 'duration_vec',
        'loudness_vec', 'tempo_vec', 'mode_vec', 'time_signature_vec', 'year_vec'
        ]
    origin_array_ignore = ['segments_start_mean_vec', 'segments_loudness_max_mean_vec','segments_loudness_max_time_mean_vec','segments_loudness_start_mean_vec','sections_start_mean_vec','beats_start_mean_vec','bars_start_mean_vec','tatums_start_mean_vec','segments_start_std_vec','segments_loudness_max_std_vec','segments_loudness_max_time_std_vec','segments_loudness_start_std_vec','sections_start_std_vec','beats_start_std_vec','bars_start_std_vec','tatums_start_std_vec']
    ignore = stringTypes_ignore+stringArrTypes_ignore+floatTypes_ignore+feature_cols+origin_array_ignore # columns to ignore (str)
    scaled_features = [name for name in final_df.schema.names if name not in ignore]
    final_assembler = VectorAssembler(
        # inputCols=[x for x in df.columns if x not in ignore], 
        inputCols=scaled_features, 
        outputCol='features')

    train, test = final_df.randomSplit([0.8, 0.2])

    # Pipeline
    pipeline = Pipeline(stages=[
        # Transformers
        artist_id_tf,
        artist_name_tf,
        title_tf,
        similar_artists_tf,
        artist_familiarity_tf,
        artist_hotttnesss_tf,
        danceability_tf,
        energy_tf,
        end_of_fade_in_tf,
        start_of_fade_out_tf,
        key_tf,
        key_confidence_tf,
        duration_tf,
        loudness_tf,
        tempo_tf,
        mode_tf,
        time_signature_tf,
        year_tf,
        ##
        segments_start_mean_tf,
        segments_loudness_max_mean_tf,
        segments_loudness_max_time_mean_tf,
        segments_loudness_start_mean_tf,
        sections_start_mean_tf,
        beats_start_mean_tf,
        bars_start_mean_tf,
        tatums_start_mean_tf,
        segments_start_std_tf,
        segments_loudness_max_std_tf,
        segments_loudness_max_time_std_tf,
        segments_loudness_start_std_tf,
        sections_start_std_tf,
        beats_start_std_tf,
        bars_start_std_tf,
        tatums_start_std_tf,
        # # Assemble everything
        final_assembler,
        # VectorAssembler(inputCols=['segments_timbre_vec'], outputCol='features'),
        KMeans(featuresCol='features', k=20)    #TODO: assume 20 clusters, optimize later
    ])

    model = pipeline.fit(train)

    # Summarize the model over the training set and print out some metrics

    # Make predictions
    predictions = model.transform(test)
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    
    spark = SparkSession.builder.appName('msd-kNN').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')
    #sc = spark.sparkContext

    main(inputs, output)