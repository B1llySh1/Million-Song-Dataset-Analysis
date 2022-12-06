import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.feature import VectorAssembler, Word2Vec, MinMaxScaler, Tokenizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from msd_schema import msd_schema

def main(inputs, output):
    """
    Naive training with all the complete data
    """
    df = spark.read.json(inputs, schema=msd_schema)

    # 2D float arrays
    specialArrTypes = ['segments_pitches', 'segments_timbre']

    # Select from the raw DF
    df = df.select(
        'filename',
        functions.explode('segments_timbre').alias('segments_timbre'))

    df = df.filter(
        df['filename'] == 'TRANIJG128F92EFC9B'
    )

    df.show(1)

    df = df.select(
        'filename',
        df.segments_timbre[0], df.segments_timbre[1], df.segments_timbre[2], df.segments_timbre[3],
        df.segments_timbre[4], df.segments_timbre[5], df.segments_timbre[6], df.segments_timbre[7],
        df.segments_timbre[8], df.segments_timbre[9], df.segments_timbre[10], df.segments_timbre[11]
    )

    df.show(1)

    df = df.groupBy('filename').avg()

    df.show(1)

    return
    # # TODO: Convert 2D Float ArrayTypes to Vectors
    # df = df.withColumn(
    #     'segments_timbre_avg', 
    #     functions.transform(        # map each element (https://spark.apache.org/docs/3.2.0/api/python/reference/api/pyspark.sql.functions.transform.html)
    #         'segments_timbre',
    #         lambda x : array_to_vector(x)
    #     )
    # # )
    # # Then array_to_vector again to get 2D Vector<Vector>?
    # # TODO: Error here
    # ).withColumn('segments_timbre_vec2', array_to_vector('segments_timbre_vec'))


    # Drop rows with any empty fields
    df = df.dropna("any") # Note: in 10k subset, only 2210 songs left
    rows = df.count() # should have less than 10k songs now
    print(f"Number of rows left: {rows}")


    """
    Build pipeline
    """

    train, test = df.randomSplit([0.8, 0.2])

    # Pipeline
    ignore = ['filename']
    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=[x for x in df.columns if x not in ignore], outputCol='features'),
        KMeans(k=20)    
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