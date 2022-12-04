import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


from msd_schema import msd_schema
# add more functions as necessary


def main(inputs, output):
    """
    A Linear Regression model of 'year' vs 'loudness'
    """
    df = spark.read.json(inputs, schema=msd_schema)

    df = df.select(
        'year',
        'loudness'
    )

    df = df.filter(
        # df['artist_familiarity'] # filter out low familiarity artists?
        df['year'] > 0,
    ).sort('loudness')

    df.show()
    rows = df.count()

    print(f"Number of rows: {rows}")

    # Transform features into Vectors
    assembler = VectorAssembler(inputCols=['year'], outputCol='feature')

    train, test = df.randomSplit([0.8, 0.2])

    train = assembler.transform(train)
    test = assembler.transform(test)


    lr = LinearRegression().setLabelCol('loudness').setFeaturesCol('feature')
    
    lrModel = lr.fit(train)

    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))

    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    testResult = lrModel.transform(test)
    testResult.show()

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    
    spark = SparkSession.builder.appName('msd-kNN').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')
    #sc = spark.sparkContext

    main(inputs, output)