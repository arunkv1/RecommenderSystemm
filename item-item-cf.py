# Making the requierd import statements
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import Rating, ALS, Matrix
from pyspark.mllib.evaluation import RegressionMetrics

# setting the conf to be compatible ith as mr, so that it can un freely
print("Setting up spark context")
conf = SparkConf()
conf.setMaster('local[*]')
conf.set('spark.driver.maxResultSize', '15G')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

print("Reading Data")
# reading the data from s3
df = sc.textFile("s3://cs657-bucket/ratings.csv")
# splitting the rows by commas
ratingsSplit = df.map(lambda row: row.split(","))
# getting the datat and making it into row
ratingsMap = ratingsSplit.map(lambda p: Rating(int(p[0]), int(p[1]), float(p[2])))
# making the dataframe
print("splitting datas")
ratings = spark.createDataFrame(ratingsMap).cache()
# these will be the parameters that I test with for cross validation
ranks = [10,15,20]
maxIters = [10,15,20]
# going through all the parameters
for r in ranks:
    for m in maxIters:
        trainTest = ratings.randomSplit([0.8, 0.2])
        trainingDF = trainTest[0]
        testDF = trainTest[1]
        
        train = ratings.rdd.map(lambda x: (x.user, x.rating)).cache()
        test = testDF.rdd.map(lambda r: (r.user, r.product))
    
        # trainImplicit is matrix factorizaation for item-item CF and will run model with each combination
        als = ALS.trainImplicit(trainingDF, r, m)  
        
        # predictAll uses matrix from the train to make predictions (item-item cf)
        prediction = als.predictAll(test).map(lambda s: (s.user, s.rating))
        allTrainAndPred = prediction.join(train).map(lambda res: res[1])
        # computing th metrics for the ouput
        metrics = RegressionMetrics(allTrainAndPred)
        print("RMSE = %s" % metrics.rootMeanSquaredError)
        print("MSE = %s" % metrics.meanSquaredError)

# WHen analyzing the cross validation hrough rint statements, we foundthe rank of 10 and
# maxIters of 20 to be the best. These were also the best when we tested with the ALS. So
# when generating the final results, we will use these parameters since they gave the best
# output
        
spark.stop()


































