# importing required packages and libraries
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
import math
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# setting the conf to be compatible ith as mr, so that it can un freely
print("Setting up spark context")
conf = SparkConf()
conf.setMaster('local[*]')
conf.set('spark.driver.maxResultSize', '15G')
conf.set('spark.sql.shuffle.partitions',300)

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

#preprocessing the data
print("Reading Data")
# reading the data from s3
df = sc.textFile("s3://cs657-bucket/ratings.csv")
# splitting the rows by commas
ratingsSplit = df.map(lambda row: row.split(","))
# getting the datat and making it into row
ratingsMap = ratingsSplit.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),\
                                  rating=float(p[2]), timestamp=int(p[3])))

print("splitting datas")
# making the dataframe
ratings = spark.createDataFrame(ratingsMap)
# this will have the ground truth rating for each user
train = ratings.rdd.map(lambda x: (Row(userId=int(x.userId), ratingOne=float(x.rating)))).cache()
print(train.take(1))
# training ALS, with the best parameters from the cross validataion
als = ALS(maxIter=10, regParam=0.1, rank=20, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
trainTest = ratings.randomSplit([0.8, 0.2], seed = 10)
trainingDF = trainTest[0]
testDF = trainTest[1]
model = als.fit(trainingDF)
# generatig predictions
predictions = model.transform(testDF).cache()
# making dataframe of userid, ground thruth rating, and predicted rating
pred = predictions.rdd.map(lambda s: Row(userId=int(s.userId), ratingP=float(round(float(s.prediction),1))))
predDf = spark.createDataFrame(pred)
trainDf = spark.createDataFrame(train)
# joining to create the one complete df
mapDf = trainDf.join(predDf, on="userId")
# formatting the data to b in results
allTrainAndPred = mapDf.rdd.map(lambda r: (r.userId, (r.ratingOne, r.ratingP)))
# saving the results to the bucket in s3
saving = allTrainAndPred.saveAsTextFile("s3://cs657-bucket/results")

spark.stop()






