# Making the requierd import statements
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

# File that will have all the cross validation resuls
file = open("crossValidationResults.txt", "w")

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
# these will be the parameters that I test with for cross validation
ranks = [10,15,20]
regParams = [0.5,0.1, 0.9]
maxIters = [10,15,20]

#als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

print("---------- CROSS VALIDATION WITH DIFFERENT PARAMS ----------\n\n")
file.write("---------- CROSS VALIDATION WITH DIFFERENT PARAMS ----------\n\n")

# this function will return the tuple of 2 if the ratings are euql, else it will return one
def getTuple(tup):
    if tup[0] == tup[1]:
        return [tup[0], tup[1]]
    return [tup[0]]

iteration = 1
# going through all the parameters
for r in ranks:
    for rp in regParams:
        for m in maxIters:
           
            print("Rank is: " + str(r))
            print("RegParam is :" + str(rp))
            print("MaxIters is: " + str(m))
            
            file.write("Rank is: " + str(r) + "\n")
            file.write("RegParam is :" + str(rp) + "\n")
            file.write("MaxIters is: " + str(m) + "\n")
            # running model with each combination
            als = ALS(maxIter=m, regParam=rp, rank=r, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
            # splitting the data in 80% and 20% wth same seed all the time
            trainTest = ratings.randomSplit([0.8, 0.2], seed = 10)
            trainingDF = trainTest[0]
            testDF = trainTest[1]
            model = als.fit(trainingDF)
            
            # generatig predictions
            predictions = model.transform(testDF).cache()
            # get RMSE
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            
            print("----- Evaluation Metrics -----\n")
            print("Root-mean-square error = " + str(rmse) + "\n")
            
            file.write("Evaluation Metrics: \n")
            file.write("Root-mean-square error = " + str(rmse) + "\n")
            # Get MSE
            evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
            mse = evaluator.evaluate(predictions)
            file.write("Mean-square error = " + str(mse) + "\n")
            print("Mean-square error = " + str(mse) + "\n")
        
           # making dataframe of userid, ground thruth rating, and predicted rating
            pred = predictions.rdd.map(lambda s: Row(userId=int(s.userId), ratingP=float(round(float(s.prediction),1))))
            predDf = spark.createDataFrame(pred)
            trainDf = spark.createDataFrame(train)
            
            print(predDf.show())
            print(trainDf.show())
            # joining to create the one complete df
            mapDf = trainDf.join(predDf, on="userId")
            print(mapDf.show())
            
            print("func")
            allTrainAndPred = mapDf.rdd.map(lambda r: getTuple([r.ratingOne, r.ratingP]))
            print("get count")
            # length of all the ratings predictd
            totalLength = allTrainAndPred.count()
            print(allTrainAndPred.take(5))
            print("filtering")
            # making sure that the rattings that are equal are chosen
            filterRdd = allTrainAndPred.filter(lambda x: len(x) == 2)
            print("another count")
            # lenght of the correct predictions
            lenOfFilter = filterRdd.count()
            # Generating MAP
            MAP = float(lenOfFilter)/float(totalLength)
            print("MAP: " + str(MAP))
            file.write("MAP: " + str(MAP) + "\n\n")
    iteration += 1

spark.stop()






