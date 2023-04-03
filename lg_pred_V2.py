from contextlib import contextmanager
import time

import joblib
import pandas as pd
import xgboost as xgb
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

import pyspark.sql.functions as F

# 获取运行时间
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{}：使用时间为{:.0f}s".format(title, time.time() - t0))

attr_name = ['Y', 'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag', 'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
# sc = SparkContext('local', 'lg predict')
# # sc = SparkContext("yarn", appName='logistic regression')
# sqlContext = SQLContext(sc)

spark = SparkSession \
    .builder \
    .appName("lg predict") \
    .master("local") \
    .getOrCreate() \

sc = spark.sparkContext

# read in csv as dataframe
df = spark.read.csv('HIGGS_5000.csv', inferSchema=True, header=False).toDF(*attr_name)
label_name = 'Y'
feature_name = list(df.columns)
feature_name.remove(label_name)

df_assembler = VectorAssembler(
    inputCols=feature_name,
    outputCol='features')
df = df_assembler.transform(df)

train_df, test_df = df.randomSplit([0.75, 0.25])

model = joblib.load('lg_model.pkl')  # 从网络中获得的二进制流中加载模型
model_sc = sc.broadcast(model)

def predictor(features):
    # y_predprob = model_sc.value.predict_proba([features])[:, 1]
    y_predprob = model_sc.value.predict([features])
    return float(y_predprob[0])

udf_predictor = F.udf(predictor, FloatType())

with timer('sc广播传播预测处理时间'):
    # print( F.col('features') )
    test_result = test_df.withColumn('prediction', udf_predictor( F.col('features') ))
    test_result.count()
    test_result.show()

    # calculate the acc
    tp = test_result[(test_result.Y == 1) & (test_result.prediction == 1)].count()
    tn = test_result[(test_result.Y == 0) & (test_result.prediction == 0)].count()
    fp = test_result[(test_result.Y == 0) & (test_result.prediction == 1)].count()
    fn = test_result[(test_result.Y == 1) & (test_result.prediction == 0)].count()
    # Accuracy
    print('test accuracy is : %f' % ((tp + tn) / (tp + tn + fp + fn)))

    # calculate the recall
    print('test recall is : %f' % (tp / (tp + fn)))
    print('test precision is : %f' % (tp / (tp + fp)))

spark.stop()
