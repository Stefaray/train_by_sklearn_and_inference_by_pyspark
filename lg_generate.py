from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

# from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #通过去除均值并缩放到单位方差来标准化特征
from sklearn.linear_model import LogisticRegression




if __name__ == "__main__":
    attr_name = ['Y', 'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag', 'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

    # read in csv as dataframe
    df = pd.read_csv('HIGGS_5000.csv', header=None, names=attr_name)

    label_name = 'Y'
    feature_name = list(df.columns)
    feature_name.remove(label_name)

    X = df[feature_name]
    Y = df['Y']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)  # 测试集为30%

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)  # 必须先fit才能使用标准化，因此我们这里对模型训练并转换
    # X_test = scaler.transform(
    #     X_test)  # 直接使用在模型构建数据上进行一个数据标准化操作，对剩余的数据（testData）使用同样的均值、方差、最大最小值等指标进行转换，transform(testData)，从而保证train、test处理方式相同:


    # 模型训练（线性模型）
    lg = LogisticRegression()
    lg.fit(X_train, Y_train)  # 训练模型
    # y_predict = lr.predict(X_test)  # 预测

    joblib.dump(lg, 'lg_model.pkl')

