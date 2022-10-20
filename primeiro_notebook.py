# Databricks notebook source
# DBTITLE 1,Database
sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")

df = sdf.toPandas()

# COMMAND ----------

# DBTITLE 1,Variáveis
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

import mlflow

target_col = "radiant_win"
id_col = "match_id"

features_col = list(set(df.columns.tolist()))
set([target_col, id_col])

y = df[target_col]
X = df[features_col]

print(y)
print(X)

# COMMAND ----------

# DBTITLE 1,Treinamento
"""
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Tamanho X treino: ", X_train.shape[0], ", ", "Tamanho Y treino: ", y_train.shape[0], " | ", "Tamanho X teste: ", X_test.shape[0], ", ", "Tamanho Y teste: ", y_test.shape[0])

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# modelo de tree decision
y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)
acc_train = metrics.accuracy_score(y_train, y_train_pred)
print("Precisão do treinamento:", acc_train)

y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)
acc_train = metrics.accuracy_score(y_test, y_test_pred)
print("Precisão do treinamento:", acc_train)
"""

# COMMAND ----------

# DBTITLE 1,Split de teste e treino.

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Tamanho X treino: ", X_train.shape[0], ", ", "Tamanho Y treino: ", y_train.shape[0], " | ", "Tamanho X teste: ", X_test.shape[0], ", ", "Tamanho Y teste: ", y_test.shape[0])

# COMMAND ----------

# DBTITLE 1,Setup do experimento.
mlflow.set_experiment("/Users/juan.c.silva@unesp.br/dota-unesp-juan")

# COMMAND ----------

# DBTITLE 1,Execução do experimento.
with mlflow.start_run():
    mlflow.sklearn.autolog()
    #model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.7, random_state=42)
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    # Treino
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print("Precisão do treinamento:", acc_train)
    # Teste
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print("Precisão do teste:", acc_test)


