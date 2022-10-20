# Databricks notebook source
import mlflow

model = mlflow.sklearn.load_model("dbfs:/databricks/mlflow-tracking/920696605047689/8ccc40bc50cb47f0abbcf75035b8cb9e/artifacts/model")

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match_new")
df = sdf.toPandas()

# COMMAND ----------

features = set(df.columns.tolist())
set(["match_id", "radiant_win"])

X = df[features]

# COMMAND ----------

score = model.predict_proba(X)

df["proba_radian_win"] = score[:, 1]
df[["match_id", "radiant_win", "proba_radian_win"]]
