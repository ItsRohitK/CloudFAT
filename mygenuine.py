import sagemaker
import boto3
import pandas as pd
import os
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Setup
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = "rohit"
region = boto3.Session().region_name

print("--- STEP 1: Preprocessing Data (Wine Quality) ---")
# Reading your uploaded wine dataset from S3
df = pd.read_csv(f"s3://{bucket}/winequality-red.csv")

# Separate features and target (Target is 'quality' in this dataset)
X = df.drop("quality", axis=1)
y = df["quality"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SageMaker's built-in XGBoost requires the target variable to be the FIRST column
train_data = pd.concat([y_train.reset_index(drop=True), pd.DataFrame(X_train_scaled)], axis=1)
test_data = pd.concat([y_test.reset_index(drop=True), pd.DataFrame(X_test_scaled)], axis=1)

# Save locally to the notebook's hard drive temporarily
train_data.to_csv("train.csv", header=False, index=False)
test_data.to_csv("test.csv", header=False, index=False)


print("\n--- STEP 2: Uploading Processed Data to S3 ---")
train_s3_path = sess.upload_data(path="train.csv", bucket=bucket, key_prefix="processed-data/train")
test_s3_path = sess.upload_data(path="test.csv", bucket=bucket, key_prefix="processed-data/test")
print(f"Training data uploaded to: {train_s3_path}")


print("\n--- STEP 3: Training the Model ---")
xgboost_container = image_uris.retrieve("xgboost", region, "1.5-1")

xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=xgboost_container,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/model-output",
    sagemaker_session=sess
)

# CHANGED: 'objective' is now 'reg:squarederror' to predict the quality score (3-8)
xgb_estimator.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    objective="reg:squarederror", 
    num_round=100
)

train_input = TrainingInput(train_s3_path, content_type="csv")

print("Starting Training Job... Please wait.")
xgb_estimator.fit({"train": train_input})


print("\n--- STEP 4: Deploying the Model ---")
print("Deploying endpoint... This takes about 5 minutes.")
xgb_predictor = xgb_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print(f"\nSUCCESS! Model deployed at: {xgb_predictor.endpoint_name}")
