from utils.data_handling import (
    data_encoder,
    data_importer,
    data_imputer,
    data_scaler,
    date_time_converter,
)
from NN_model.train import input_data_to_tensor, model_trainer
from NN_model.LR_neuralnet import LinearRegressionforInsurance
import pandas as pd
from NN_model.evaluation import test_data_to_tensor, evaluate_model_performance

data, test_data = data_importer("IR_raw_data")

data = data_imputer(data)

data = data_scaler(data)

date_cols = ["Policy Start Date"]

data = date_time_converter(data, date_cols)

data = data_encoder(data)

data["Policy Age"] = (pd.to_datetime("2025-01-01") - data["Policy Start Date"]).dt.days

data.drop(["Policy Start Date", "id"], axis=1, inplace=True)

X = data.drop(columns=["Premium Amount"], axis=1)
y = data["Premium Amount"]

X_train, y_train, X_test, y_test = input_data_to_tensor(X, y, testsize=0.3)

NNmodel = LinearRegressionforInsurance
epochs = 750
model = model_trainer(
    X_train, y_train, X_test, y_test, model_to_train=NNmodel, LR=1e-3, n_epochs=epochs
)

test_data = data_imputer(test_data)
test_data = data_scaler(test_data)
test_data = date_time_converter(test_data, date_cols)
test_data = data_encoder(test_data)

test_data["Policy Age"] = (
    pd.to_datetime("2025-01-01") - test_data["Policy Start Date"]
).dt.days

test_data.drop(["Policy Start Date", "id"], axis=1, inplace=True)
test_X = test_data.drop(columns=["Premium Amount"], axis=1)
test_y = test_data["Premium Amount"]


test_X, test_y = test_data_to_tensor(test_X, test_y)

results = evaluate_model_performance(
    model=model, X_test=test_X, y_test=test_y, is_log_transformed=True
)

print("Fin")
