from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_


def input_data_to_tensor(X, y, testsize=0.3):
    """Takes the Variables, divides them into training and testing set in given ratio and then converts the training and test splits into tensors
        then log transforms the given tensors to account of possible skewness in the response variable

    Args:
        X (dataframe): features
        y (dataframe): target variable
        ratio (int): size of the testing split (between 0 and 1)

    Returns:
        X_train, y_train, X_test, y_test: train and test splits of the feature and target variable
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testsize, random_state=13
    )

    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)

    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

    y_train = torch.log(y_train + 1)
    y_test = torch.log(y_test + 1)
    print("Conversion to tensors complete")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train, y_train, X_test, y_test


def model_trainer(X_train, y_train, X_test, y_test, model_to_train, LR, n_epochs=500):
    """Trains the given model over given epochs

    Args:
        X_train (tensor): X train
        y_train (tensor): y train
        X_test (tensor): X test
        y_test (tensor): y test
        model_to_train (torch.nn): neural network to be trained
        LR (number): learning rate to be used while training
        n_epochs(integer): number of training epochs

    Returns:
        _type_: _description_
    """
    model = model_to_train(X_train.shape[1])

    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=LR)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=125, gamma=0.5
    )

    epochs = n_epochs
    max_grad_norm = 1.0

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        with torch.no_grad():
            model.eval()
            y_test_pred = model(X_test)
            test_loss = loss_function(y_test_pred, y_test)
        if epoch % 25 == 0:
            print(
                f"EPOCH : {epoch + 1},  TRAINLOSS: {loss.item():.4f} , TESTLOSS: {test_loss.item():.4f}"
            )
    return model
