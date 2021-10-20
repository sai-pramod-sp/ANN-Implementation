import os
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
import argparse

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = read_config["params"]["loss_function"]
    OPTIMIZER = read_config["params"]["optimizer"]
    METRICS = read_config["params"]["metrics"]
    no_classes = read_config["params"]["no_classes"]
    
    model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,no_classes)
    EPOCHS = read_config["params"]["epochs"]
    VALIDATION = (x_valid, y_valid)

    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)
