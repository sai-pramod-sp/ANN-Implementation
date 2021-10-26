import tensorflow as tf
import os
import numpy as np
import time


def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":","_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name


def get_callbacks(config, x_train):
    logs = config["logs"]
    unique_dir_name = get_timestamp("tb_logs")
    tensorboard_root_log = os.path.join(logs["logs_dir"], logs["tensorboard_root_log_dir"], unique_dir_name)

    os.makedirs(tensorboard_root_log, exist_ok = True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_root_log)

    file_writer = tf.summary.create_file_writer(logdir=tensorboard_root_log)

    with file_writer.as_default():
        images = np.reshape(x_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handritten digit samples", images, max_outputs=25,step=0)

    
    params = config["params"]
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        patience=params["patience"], 
        restore_best_weights=params["restore_best_weights"])
    
    artifacts = config["artifacts"]
    
    ckpt_dir = os.path.join(artifacts["artifacts_dir"], artifacts["checkpoint_dir"])
    os.makedirs(tensorboard_root_log, exist_ok = True)
    ckpt_path = os.path.join(ckpt_dir, "model_ckpt.h5")


    check_pointing_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
    
    return [tensorboard_cb, early_stop_cb, check_pointing_cb]