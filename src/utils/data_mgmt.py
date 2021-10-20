import tensorflow as tf

def get_data(validation_datasize):
    mnist = tf.keras.datasets.mnist
    (x_train_full,y_train_full),(x_test,y_test) = mnist.load_data()

    ## create a validation dataset from training data
    ## scaling the data between 0 to 1 by dividing by 255 beacuse every pixel has range betweeen 0 to 255
    x_valid,x_train = x_train_full[:validation_datasize]/255. , x_train_full[validation_datasize:]/255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    # scaling the test data as well
    x_test = x_test / 255.
    return (x_train,y_train),(x_valid,y_valid),(x_test,y_test)
