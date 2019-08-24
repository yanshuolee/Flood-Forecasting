import numpy as np

def generate(data_length, batch_size, shuffle=False):
    row = data_length // batch_size
    row_remain = data_length % batch_size
    col = batch_size
    total_length = row * col
    arr = np.arange(total_length)
    if row_remain:
        arr_adder = np.arange(total_length + row_remain - 1, total_length + row_remain - 1 - batch_size, -1)

    if shuffle:
        arr = arr.reshape((row, col))
        concat = np.append(arr, arr_adder, axis=0)
        np.random.shuffle(concat)
        concat = concat.reshape((row, col))
    else:
        arr = arr.reshape((row, col))
        concat = np.append(arr, [arr_adder], axis=0)

    return concat