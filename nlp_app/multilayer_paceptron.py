import numpy as np


def f_1(z):
    return np.maximum(z, 0)


def layer_1(x):
    w_1 = np.array([
        [-0.423, -0.795, 1.223],
        [1.402, 0.885, 0.774]
    ])
    b_1 = np.array([0.546, 0.774])

    z_1 = b_1 + np.dot(w_1, x)
    output_1 = f_1(z_1)
    return output_1


x = np.array([0.2, 0.4, -0.1])
out_1 = layer_1(x)


def f_2(z):
    return 1.0/(1.0 + np.exp(-z))


def layer_2(x):
    w_2 = np.array([
        [1.567, -1.645],
    ])

    b_2 = np.array([0.255])
    z_2 = b_2 + np.dot(w_2, x)
    output_2 = f_2(z_2)
    return output_2


out_2 = layer_2(out_1)
print(out_2)
