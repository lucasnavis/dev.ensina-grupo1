def triangular (x, a, b, c):
    return np.maximum(
        np.minimum((x - a) / (b - a), (c - x) / (c - b)),
        0
    )