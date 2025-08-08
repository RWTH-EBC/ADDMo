def split_target_features(target_name, xy):
    x = xy.drop(target_name, axis=1)
    y = xy[target_name]
    return (x, y)