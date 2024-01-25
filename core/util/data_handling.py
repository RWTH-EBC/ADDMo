import git

def root_dir():
    return git.Repo('.', search_parent_directories=True).working_tree_dir

# class DataHandler():
#
#     def __init__(self, config_gui):
#         self.x = None
#         self.y = None
#         self.x_train = None
#         self.y_train = None
#         self.x_test = None
#         self.y_test = None
#         self.x_val = None
#         self.y_val = None
#
#     def split_train_test(self):
#         self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
#                                                                                 test_size=0.25)



def split_target_features(target_name, xy):
    x = xy.drop(target_name, axis=1)
    y = xy[target_name]
    return (x, y)