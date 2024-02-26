from machine_learning_util.ann import TunerModel, TunerBatchNormalizing, TunerRescaling, TunerDense
from use_cases.train_ann import train_ann

#######################################################################################################################

#######################################################################################################################
# name = 'Carnot_uncertain_short'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     TunerRescaling(scale=12.3, offset=0),
#     name='KerasTuner' + '_' + 'Carnot_short',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 336))

# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Carnot_uncertain_mid'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     TunerRescaling(scale=12.3, offset=0),
#     name='KerasTuner' + '_' + 'Carnot_mid',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 744))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Carnot_uncertain_long'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     TunerRescaling(scale=12.3, offset=0),
#     name='KerasTuner' + '_' + 'Carnot_long',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 1416))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Boptest_Pel_short'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('relu',)),
#     TunerRescaling(scale=2000, offset=0),
#     name='KerasTuner' + '_' + 'Boptest_Pel_short',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 672))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Boptest_Pel_mid'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('relu',)),
#     TunerRescaling(scale=2000, offset=0),
#     name='KerasTuner' + '_' + 'Boptest_Pel_mid',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 1488))
#
# train_ann(name, tuner, training_interval)
# #######################################################################################################################
# name = 'Boptest_Pel_long'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('relu',)),
#     TunerRescaling(scale=2000, offset=0),
#     name='KerasTuner' + '_' + 'Boptest_Pel_long',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 2832))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Boptest_TAir_short'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     name='KerasTuner' + '_' + 'Boptest_TAir_short',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 672))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Boptest_TAir_mid'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     name='KerasTuner' + '_' + 'Boptest_TAir_mid',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 1488))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Boptest_TAir_long'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     name='KerasTuner' + '_' + 'Boptest_TAir_long',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 2832))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Carnot_short'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     TunerRescaling(scale=12.3, offset=0),
#     name='KerasTuner' + '_' + 'Carnot_short_0',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 336))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Carnot_mid'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     TunerRescaling(scale=12.3, offset=0),
#     name='KerasTuner' + '_' + 'Carnot_mid_0',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 744))
#
# train_ann(name, tuner, training_interval)
#######################################################################################################################
# name = 'Carnot_long'
# # Specify Tuner Model
# tuner = TunerModel(
#     TunerBatchNormalizing(),
#     TunerDense(units=(4, 8, 16, 32), activations=('sigmoid',)),
#     TunerRescaling(scale=12.3, offset=0),
#     name='KerasTuner' + '_' + 'Carnot_long_0',
# )
# # Specify data indices used for training, validation and testing
# training_interval = list(range(0, 1416))
#
# train_ann(name, tuner, training_interval)



name = 'Carnot_uncertain_mid_32Neurons'
# Specify Tuner Model
tuner = TunerModel(
    TunerBatchNormalizing(),
    TunerDense(units=(32, 32), activations=('sigmoid',)),
    TunerRescaling(scale=12.3, offset=0),
    name='KerasTuner' + '_' + 'Carnot_mid',
)
# Specify data indices used for training, validation and testing
training_interval = list(range(0, 744))

train_ann(name, tuner, training_interval, n=2)
