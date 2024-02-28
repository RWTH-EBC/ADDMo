from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import time

import warnings

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scipy.io import loadmat

from pyod.utils.utility import precision_n_scores

from extrapolation_detection.machine_learning_util.util import rearrange_training_data
from extrapolation_detection.detector import scoring
from extrapolation_detection.detector.detector_tuners import Hyper_KNN, Hyper_OCSVM, Hyper_KDE, Hyper_GP, Hyper_IF, Hyper_ABOD, Hyper_LOF, \
    Hyper_MCD, Hyper_GMM, Hyper_HBOS, Hyper_ECOD, Hyper_DSVDD, Hyper_RNN, Hyper_PCA, Hyper_FB_KNN

warnings.filterwarnings("ignore")

###################################################################################################################
# TODO define parameters here

# TODO Define benchmark tests
mat_file_list = [
    'ionosphere.mat',
    'wbc.mat',
    'pima.mat',
    'vowels.mat',
    'letter.mat',
    'cardio.mat',
    'satimage-2.mat',
    'satellite.mat',
    'pendigits.mat',
    'shuttle.mat',
    'cover.mat',
    'http.mat',
    # 'breastw.mat',
    # 'vertebral.mat'
]

# TODO Define number of iterations
n_ite = 10

# TODO Define classifiers
clf_names = [
    'KNN',
    'ABOD',
    'LOF',
    'MCD',
    'GMM',
    'HBOS',
    'ECOD',
    'GPR'
    'KDE',
    'OCSVM',
    'DSVDD',
    'RNN',
    'PCA',
    'FB',
    'IF',
]

# TODO Define score options
beta = 1
score = 'fbeta'

###################################################################################################################

# Get number of classifiers
n_classifiers = len(clf_names)

# Define result dataframes
time_train_df = None
time_test_df = None
roc_df = None
prn_df = None
hp_df = None
fscore_df = None

# Loop through all benchmark tests
for j in range(len(mat_file_list)):

    # Prepare result dataframe header
    df_columns = ['Data', '#Samples', 'Dim', 'Outlier Perc']
    hp_df_columns = df_columns.copy()
    for clf_name in clf_names:
        df_columns.append(clf_name)

    # Load data
    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join('data', mat_file))

    # Get data from benchmark test
    X = mat['X']
    y = mat['y'].ravel()
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # Save benchmark test statistics
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_train_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_test_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    hp_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    fscore_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    # Create empty results matrices
    roc_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
    time_train_mat = np.zeros([n_ite, n_classifiers])
    time_test_mat = np.zeros([n_ite, n_classifiers])
    fscore_mat = np.zeros([n_ite, n_classifiers])

    # Loop through iterations
    for i in range(n_ite):
        print("\n... Processing", mat_file, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # Split in training 80%, validation 10% and test 10% split
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=1 / 10, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 9, random_state=random_state)
        X_train, X_val, y_train, y_val = rearrange_training_data(X_train, X_val, y_train, y_val)

        # Hyperparameter optimization
        classifiers = dict()
        for clf_name in clf_names:

            # K nearest neighbors
            if clf_name == 'KNN':
                hyper_KNN = Hyper_KNN(outliers_fraction, score_name=score, beta=beta)
                knn_clf, threshold = hyper_KNN.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('KNN_n_neighbors_' + str(i))
                hp_list.append(knn_clf.n_neighbors)
                hp_df_columns.append('KNN_p_' + str(i))
                hp_list.append(knn_clf.p)
                hp_df_columns.append('KNN_method_' + str(i))
                hp_list.append(knn_clf.method)
                hp_df_columns.append('KNN_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['KNN'] = knn_clf

            # Angle based outlier detection
            if clf_name == 'ABOD':
                hyper_ABOD = Hyper_ABOD(outliers_fraction, score_name=score, beta=beta)
                abod_clf, threshold = hyper_ABOD.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('ABOD_n_neighbors_' + str(i))
                hp_list.append(abod_clf.n_neigbors)
                hp_df_columns.append('ABOD_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['ABOD'] = abod_clf

            # Local Outlier Factor
            if clf_name == 'LOF':
                hyper_LOF = Hyper_LOF(outliers_fraction, score_name=score, beta=beta)
                lof_clf, threshold = hyper_LOF.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('LOF_n_neighbors_' + str(i))
                hp_list.append(lof_clf.n_neighbors)
                hp_df_columns.append('LOF_p_' + str(i))
                hp_list.append(lof_clf.p)
                hp_df_columns.append('LOF_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['LOF'] = lof_clf

            # minimum covariance determinant
            if clf_name == 'MCD':
                hyper_mcd = Hyper_MCD(outliers_fraction, score_name=score, beta=beta)
                mcd_clf, threshold = hyper_mcd.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('MCD_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['MCD'] = mcd_clf

            # gaussian mixture model
            if clf_name == 'GMM':
                hyper_GMM = Hyper_GMM(outliers_fraction, score_name=score, beta=beta)
                gmm_clf, threshold = hyper_GMM.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('GMM_n_components_' + str(i))
                hp_list.append(gmm_clf.n_components)
                hp_df_columns.append('GMM_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['GMM'] = gmm_clf

            # histogram-based outlier detection
            if clf_name == 'HBOS':
                hyper_HBOS = Hyper_HBOS(outliers_fraction, score_name=score, beta=beta)
                hbos_clf, threshold = hyper_HBOS.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('HBOS_n_bins_' + str(i))
                hp_list.append(hbos_clf.n_bins)
                hp_df_columns.append('HBOS_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['HBOS'] = hbos_clf

            # Kernel density estimation
            if clf_name == 'KDE':
                hyper_kde = Hyper_KDE(outliers_fraction, score_name=score, beta=beta)
                kde_clf, threshold = hyper_kde.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('KDE_bandwith_' + str(i))
                hp_list.append(kde_clf.bandwith)
                hp_df_columns.append('KDE_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['KDE'] = kde_clf

            # Empirical Cumulative Distribution Functions (ECOD)
            if clf_name == 'ECOD':
                hyper_ecod = Hyper_ECOD(outliers_fraction, score_name=score, beta=beta)
                ecod_clf, threshold = hyper_ecod.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('ECOD_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['ECOD'] = ecod_clf

            # One class support vector machine
            if clf_name == 'OCSVM':
                hyper_ocsvm = Hyper_OCSVM(outliers_fraction, score_name=score, beta=beta)
                ocsvm_clf, threshold = hyper_ocsvm.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('OCSVM_nu_' + str(i))
                hp_list.append(ocsvm_clf.nu)
                hp_df_columns.append('OCSVM_gamma_' + str(i))
                hp_list.append(ocsvm_clf.gamma)
                hp_df_columns.append('OCSVM_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['OCSVM'] = ocsvm_clf

            # Deep One-Class Classification
            if clf_name == 'DSVDD':
                hyper_dsvdd = Hyper_DSVDD(outliers_fraction, score_name=score, beta=beta)
                dsvdd_clf, threshold = hyper_dsvdd.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('DSVDD_n_neurons_' + str(i))
                hp_list.append(dsvdd_clf.n_neurons)
                hp_df_columns.append('DSVDD_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['DSVDD'] = dsvdd_clf

            # Auto Encoder / Replicator Neural Network
            if clf_name == 'RNN':
                hyper_RNN = Hyper_RNN(outliers_fraction, score_name=score, beta=beta)
                rnn_clf, threshold = hyper_RNN.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('RNN_n_neurons_' + str(i))
                hp_list.append(rnn_clf.n_neurons)
                hp_df_columns.append('RNN_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['RNN'] = rnn_clf

            # Principal Component Analysis
            if clf_name == 'PCA':
                hyper_PCA = Hyper_PCA(outliers_fraction, score_name=score, beta=beta)
                pca_clf, threshold = hyper_PCA.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('PCA_n_components_' + str(i))
                hp_list.append(pca_clf.n_components)
                hp_df_columns.append('PCA_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['PCA'] = pca_clf

            # Feature bagging with K nearest neighbors
            if clf_name == 'FB':
                hyper_FB_KNN = Hyper_FB_KNN(outliers_fraction, score_name=score, beta=beta)
                fb_knn_clf, threshold = hyper_FB_KNN.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('FB_KNN_n_estimators_' + str(i))
                hp_list.append(fb_knn_clf.n_estimators)
                hp_df_columns.append('FB_KNN_n_neighbors_' + str(i))
                hp_list.append(fb_knn_clf.n_neighbors)
                hp_df_columns.append('FB_KNN_p_' + str(i))
                hp_list.append(fb_knn_clf.p)
                hp_df_columns.append('FB_KNN_method_' + str(i))
                hp_list.append(fb_knn_clf.method)
                hp_df_columns.append('FB_KNN_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['FB'] = fb_knn_clf

            # Isolation forest
            if clf_name == 'IF':
                hyper_if = Hyper_IF(outliers_fraction, score_name=score, beta=beta)
                if_clf, threshold = hyper_if.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('IF_seed_' + str(i))
                hp_list.append(if_clf.random_state)
                hp_df_columns.append('IF_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['IF'] = if_clf

            # Gaussian process regression
            if clf_name == 'GPR':
                hyper_gpr = Hyper_GP(outliers_fraction, score_name=score, beta=beta)
                gpr_clf, threshold = hyper_gpr.get_clf(X_train, X_val, y_val.reshape((-1, 1)))
                hp_df_columns.append('GPR_lengthscale_' + str(i))
                hp_list.append(gpr_clf.kernel.k2.length_scale)
                hp_df_columns.append('GPR_threshold_' + str(i))
                hp_list.append(threshold)
                classifiers['GPR'] = gpr_clf

        # Loop through classifiers
        for c, clf_name in enumerate(clf_names):
            # Train
            t0 = time.perf_counter_ns()
            classifiers[clf_name].train(X_train)
            t1 = time.perf_counter_ns()

            # Test
            test_scores = classifiers[clf_name].score(X_test)
            t2 = time.perf_counter_ns()

            # Duration
            duration_train = t1 - t0
            duration_test = (t2 - t1) / len(X_test) * 100

            # Score
            roc = roc_auc_score(y_test, test_scores)
            prn = precision_n_scores(y_test, test_scores)
            fscore = scoring.score_samples(y_test.reshape((-1, 1)), test_scores.reshape((-1, 1)),
                                           classifiers[clf_name].threshold, beta=beta, print_opt=False)[score]

            print(
                '{clf_name} Fscore:{fscore}, train time: {train_time}, test time: {test_time}'.
                format(clf_name=clf_name,
                       fscore=fscore,
                       train_time=duration_train,
                       test_time=duration_test))

            # Save results
            time_train_mat[i, c] = duration_train
            time_test_mat[i, c] = duration_test
            roc_mat[i, c] = roc
            prn_mat[i, c] = prn
            fscore_mat[i, c] = fscore

    # Save training time results
    time_train_list = time_train_list + np.mean(time_train_mat, axis=0).tolist()
    time_train_df_temp = pd.DataFrame(time_train_list).transpose()
    time_train_df_temp.columns = df_columns
    if time_train_df is None:
        time_train_df = time_train_df_temp
    else:
        time_train_df = pd.concat([time_train_df, time_train_df_temp], axis=0)

    # Save test time results
    time_test_list = time_test_list + np.mean(time_test_mat, axis=0).tolist()
    time_test_df_temp = pd.DataFrame(time_test_list).transpose()
    time_test_df_temp.columns = df_columns
    if time_test_df is None:
        time_test_df = time_test_df_temp
    else:
        time_test_df = pd.concat([time_test_df, time_test_df_temp], axis=0)

    # Save roc results
    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    roc_df_temp = pd.DataFrame(roc_list).transpose()
    roc_df_temp.columns = df_columns
    if roc_df is None:
        roc_df = roc_df_temp
    else:
        roc_df = pd.concat([roc_df, roc_df_temp], axis=0)

    # Save prn results
    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    prn_df_temp = pd.DataFrame(prn_list).transpose()
    prn_df_temp.columns = df_columns
    if prn_df is None:
        prn_df = prn_df_temp
    else:
        prn_df = pd.concat([prn_df, prn_df_temp], axis=0)

    # Save fscore results
    fscore_list = fscore_list + np.mean(fscore_mat, axis=0).tolist()
    fscore_df_temp = pd.DataFrame(fscore_list).transpose()
    fscore_df_temp.columns = df_columns
    if fscore_df is None:
        fscore_df = fscore_df_temp
    else:
        fscore_df = pd.concat([fscore_df, fscore_df_temp], axis=0)

    # Save hyperparameters
    hp_df_temp = pd.DataFrame(hp_list).transpose()
    hp_df_temp.columns = hp_df_columns
    if hp_df is None:
        hp_df = hp_df_temp
    else:
        hp_df = pd.concat([hp_df, hp_df_temp], axis=0)

    # Save the results for each run
    if not os.path.exists('output'):
        os.makedirs('output')
    time_train_df.to_csv('output/time_train.csv', index=False)
    time_test_df.to_csv('output/time_test.csv', index=False)
    roc_df.to_csv('output/roc.csv', index=False)
    prn_df.to_csv('output/prc.csv', index=False)
    fscore_df.to_csv('output/fscore.csv', index=False)
    hp_df.to_csv('output/hp.csv', index=False)
