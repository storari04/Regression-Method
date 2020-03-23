import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import graphviz
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from sklearn import model_selection, svm, metrics, datasets, tree
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.scores import MLE,CRPS
from ngboost.distns import Normal
from tqdm import tqdm_notebook as tqdm
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import math
import optuna
import shap

max_pls_component_number = 50 # max component number in PLS
ridge_lambdas = 2 ** np.arange(-5, 10, dtype=float)  # L2 weight in ridge regression
lasso_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # L1 weight in LASSO
elastic_net_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # Lambda in elastic net
elastic_net_alphas = np.arange(0.01, 1.00, 0.01, dtype=float)  # Alpha in elastic net
linear_svr_cs = 2 ** np.arange(-5, 5, dtype=float)  # C for linear svr
linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for linear svr
nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # C for nonlinear svr
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for nonlinear svr
nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # Gamma for nonlinear svr
dt_max_max_depth = 30  # 木の深さの最大値、の最大値
dt_min_samples_leaf = 3  # 葉ごとのサンプル数の最小値
random_forest_number_of_trees = 300  # Number of decision trees for random forest
random_forest_x_variables_rates = np.arange(1, 10,
                                            dtype=float) / 10  # Ratio of the number of X-variables for random forest

method_flag = 11 #1 OLS #2 PLSRegression #3 Ridge #4 Lasso #5 Elastic Net #6 Linear SVR #7 Nonlinear SVR
                #8 Decision Tree #9 Randomforest #10 ExtraTrees
                #11 LGBMRegressor #12 XGBRegressor #13 scikit-learn #14 CatBoostRegressor #15 NGRegressor
optimization_method = 2 #1 GridSearchCV #2 Optuna
fold_number = 5
trials = 15 # number of optuna trials
fraction_of_validation_samples = 0.2
number_of_sub_models = 1000
number_of_test_samples = 0.2

# load boston dataset
boston = datasets.load_boston()
x = boston.data
x = pd.DataFrame(x)
x.columns = boston.feature_names
y = boston.target

# Divide samples into training samples and test samples
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

#autoscaling
autocalculated_train_x = (train_x - train_x.mean(axis = 0))/train_x.std(ddof = 1, axis = 0)
autocalculated_test_x = (test_x - train_x.mean(axis = 0))/train_x.std(ddof = 1, axis = 0)
autocalculated_train_y = (train_y - train_y.mean(axis = 0))/train_y.std(ddof = 1, axis = 0)

if method_flag == 1:  # Ordinary Least Squares
    regression_model = LinearRegression()

if method_flag == 2: # PLSRegression
    PLS_components = np.arange(1,min(np.linalg.matrix_rank(autocalculated_train_x) + 1, max_pls_component_number + 1), 1)
    r2cvall = list()
    for PLS_component in PLS_components:
        pls_model_in_cv = PLSRegression(n_components = PLS_component)
        pls_model_in_cv.fit(autocalculated_train_x, autocalculated_train_y)
        calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autocalculated_train_x))
        estimated_y_in_cv = np.ndarray.flatten(model_selection.cross_val_predict(pls_model_in_cv, autocalculated_train_x, autocalculated_train_y, cv = fold_number))
        calculated_y_in_cv = calculated_y_in_cv * train_y.std(ddof = 1) + train_y.mean(axis = 0)
        estimated_y_in_cv = estimated_y_in_cv * train_y.std(ddof = 1) + train_y.mean(axis = 0)
        r2cvall.append(float(1 - sum((train_y - estimated_y_in_cv) ** 2) / sum((train_y - train_y.mean()) ** 2)))
    plt.plot(PLS_components, r2cvall, 'bo-')
    plt.ylim(0,1)
    plt.xlabel('Number of PLS components')
    plt.ylabel('r2cv(blue)')
    plt.show()
    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
    optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
    regression_model = PLSRegression(n_components=optimal_pls_component_number)
elif method_flag == 3: #Ridgeregression
    r2cvall = list()
    for ridge_lambda in ridge_lambdas:
        rr_model_in_cv = Ridge(alpha = ridge_lambda)
        estimated_y_in_cv = np.ndarray.flatten(model_selection.cross_val_predict(rr_model_in_cv, autocalculated_train_x, autocalculated_train_y, cv = fold_number))
        estimated_y_in_cv = np.ndarray.flatten(estimated_y_in_cv * train_y.std(ddof = 1) + train_y.mean(axis = 0))
        r2cvall.append(float(1 - sum((train_y - estimated_y_in_cv) ** 2) / sum((train_y - train_y.mean()) ** 2)))
    plt.plot(ridge_lambdas, r2cvall, 'bo-')
    plt.ylim(0,1)
    plt.xlabel('alpha')
    plt.ylabel('r2cv(blue)')
    plt.show()
    optimal_ridge_lambda = np.where(r2cvall == np.max(r2cvall))
    optimal_ridge_lambda = optimal_ridge_lambda[0][0] + 1
    regression_model = Ridge(alpha=optimal_ridge_lambda)
elif method_flag == 4: #Lassoregression
    r2cvall = list()
    for lasso_lambda in lasso_lambdas:
        lr_model_in_cv = Lasso(alpha = lasso_lambda)
        estimated_y_in_cv = model_selection.cross_val_predict(lr_model_in_cv, autocalculated_train_x, autocalculated_train_y, cv = fold_number)
        estimated_y_in_cv = estimated_y_in_cv * train_y.std(ddof = 1) + train_y.mean(axis = 0)
        r2cvall.append(float(1 - sum((train_y - estimated_y_in_cv) ** 2) / sum((train_y - train_y.mean()) ** 2)))
    plt.figure()
    plt.plot(lasso_lambdas, r2cvall, 'k', linewidth=2)
    plt.xlabel('Weight for LASSO')
    plt.ylabel('r2cv for LASSO')
    plt.show()
    optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
    regression_model = Lasso(alpha=optimal_lasso_lambda)
elif method_flag == 5: #Elastic net
    elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
    elastic_net_in_cv.fit(autocalculated_train_x, autocalculated_train_y)
    optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
    optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
    regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
elif method_flag == 6: #LinearSVR
    if optimization_method == 1:
        #GridSearchCV
        linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                    cv=fold_number)
        linear_svr_in_cv.fit(autocalculated_train_x, autocalculated_train_y)
        optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']
        optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']

        regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)

    else:
        #optuna
        def objective(trial):
            param = {
                    'kernel': 'linear',
                    'C': trial.suggest_loguniform('C', 1e-2, 1e2),
                    'epsilon': trial.suggest_loguniform('epsilon', 1e-4, 1e1),
            }

            regression_model = svm.SVR(**param)
            estimated_y_in_cv = model_selection.cross_val_predict(regression_model, autocalculated_train_x, autocalculated_train_y, cv=fold_number)
            estimated_y_in_cv = estimated_y_in_cv * train_y.std() + train_y.mean()
            r2 = metrics.r2_score(train_y, estimated_y_in_cv)
            return 1.0 - r2

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    regression_model = svm.SVR(**study.best_params)

elif method_flag == 7:  # Nonlinear SVR
    variance_of_gram_matrix = list()
    numpy_autoscaled_Xtrain = np.array(autocalculated_train_x)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(
            -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(
                axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[
        np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]

    if optimization_method == 1:
        #GridSearchCV
        nonlinear_svr_in_cv = GridSearchCV(svm.SVR(kernel='rbf', gamma=optimal_nonlinear_gamma),
                                       {'C': nonlinear_svr_cs, 'epsilon': nonlinear_svr_epsilons}, cv=fold_number)
        nonlinear_svr_in_cv.fit(autocalculated_train_x, autocalculated_train_y)
        optimal_nonlinear_c = nonlinear_svr_in_cv.best_params_['C']
        optimal_nonlinear_epsilon = nonlinear_svr_in_cv.best_params_['epsilon']
        regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                               gamma=optimal_nonlinear_gamma)

    if optimization_method == 2:
        #optuna
        def objective(trial):
            param = {
                    'kernel': 'rbf',
                    'C': trial.suggest_loguniform('C', 1e-2, 1e2),
                    'epsilon': trial.suggest_loguniform('epsilon', 1e-4, 1e1),
            }

            regression_model = svm.SVR(gamma = optimal_nonlinear_gamma, **param)
            estimated_y_in_cv = model_selection.cross_val_predict(regression_model, autocalculated_train_x, autocalculated_train_y, cv=fold_number)
            estimated_y_in_cv = estimated_y_in_cv * train_y.std() + train_y.mean()
            r2 = metrics.r2_score(train_y, estimated_y_in_cv)
            return 1.0 - r2

        study = optuna.create_study()
        study.optimize(objective, n_trials=trials)

        regression_model = svm.SVR(gamma=optimal_nonlinear_gamma, **study.best_params)

elif method_flag == 8:  # Decision tree
    # クロスバリデーションによる木の深さの最適化
    r2cv_all = []
    for max_depth in range(2, dt_max_max_depth):
        model_in_cv = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=dt_min_samples_leaf)
        estimated_y_in_cv = model_selection.cross_val_predict(model_in_cv, autocalculated_train_x, autocalculated_train_y,
                                                              cv=fold_number) * train_y.std(ddof=1) + train_y.mean()
        r2cv_all.append(1 - sum((train_y - estimated_y_in_cv) ** 2) / sum((train_y - train_y.mean()) ** 2))
    optimal_max_depth = np.where(r2cv_all == np.max(r2cv_all))[0][0] + 2  # r2cvが最も大きい木の深さ
    regression_model = tree.DecisionTreeRegressor(max_depth=optimal_max_depth,
                                                  min_samples_leaf=dt_min_samples_leaf)  # DTモデルの宣言

elif method_flag == 9:  # Random forest
    rmse_oob_all = list()
    for random_forest_x_variables_rate in random_forest_x_variables_rates:
        RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
            max(math.ceil(train_x.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
        RandomForestResult.fit(autocalculated_train_x, autocalculated_train_y)
        estimated_y_in_cv = RandomForestResult.oob_prediction_
        estimated_y_in_cv = estimated_y_in_cv * train_y.std(ddof=1) + train_y.mean()
        rmse_oob_all.append((sum((train_y - estimated_y_in_cv) ** 2) / len(train_y)) ** 0.5)
    plt.figure()
    plt.plot(random_forest_x_variables_rates, rmse_oob_all, 'k', linewidth=2)
    plt.xlabel('Ratio of the number of X-variables')
    plt.ylabel('RMSE of OOB')
    plt.show()
    optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
        np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
    regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
        max(math.ceil(train_x.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)

elif method_flag == 10:  # Extremely Randomized Tree
    #optuna
    def objective(trial):
        param = {
                'n_estimators': trial.suggest_int('n_estimators', 20, 400),
                'max_features': trial.suggest_uniform('max_features', 0.1, 0.9),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
            }

        regression_model = ExtraTreesRegressor(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(regression_model, autocalculated_train_x, autocalculated_train_y, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * train_y.std() + train_y.mean()
        r2 = metrics.r2_score(train_y, estimated_y_in_cv)
        return 1.0 - r2

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    regression_model = ExtraTreesRegressor(**study.best_params)

elif method_flag == 11:  # LightGBM
    train_x_tmp, train_x_validation, train_y_tmp, train_y_validation = train_test_split(train_x,
                                                                                        train_y,
                                                                                        test_size=fraction_of_validation_samples,
                                                                                        random_state=0)
    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        regression_model = lgb.LGBMRegressor(n_estimators = 1000)
        regression_model.fit(train_x_tmp, train_y_tmp, eval_set=(train_x_validation, train_y_validation), eval_metric = 'rmse', early_stopping_rounds = 100)
        best_n_estimators_in_cv = regression_model.best_iteration_

    def objective(trial):
        param = {
            'objective': 'regression',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
            #'num_boost_round': trial.suggest_int('num_boost_round', 10, 100000),
            #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100000),
            #'min_child_samples': trial.suggest_int('min_child_samples', 5, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 500),
        }

        if param['boosting_type'] == 'dart':
            param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        if param['boosting_type'] == 'goss':
            param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])

        regression_model = lgb.LGBMRegressor(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(regression_model, train_x, train_y, cv=fold_number)
        r2 = metrics.r2_score(train_y, estimated_y_in_cv)
        return 1.0 - r2

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)
    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        regression_model = lgb.LGBMRegressor(**study.best_params, n_estimators=1000)
        regression_model.fit(train_x_tmp, train_y_tmp, eval_set=(train_x_validation, train_y_validation),
                  eval_metric='rmse', early_stopping_rounds=100)
        best_n_estimators = regression_model.best_iteration_
    regression_model = lgb.LGBMRegressor(**study.best_params)

elif method_flag == 12:  # XGBoost
    train_x_tmp, train_x_validation, autocalculated_train_y_tmp, autocalculated_train_y_validation = train_test_split(train_x,
                                                                                                                      autocalculated_train_y,
                                                                                                                      test_size=fraction_of_validation_samples,
                                                                                                                      random_state=0)
    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        regression_model = xgb.XGBRegressor(n_estimators=1000)
        regression_model.fit(train_x_tmp, autocalculated_train_y_tmp,
                  eval_set=[(train_x_validation, autocalculated_train_y_validation.reshape([len(autocalculated_train_y_validation), 1]))],
                  eval_metric='rmse', early_stopping_rounds=100)
        best_n_estimators_in_cv = regression_model.best_iteration

    def objective(trial):
        param = {
            'silent': 1,
            'objective': 'reg:linear',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
        }

        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

        regression_model = xgb.XGBRegressor(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(regression_model, train_x, autocalculated_train_y, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * train_y.std() + train_y.mean()
        r2 = metrics.r2_score(train_y, estimated_y_in_cv)
        return 1.0 - r2
    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        regression_model = xgb.XGBRegressor(**study.best_params, n_estimators=1000)
        regression_model.fit(train_x_tmp, autocalculated_train_y_tmp,
                  eval_set=[(train_x_validation, autocalculated_train_y_validation.reshape([len(autocalculated_train_y_validation), 1]))],
                  eval_metric='rmse', early_stopping_rounds=100)
        best_n_estimators = regression_model.best_iteration

    regression_model = xgb.XGBRegressor(**study.best_params)

elif method_flag == 13:  # scikit-learn
    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        regression_model = GradientBoostingRegressor(n_estimators=1000, validation_fraction=fraction_of_validation_samples,
                                          n_iter_no_change=100)
        regression_model.fit(train_x, train_y)
        best_n_estimators_in_cv = len(regression_model.estimators_)

    def objective(trial):
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 20),
            'max_features': trial.suggest_loguniform('max_features', 0.1, 1.0)
        }

        regression_model = GradientBoostingRegressor(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(regression_model, train_x, train_y, cv=fold_number)
        r2 = metrics.r2_score(train_y, estimated_y_in_cv)
        return 1.0 - r2

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)
    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        regression_model = GradientBoostingRegressor(**study.best_params, n_estimators=1000,
                                          validation_fraction=fraction_of_validation_samples, n_iter_no_change=100)
        regression_model.fit(train_x, train_y)
        best_n_estimators = len(regression_model.estimators_)
    regression_model = GradientBoostingRegressor(**study.best_params)

elif method_flag == 14:  # catboost
    train_x_tmp, train_x_validation, train_y_tmp, train_y_validation = train_test_split(train_x,
                                                                                        train_y,
                                                                                        test_size=fraction_of_validation_samples,
                                                                                        random_state=0)
    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        regression_model = cat.CatBoostRegressor(n_estimators=500, logging_level='Silent')
        regression_model.fit(train_x_tmp, train_y_tmp,
                 eval_set=[(train_x_validation, train_y_validation)],
                 early_stopping_rounds=30)
        best_n_estimators_in_cv = regression_model.best_iteration_

    def objective(trial):
        param = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e0),
            'random_strength': trial.suggest_int('random_strength', 0, 100),
            'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int('od_wait', 10, 50)
        }

        regression_model = cat.CatBoostRegressor(**param, n_estimators=best_n_estimators_in_cv, logging_level='Silent')
        estimated_y_in_cv = model_selection.cross_val_predict(regression_model, train_x, train_y, cv=fold_number)
        r2 = metrics.r2_score(train_y, estimated_y_in_cv)
        return 1.0 - r2

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        regression_model = cat.CatBoostRegressor(**study.best_params, n_estimators=3000, logging_level='Silent')
        regression_model.fit(train_x_tmp, train_y_tmp,
                  eval_set=[(train_x_validation, train_y_validation)],
                  early_stopping_rounds=100)
        best_n_estimators = regression_model.best_iteration_
    regression_model = cat.CatBoostRegressor(**study.best_params, n_estimators=best_n_estimators, logging_level='Silent')

elif method_flag == 15:
    train_x_tmp, train_x_validation, train_y_tmp, train_y_validation = train_test_split(train_x,
                                                                                        train_y,
                                                                                        test_size=fraction_of_validation_samples,
                                                                                        random_state=0)

    # 決定木
    ngb_tree = NGBRegressor(Base=default_tree_learner).fit(train_x_tmp, train_y_tmp)
    Y_preds_tree = ngb_tree.predict(train_x_validation)
    Y_dists_tree = ngb_tree.pred_dist(train_x_validation)

    # test Mean Squared Error
    test_MSE_tree = mean_squared_error(Y_preds_tree, train_y_validation)
    print('Test MSE_tree', test_MSE_tree)

    # test Negative Log Likelihood
    test_NLL_tree = -Y_dists_tree.logpdf(train_y_validation.flatten()).mean()
    print('Test NLL_tree', test_NLL_tree)

    # Ridge
    ngb_ridge = NGBRegressor(Base=default_linear_learner).fit(train_x_tmp, train_y_tmp)
    Y_preds_ridge = ngb_ridge.predict(train_x_validation)
    Y_dists_ridge = ngb_ridge.pred_dist(train_x_validation)

    # test Mean Squared Error
    test_MSE_ridge = mean_squared_error(Y_preds_ridge, train_y_validation)
    print('Test MSE_ridge', test_MSE_ridge)

    # test Negative Log Likelihood
    test_NLL_ridge = -Y_dists_ridge.logpdf(train_y_validation.flatten()).mean()
    print('Test NLL_ridge', test_NLL_ridge)

    if test_MSE_tree <= test_MSE_ridge:
        best_base = default_linear_learner
    else:
        best_base = default_tree_learner

    def objective(trial):
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'minibatch_frac':trial.suggest_discrete_uniform('minibatch_frac', 0.1, 0.9, 0.1),
        }

        regression_model = NGBRegressor(**param, Base= best_base, Dist=Normal, Score=MLE(), natural_gradient=True, verbose=False)
        estimated_y_in_cv = model_selection.cross_val_predict(regression_model, train_x, train_y, cv=fold_number)
        r2 = metrics.r2_score(train_y, estimated_y_in_cv)
        return 1.0 - r2

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)
    regression_model = NGBRegressor(**study.best_params)

if method_flag <= 10:
    clf = regression_model.fit(autocalculated_train_x, autocalculated_train_y)
elif method_flag == 11:
    clf = regression_model.fit(train_x, train_y)
elif method_flag == 13 or method_flag == 14 or method_flag == 15:
    clf = regression_model.fit(train_x, train_y)
elif method_flag == 12:
    clf = regression_model.fit(train_x, autocalculated_train_y)

#Visualization
#calculated, estimated
if method_flag <= 10:
    calculated_ytrain = np.ndarray.flatten(clf.predict(autocalculated_train_x))
    estimated_ytrain = np.ndarray.flatten(model_selection.cross_val_predict(clf, autocalculated_train_x, autocalculated_train_y, cv = fold_number))
    calculated_ytrain = calculated_ytrain * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)
    estimated_ytrain = estimated_ytrain * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)
elif method_flag == 11 or method_flag == 13 or method_flag == 14:
    calculated_ytrain = np.ndarray.flatten(clf.predict(train_x))
    estimated_ytrain = np.ndarray.flatten(model_selection.cross_val_predict(clf, train_x, train_y, cv = fold_number))
elif method_flag == 12:
    calculated_ytrain = np.ndarray.flatten(clf.predict(train_x))
    estimated_ytrain = np.ndarray.flatten(model_selection.cross_val_predict(clf, train_x, autocalculated_train_y, cv = fold_number))
    calculated_ytrain = calculated_ytrain * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)
    estimated_ytrain = estimated_ytrain * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)
elif method_flag == 15:
    calculated_ytrain = clf.predict(train_x)
    calculated_ytrain_dist = clf.pred_dist(train_x)
    estimated_ytrain = model_selection.cross_val_predict(clf, train_x, train_y, cv = fold_number)

# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((train_y - calculated_ytrain) ** 2) / sum((train_y - train_y.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((train_y - calculated_ytrain) ** 2) / len(train_y)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(train_y - calculated_ytrain)) / len(train_y))))

#yy plot for calculated Y
plt.figure(figsize=figure.figaspect(1))
plt.scatter(train_y, calculated_ytrain)
YMax = np.max(np.array([np.array(train_y), calculated_ytrain]))
YMin = np.min(np.array([np.array(train_y), calculated_ytrain]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()

# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((train_y - estimated_ytrain) ** 2) / sum((train_y - train_y.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((train_y - estimated_ytrain) ** 2) / len(train_y)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(train_y - estimated_ytrain)) / len(train_y))))

#yy plot for estimated Y
plt.figure(figsize=figure.figaspect(1))
plt.scatter(train_y, estimated_ytrain)
YMax = np.max(np.array([np.array(train_y), estimated_ytrain]))
YMin = np.min(np.array([np.array(train_y), estimated_ytrain]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y')
plt.show()

#Visualization
if method_flag == 1:
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(clf.coef_)  # Pandas の DataFrame 型に変換
    standard_regression_coefficients.index = pd.DataFrame(train_x).columns  # 説明変数に対応する名前を、元のデータセットにおける説明変数の名前に
    standard_regression_coefficients.columns = ['standard_regression_coefficients']  # 列名を変更
    standard_regression_coefficients.to_csv(
        'standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

if method_flag == 2:
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(clf.coef_, index=train_x.columns,
                                                columns=['standard_regression_coefficients'])  # Pandas の DataFrame 型に変換
    standard_regression_coefficients.to_csv(
    'pls_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

if method_flag == 8:
    """
    #Tree Plot
    fig = plt.figure(figsize=figure.figaspect(1))
    ax = fig.add_subplot()
    plot_tree(clf, feature_names=boston.feature_names, ax=ax, filled=True);
    plt.show()
    """

    #Graphviz
    dot_data = export_graphviz(
                        clf,
                        #class_names=iris.target_names,
                        feature_names=boston.feature_names,
                        filled=True,
                        rounded=True,
                        out_file=None
                    )
    graph = graphviz.Source(dot_data)
    graph.render("boston-tree", format="png")

if method_flag == 9:
    # 説明変数の重要度
    x_importances = pd.DataFrame(clf.feature_importances_, index=pd.DataFrame(train_x).columns, columns=['importance'])
    x_importances.to_csv('rf_x_importances.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

if method_flag == 11:
    lgb.plot_importance(clf, figsize = (18,8), max_num_features=30)
    plt.show()

if method_flag == 12:
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    xgb.plot_importance(clf, ax = ax, max_num_features = 30, importance_type = 'weight')
    plt.show()

if method_flag == 13:
    feature_importances = clf.feature_importances_
    # make importances relative to max importance
    feature_importances = 100.0 * (feature_importances / feature_importances.max())
    sorted_idx = np.argsort(feature_importances)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importances[sorted_idx], align='center')
    plt.yticks(pos, boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

"""
if method_flag == 15:
    offset = np.ptp(calculated_ytrain)*0.1
    y_range = np.linspace(min(train_y)-offset, max(train_y)+offset, 200).reshape((-1, 1))
    dist_values = calculated_ytrain_dist.pdf(y_range).transpose()

    plt.figure(figsize=(25, 120))
    for idx in tqdm(np.arange(train_x.shape[0])):

        plt.subplot(135, 3, idx+1)
        plt.plot(y_range, dist_values[idx])

        plt.vlines(calculated_ytrain[idx], 0, max(dist_values[idx]), "r", label="ngb pred")
        plt.vlines(train_y[idx], 0, max(dist_values[idx]), "pink", label="ground truth")
        plt.legend(loc="best")
        plt.title(f"idx: {idx}")
        plt.xlim(y_range[0], y_range[-1])
    plt.tight_layout()
    plt.show()
"""

#SHAP_visualization
if method_flag == 1 or 3 <= method_flag <= 5:
    explainer = shap.LinearExplainer(clf, train_x)
    shap_values = explainer.shap_values(train_x.loc[[0]])
    shap.force_plot(explainer.expected_value, shap_values[0], train_x.loc[[0]], matplotlib = True)

"""
    shap_values = explainer.shap_values(train_x)
    shap.summary_plot(shap_values, features = train_x,
                #plot_type = 'bar'
                )
    shap.dependence_plot(ind="RM", shap_values=shap_values, features = train_x,
                    interaction_index = 'TSTAT',
                    )
"""

if method_flag == 2:
    explainer = shap.KernelExplainer(clf.predict, train_x)
    shap_values = explainer.shap_values(train_x.loc[[0]])
    shap.force_plot(explainer.expected_value, shap_values[0], train_x.loc[[0]], matplotlib = True)

"""
    shap_values = explainer.shap_values(train_x)
    shap.summary_plot(shap_values, features = train_x,
                #plot_type = 'bar'
                )
    shap.dependence_plot(ind="RM", shap_values=shap_values, features = train_x,
                    interaction_index = 'TSTAT',
                    )
"""

if method_flag == 6 or method_flag == 7:
    explainer = shap.KernelExplainer(clf.predict, train_x)
    shap_values = explainer.shap_values(train_x.loc[[0]])
    shap.force_plot(explainer.expected_value, shap_values[0], train_x.loc[[0]], matplotlib = True)

"""
    shap_values = explainer.shap_values(train_x)
    shap.summary_plot(shap_values, features = train_x,
                #plot_type = 'bar'
                )

    shap.dependence_plot(ind="RM", shap_values=shap_values, features = train_x,
                    interaction_index = 'TSTAT',
                    )
"""

if 8 <= method_flag <= 14:
    explainer = shap.TreeExplainer(clf)

    #shap_values = explainer.shap_values(train_x.loc[[0]])
    #shap.force_plot(explainer.expected_value, shap_values[0], train_x.loc[[0]], matplotlib = True)

    shap_values = explainer.shap_values(train_x)
    shap.summary_plot(shap_values, features = train_x,
                    #plot_type = 'bar'
                    )

    shap.dependence_plot(ind="RM", shap_values=shap_values, features = train_x,
                    interaction_index = 'LSTAT',
                    )

"""
    shap.decision_plot(explainer.expected_value, shap_values[1][0:20], train_x[0:20],
                       #link="logit",
                       #highlight=misclassified[0:20]
                    )
"""

#prediction
if method_flag <= 10:
    predicted_ytest = np.ndarray.flatten(clf.predict(autocalculated_test_x))
    predicted_ytest = predicted_ytest * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)
elif method_flag == 11 or method_flag == 13 or method_flag == 14:
    predicted_ytest = np.ndarray.flatten(clf.predict(test_x))
elif method_flag == 12:
    predicted_ytest = np.ndarray.flatten(clf.predict(test_x))
    predicted_ytest = predicted_ytest * train_y.std(ddof = 1, axis = 0) + train_y.mean(axis = 0)
elif method_flag == 15:
    predicted_ytest = clf.predict(test_x)

# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((test_y - predicted_ytest) ** 2) / sum((test_y - test_y.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((test_y - predicted_ytest) ** 2) / len(test_y)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(test_y - predicted_ytest)) / len(test_y))))

#yy plot for real Y
plt.figure(figsize=figure.figaspect(1))
plt.scatter(test_y, predicted_ytest)
YMax = np.max(np.array([np.array(test_y), predicted_ytest]))
YMin = np.min(np.array([np.array(test_y), predicted_ytest]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y')
plt.show()

"""
#submit file
submit_file = pd.DataFrame({'Id' : test_id , 'revenue' : predicted_ytest})
submit_file.to_csv('Officesci.csv', index = False)
"""
