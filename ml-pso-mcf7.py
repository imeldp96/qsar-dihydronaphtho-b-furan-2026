# ================================
# Multi-target regression pipeline
# Models: RF, SVR, GBR, XGB, MLP (ANN)
# PSO hyperparameter optimization
# ================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GBR
from sklearn.svm import SVR
from xgboost import XGBRegressor as XGB
from sklearn.neural_network import MLPRegressor as ANN
import pyswarms as ps

# =========================================
# 0. Set HPC CPU cores for PSO
# =========================================
n_cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

# =========================================
# 1. Preprocessing for SVR/MLP
# =========================================
data = pd.read_csv('./mcf7.csv', index_col=[0])
data = data.fillna(0)

X= data.drop(['pLD50'], axis=1)
y = data['pLD50']

test_compounds = ['3a', '3d', '3e', '3h', '3f', '3g']

# subset X1 and y1
X_train_fi, X_test_fi = X.loc[~X.index.isin(test_compounds)], X.loc[test_compounds]
y_train, y_test = y.loc[~y.index.isin(test_compounds)], y.loc[test_compounds]

X_train = X_train_fi[['VSA_EState3', 'AXp-0dv', 'GATS1i']] #pi-gbr
X_test = X_test_fi[['VSA_EState3', 'AXp-0dv', 'GATS1i']]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv_splitter = KFold(n_splits=4, shuffle=True, random_state=42)

# =========================================
# 2. Random Forest (RF) + PSO
# =========================================
rf_bounds = (np.array([2, 1, 1, 0]), np.array([500, 20, 20, 1]))  # Example bounds

def rf_pso_objective(params):
    scores = []
    for param in params:
        n_estimators = int(param[0])
        min_samples_split = int(param[1])
        min_samples_leaf = int(param[2])
        bootstrap = bool(round(param[3]))
        try:
            model = RF(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                random_state=42
            )
            score = -np.mean(cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='r2'))
        except Exception:
            score = 9999
        scores.append(score)
    return np.array(scores)

base_pos=np.array([2.00992795, 9.59302585, 1.69068321, 0.46402595])
n_particles = 250
dimensions = 4
noise_scale = 0  # adjust as needed
init_pos = np.tile(base_pos, (n_particles, 1)) + np.random.uniform(-noise_scale, noise_scale, size=(n_particles, dimensions))

rf_optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=4, 
                options={'c1':0.5,'c2':0.3,'w':0.9}, bounds=rf_bounds, init_pos=init_pos)

rf_best_score, rf_best_params = rf_optimizer.optimize(rf_pso_objective, iters=200, n_processes=n_cpu)

rf_params_formatted = {
    'n_estimators': int(rf_best_params[0]),
    'min_samples_split': int(rf_best_params[1]),
    'min_samples_leaf': int(rf_best_params[2]),
    'bootstrap': bool(round(rf_best_params[3]))
}
print("Best RF params:", rf_params_formatted)

model_rf = RF(**rf_params_formatted, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_train_pred_rf = model_rf.predict(X_train)
y_test_pred_rf = model_rf.predict(X_test)

# =========================================
# 3. SVR + PSO
# =========================================
svr_param_bounds = {
    'C': (0.01, 20),
    'gamma': (0.01, 1.0)
    }

lower_bounds = [svr_param_bounds[k][0] for k in svr_param_bounds]
upper_bounds = [svr_param_bounds[k][1] for k in svr_param_bounds]

def svr_pso_objective(params):
    scores=[]
    for param in params:
        C, gamma = param
        try:
            model = SVR(C=C, gamma=gamma, kernel='linear')
            score=-np.mean(cross_val_score(model, X_train_scaled, y_train, cv=cv_splitter, scoring='r2'))
        except:
            score=9999
        scores.append(score)
    return np.array(scores)

base_pos=np.array([3.23923354, 0.06375312])
n_particles = 250
dimensions = 2
noise_scale = 0.00  # adjust as needed
init_pos = np.tile(base_pos, (n_particles, 1)) + np.random.uniform(-noise_scale, noise_scale, size=(n_particles, dimensions))

svr_optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=2,
                                        options={'c1':0.5,'c2':0.3,'w':0.9},
                                        bounds=(np.array(lower_bounds), np.array(upper_bounds)), init_pos=init_pos)
svr_best_score, svr_best_params = svr_optimizer.optimize(svr_pso_objective, iters=500, n_processes=n_cpu)

best_svr_params={'C':svr_best_params[0],'gamma':svr_best_params[1]}
print("Best SVR params:", best_svr_params)

model_svr = SVR(**best_svr_params)
model_svr.fit(X_train_scaled, y_train)
y_train_pred_svr = model_svr.predict(X_train_scaled)
y_test_pred_svr = model_svr.predict(X_test_scaled)

# =========================================
# 4. GBR + PSO
# =========================================
gbr_bounds=(np.array([2,1,0.1,1,0.01]), np.array([600,30,1.0,20,1.0]))

def gbr_pso_objective(params):
    scores=[]
    for param in params:
        try:
            model=GBR(
                n_estimators=int(param[0]),
                min_samples_split=int(param[1]),
                subsample=float(param[2]),
                min_samples_leaf=int(param[3]),
                learning_rate=float(param[4]),
                random_state=42
            )
            score=-np.mean(cross_val_score(model,X_train,y_train,cv=cv_splitter,scoring='r2'))
        except:
            score=9999
        scores.append(score)
    return np.array(scores)

base_pos=np.array([27.85095938, 9.21839968, 0.92221863, 4.32360259, 0.95087352])
n_particles = 250
dimensions = 5
noise_scale = 0.00  # adjust as needed
init_pos = np.tile(base_pos, (n_particles, 1)) + np.random.uniform(-noise_scale, noise_scale, size=(n_particles, dimensions))


gbr_optimizer=ps.single.GlobalBestPSO(n_particles=500, dimensions=5,
                                      options={'c1':0.5,'c2':0.3,'w':0.9},
                                      bounds=gbr_bounds, init_pos=init_pos)
gbr_best_score,gbr_best_params=gbr_optimizer.optimize(gbr_pso_objective,iters=200,n_processes=n_cpu)
gbr_params_formatted={
    'n_estimators':int(gbr_best_params[0]),
    'min_samples_split':int(gbr_best_params[1]),
    'subsample':float(gbr_best_params[2]),
    'min_samples_leaf':int(gbr_best_params[3]),
    'learning_rate':float(gbr_best_params[4])
}
print("Best GBR params:", gbr_params_formatted)
model_gbr = GBR(**gbr_params_formatted, random_state=42)
model_gbr.fit(X_train,y_train)
y_train_pred_gbr=model_gbr.predict(X_train)
y_test_pred_gbr=model_gbr.predict(X_test)

# =========================================
# 5. XGB + PSO
# =========================================
xgb_bounds=(np.array([0.1, 0.0, 0, 0, 0.0, 0.0, 0, 0]), 
            np.array([1.0, 10, 30, 10, 1.0, 1.0, 10, 1]))

def xgb_pso_objective(params):
    scores=[]
    for param in params:
        try:
            model=XGB(
                learning_rate=float(param[0]),
                gamma=float(param[1]),
                max_depth=int(round(param[2])),
                min_child_weight=int(round(param[3])),
                subsample=float(param[4]),
                colsample_bytree=float(param[5]),
                reg_lambda=float(param[6]),
                reg_alpha=float(param[7]),
                objective='reg:squarederror', random_state=42, n_jobs=1, verbosity=0
            )
            score=-np.mean(cross_val_score(model,X_train,y_train,cv=cv_splitter,scoring='r2'))
        except:
            score=9999
        scores.append(score)
    return np.array(scores)

base_pos=np.array([6.51398717e-01, 7.41600592e-03, 1.24096274e+01, 2.22717253e+00,
 5.21131973e-01, 5.56662016e-01, 2.41373093e+00, 3.67972904e-02])
n_particles = 250
dimensions = 8
noise_scale = 0.00  # adjust as needed
init_pos = np.tile(base_pos, (n_particles, 1)) + np.random.uniform(-noise_scale, noise_scale, size=(n_particles, dimensions))


xgb_optimizer=ps.single.GlobalBestPSO(n_particles=500,dimensions=8,
                                      options={'c1':0.5,'c2':0.3,'w':0.9},
                                      bounds=xgb_bounds, init_pos=init_pos)
xgb_best_score,xgb_best_params=xgb_optimizer.optimize(xgb_pso_objective,iters=200,n_processes=n_cpu)
xgb_params_formatted={
    'learning_rate':float(xgb_best_params[0]),
    'gamma':float(xgb_best_params[1]),
    'max_depth':int(round(xgb_best_params[2])),
    'min_child_weight':int(round(xgb_best_params[3])),
    'subsample':float(xgb_best_params[4]),
    'colsample_bytree':float(xgb_best_params[5]),
    'reg_lambda':float(xgb_best_params[6]),
    'reg_alpha':float(xgb_best_params[7])
}
print("Best XGB params:", xgb_params_formatted)
model_xgb=XGB(**xgb_params_formatted,objective='reg:squarederror')
model_xgb.fit(X_train,y_train)
y_train_pred_xgb=model_xgb.predict(X_train)
y_test_pred_xgb=model_xgb.predict(X_test)

# =========================================
# 6. MLP + PSO
# =========================================
mlp_bounds=(np.array([2, 1e-5, 1000, 1e-5, 10000]), 
            np.array([500, 100, 100000, 1, 100000]))

def mlp_pso_objective(params):
    scores=[]
    for param in params:
        try:
            model=ANN(hidden_layer_sizes=(int(round(param[0])),),
                                           alpha=float(param[1]),
                                           max_iter=int(round(param[2])),
                                           tol=float(param[3]),
                                           solver='lbfgs',
                                           max_fun=int(round(param[4])),
                                           random_state=42)
            score=-np.mean(cross_val_score(model,X_train_scaled,y_train,cv=cv_splitter,scoring='r2'))
        except:
            score=9999
        scores.append(score)
    return np.array(scores)


base_pos=np.array([2.51045220e+02, 1.64355793e+00, 6.01976189e+04, 2.97454044e-02,
 5.87562528e+04])
n_particles = 250
dimensions = 5
noise_scale = 0.00  # adjust as needed
init_pos = np.tile(base_pos, (n_particles, 1)) + np.random.uniform(-noise_scale, noise_scale, size=(n_particles, dimensions))


mlp_optimizer=ps.single.GlobalBestPSO(n_particles=500,dimensions=5,
                                      options={'c1':0.5,'c2':0.3,'w':0.9},
                                      bounds=mlp_bounds, init_pos=init_pos)
mlp_best_score,mlp_best_params=mlp_optimizer.optimize(mlp_pso_objective,iters=200,n_processes=n_cpu)
mlp_params_formatted={
    'hidden_layer_sizes':int(round(mlp_best_params[0])),
    'alpha':float(mlp_best_params[1]),
    'max_iter':int(round(mlp_best_params[2])),
    'tol':float(mlp_best_params[3]),
    'max_fun':int(round(mlp_best_params[4]))
}
print("Best MLP params:",mlp_params_formatted)
model_ann=ANN(hidden_layer_sizes=(mlp_params_formatted['hidden_layer_sizes'],),
                                   alpha=mlp_params_formatted['alpha'],
                                   max_iter=mlp_params_formatted['max_iter'],
                                   tol=mlp_params_formatted['tol'],
                                   solver='lbfgs',
                                   max_fun=mlp_params_formatted['max_fun'],
                                   random_state=42)
model_ann.fit(X_train_scaled,y_train)
y_train_pred_ann=model_ann.predict(X_train_scaled)
y_test_pred_ann=model_ann.predict(X_test_scaled)

# =========================================
# 7. Visualization & Save
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error

# Selected features
feature_names = ['VSA_EState3', 'AXp-0dv', 'GATS1i']

# Train dataframe
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df['pLD50_true'] = y_train.values
train_df['pLD50_pred_rf'] = y_train_pred_rf
train_df['pLD50_pred_svr'] = y_train_pred_svr
train_df['pLD50_pred_gbr'] = y_train_pred_gbr
train_df['pLD50_pred_xgb'] = y_train_pred_xgb
train_df['pLD50_pred_ann'] = y_train_pred_ann
train_df['dataset'] = 'train'

# Test dataframe
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['pLD50_true'] = y_test.values
test_df['pLD50_pred_rf'] = y_test_pred_rf
test_df['pLD50_pred_svr'] = y_test_pred_svr
test_df['pLD50_pred_gbr'] = y_test_pred_gbr
test_df['pLD50_pred_xgb'] = y_test_pred_xgb
test_df['pLD50_pred_ann'] = y_test_pred_ann
test_df['dataset'] = 'test'

# Add compound name/index as a column
train_df['Compound'] = X_train.index
test_df['Compound'] = X_test.index

# Merge train and test
all_df = pd.concat([train_df, test_df], ignore_index=False)
ordered_cols = ['Compound']+feature_names + [
    'pLD50_true',
    'pLD50_pred_rf',
    'pLD50_pred_svr',
    'pLD50_pred_gbr',
    'pLD50_pred_xgb',
    'pLD50_pred_ann',
    'dataset'
]
all_df = all_df[ordered_cols]
all_df.to_csv('all_data_mcf7-121025.csv', index=False)

# ----------------------
# Plotting
# ----------------------
def calc_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
    r2 = r2_score(y_true_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_true_test, y_test_pred))
    return r2, rmse_train, rmse_test

fig, axes = plt.subplots(1, 5, figsize=(16, 4))

models = ['RF', 'SVR', 'GBR', 'XGB', 'ANN']
y_train_preds = [y_train_pred_rf, y_train_pred_svr, y_train_pred_gbr, y_train_pred_xgb, y_train_pred_ann]
y_test_preds = [y_test_pred_rf, y_test_pred_svr, y_test_pred_gbr, y_test_pred_xgb, y_test_pred_ann]

for col, (model, Ytrain_pred, Ytest_pred) in enumerate(zip(models, y_train_preds, y_test_preds)):
    ax = axes[col]
    y_train_true = y_train.values
    y_test_true = y_test.values
    y_train_pred = Ytrain_pred
    y_test_pred = Ytest_pred

    metrics = calc_metrics(y_train_true, y_train_pred, y_test_true, y_test_pred)

    ax.scatter(y_train_true, y_train_pred, color='blue', marker='o', edgecolor='k',
               s=80, alpha=0.9, label='Train')
    ax.scatter(y_test_true, y_test_pred, color='red', marker='s', edgecolor='k',
               s=80, alpha=0.9, label='Test')

    min_val, max_val = 3.5, 5.0
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.set_title(model, fontsize=14)
    ax.set_xlabel('Experimental pLD$_{50}$', fontsize=12)
    ax.set_ylabel('Predicted pLD$_{50}$', fontsize=12)

    inset_text = f"R² train={metrics[0]:.3f}\nRMSE train={metrics[1]:.3f}\nRMSE test={metrics[2]:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, inset_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plot_expvspred_mcf7-121025.tif', dpi=300, format='tif')
plt.show()


# =========================================
# Extra validation: LOOCV, LMOCV, Y-Randomization
# =========================================
from sklearn.model_selection import LeaveOneOut, RepeatedKFold
from sklearn.base import clone
from sklearn.utils import shuffle

def run_loocv(model, X, y, scaled=False):
    if scaled:
        X = StandardScaler().fit_transform(X)
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y.iloc[train_idx])
        pred = model_clone.predict(X[test_idx])
        y_true.append(y.iloc[test_idx].values)
        y_pred.append(pred)
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return {
        'R2': r2_score(y_true, y_pred, multioutput='uniform_average'),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }

def run_lmocv(model, X, y, n_splits=4, n_repeats=20, scaled=False):
    if scaled:
        X = StandardScaler().fit_transform(X)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores_r2, scores_rmse = [], []
    for train_idx, test_idx in rkf.split(X):
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y.iloc[train_idx])
        pred = model_clone.predict(X[test_idx])
        scores_r2.append(r2_score(y.iloc[test_idx], pred, multioutput='uniform_average'))
        scores_rmse.append(np.sqrt(mean_squared_error(y.iloc[test_idx], pred)))
    return {
        'R2_mean': np.mean(scores_r2), 'R2_std': np.std(scores_r2),
        'RMSE_mean': np.mean(scores_rmse), 'RMSE_std': np.std(scores_rmse)
    }

# =========================================
# Run validation for all models
# =========================================
models_dict = {
    'RF': (model_rf, X_train.values, y_train, False),
    'SVR': (model_svr, X_train_scaled, y_train, True),
    'GBR': (model_gbr, X_train.values, y_train, False),
    'XGB': (model_xgb, X_train.values, y_train, False),
    'ANN': (model_ann, X_train_scaled, y_train, True),
}

results = []
for name, (mdl, Xtr, ytr, scaled) in models_dict.items():
    loocv_res = run_loocv(mdl, Xtr, ytr, scaled=scaled)
    lmocv_res = run_lmocv(mdl, Xtr, ytr, n_splits=4, n_repeats=20, scaled=scaled)
    
    results.append({
        'Model': name,
        'LOOCV_R2': loocv_res['R2'],
        'LOOCV_RMSE': loocv_res['RMSE'],
        'LMOCV_R2_mean': lmocv_res['R2_mean'],
        'LMOCV_R2_std': lmocv_res['R2_std'],
        'LMOCV_RMSE_mean': lmocv_res['RMSE_mean'],
        'LMOCV_RMSE_std': lmocv_res['RMSE_std']
    })

results_df = pd.DataFrame(results)
results_df.to_csv("validation_results_loocv_lmocv-mcf7-121025.csv", index=False)
print(results_df)

