import os
from sklearn.linear_model import QuantileRegressor
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm
import pandas as pd
import csv
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, SGDRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit, HuberRegressor, ElasticNet, LinearRegression, LassoLars, ARDRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
#from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(12345)
os.makedirs("DataOut", exist_ok=True)
removeLowVaryingGenes = 0.2
printOutput = False

CTRPv2_AUC = pd.read_csv("./DataIn/CTRPv2/CTRPv2_AUC_clean.txt", sep="\t")
CTRPv2_RNAseq_TPM = pd.read_csv("./DataIn/CTRPv2/CTRPv2_RNAseq_TPM_clean.txt", sep="\t")
input_file = "Output/ALL_DRUGS_ALL_MODELS_RMSE_greater_than_1.5.csv"  # Replace with your input file path

df = pd.read_csv(input_file)

# Step 2: Filter the DataFrame for drugs with RMSE greater than 1
filtered_df = df[df['RMSE'] >= 1.5]

def do_variable_selection(expr_mat, remove_low_varying_genes):
    vars = np.apply_along_axis(np.var, axis=1, arr=expr_mat)
    keep_rows = np.argsort(vars)[::-1][:int(expr_mat.shape[0] * (1 - remove_low_varying_genes))]
    return keep_rows
def plot_gene_expression_histograms(X_before, X_after):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(data=X_before, ax=axes[0], kde=True)
    axes[0].set_title('Before Scaling')
    sns.histplot(data=X_after, ax=axes[1], kde=True)
    axes[1].set_title('After Scaling')
    plt.show()

def print_feature_importance_with_names(feature_importance, gene_names):
    pc1_importance = feature_importance[0]
    sorted_indices = pc1_importance.argsort()[::-1]
    top_20_indices = sorted_indices[:20]
    top_genes = [gene_names[idx] for idx in top_20_indices]
    return top_genes



def calc_phenotype(drug,training_expr_data_o, training_ptype,remove_low_varying_genes=0.2,print_output=True):
    if not isinstance(training_expr_data_o, pd.DataFrame):
        raise ValueError("ERROR: 'test_expr_data' and 'training_expr_data' must be DataFrames.")
    if not isinstance(training_ptype, pd.Series):
        raise ValueError("ERROR: 'training_ptype' must be a pandas Series.")
    if training_expr_data_o.shape[1] != len(training_ptype):
        raise ValueError("The training phenotype must be of the same length as the number of columns of the training expression matrix.")

    # Assuming training_expr_data is your DataFrame containing gene expression data

    training_expr_data=training_expr_data_o
    if 0 < remove_low_varying_genes < 1:
        evaluabe_genes = training_expr_data.index
        keep_rows_train = do_variable_selection(training_expr_data.loc[evaluabe_genes, :],
                                                remove_low_varying_genes=remove_low_varying_genes)
        keep_rows = keep_rows_train
        number_genes_removed = training_expr_data.shape[0] - len(keep_rows)
        if print_output:
            print(f"\n{number_genes_removed} low variability genes filtered.")
    else:
        keep_rows = np.arange(training_expr_data.shape[0])
    if print_output:
        print("\nFitting Ridge Regression model... ", end="")

    if printOutput:
        print("--------------  training_expr_data  ------------------")
        print(training_expr_data.shape)
        print(training_expr_data.head())


    hom_data_train_subset = training_expr_data.iloc[keep_rows, :].T
    training_ptype_reshaped = training_ptype.values.reshape(-1, 1)
    training_ptype_df = pd.DataFrame(training_ptype_reshaped, index=training_ptype.index, columns=['Phenotype'])
    combined_data = pd.concat([hom_data_train_subset,training_ptype_df], axis=1)
    training_ptype= training_ptype.values.reshape(-1, 1)
    if printOutput:
        print("------------------  training_ptype   ---------------------")
        print(training_ptype.shape)
        print(type(training_ptype))

    # normalized_features= (hom_data_train_subset- hom_data_train_subset.min()) / (hom_data_train_subset.max() - hom_data_train_subset.min())
    ################## Regularization ##############################
    X_train_, X_test_, y_train, y_test = train_test_split(hom_data_train_subset, training_ptype, test_size=0.3, random_state=42)
    pca = PCA(n_components=0.90)  # Retain 90% of variance
    X_train_pca= pca.fit_transform(X_train_)
    X_test_pca= pca.transform(X_test_)
    gene_names = combined_data.drop(columns=['Phenotype']).columns.tolist()
    feature_importance = pca.components_
    print("Number of selected features:", X_test_pca.shape[1])
    # To get the list of top genes
    top_genes_pc1 =print_feature_importance_with_names(feature_importance, gene_names)
    print("List of top genes in PC1:", top_genes_pc1)
    scaler_zscore = StandardScaler()
    X_train = scaler_zscore.fit_transform(X_train_pca)
    X_test = scaler_zscore.transform(X_test_pca)

    # param_grids = {
    #     'DecisionTreeRegressor': {'max_depth': [None, 3, 5, 7]},
    #     'Ridge': {'alpha': np.logspace(-3, 3, 7)},
    #     'LinearRegression': {'fit_intercept': [True, False]},
    #     'SVR': {'C': np.logspace(-3, 3, 7), 'kernel': ['linear', 'rbf', 'poly']},
    #     'GradientBoostingRegressor': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1]},
    #     'CatBoostRegressor': {'depth': [4, 6, 8], 'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1]},
    #     'XGBRegressor': {'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.05, 0.1]},
    #     'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 9]},
    #     'ElasticNet': {'alpha': np.logspace(-3, 3, 7), 'l1_ratio': [0.1, 0.5, 0.9]},
    #     'RandomForestRegressor': {'n_estimators': [100, 200, 300], 'max_depth': [None, 3, 5, 7]}
    # }
    # param_grids = {
    #     'BayesianRidge': {'alpha_1': [1e-6, 1e-5, 1e-4], 'alpha_2': [1e-6, 1e-5, 1e-4], 'lambda_1': [1e-6, 1e-5, 1e-4], 'lambda_2': [1e-6, 1e-5, 1e-4]},
    #     'Ridge': {'alpha': [0.01, 0.1, 1.0]},
    #     'AdaBoostRegressor': {'n_estimators': [100, 200, 300]},
    #     'ExtraTreesRegressor': {'n_estimators': [200, 400, 600]},
    #     'SGDRegressor': {'alpha': [0.0001, 0.001, 0.01]},
    #     'LassoLars': {'alpha': [0.01, 0.1, 1.0]},
    #     'PassiveAggressiveRegressor': {'C': [0.01, 0.1, 1.0]},
    #     'OrthogonalMatchingPursuit': {'n_nonzero_coefs': [5, 10, 15]},
    #     'HuberRegressor': {'epsilon': [1.0, 1.5, 2.0]},
    #     'PLSRegression': {'n_components': [2, 4, 6]},
    #     'HistGradientBoostingRegressor': {'max_iter': [200, 400, 600]},
    #     'ARDRegression': {'alpha_1': [1e-6, 1e-5, 1e-4], 'alpha_2': [1e-6, 1e-5, 1e-4]},
    #     'QuantileRegressor': {'alpha': [0.1, 0.5, 1.0]},
    #     'LinearRegression': {'fit_intercept': [True, False]},
    #     'SVR': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
    #     'GradientBoostingRegressor': {'n_estimators': [200, 400, 600], 'learning_rate': [0.01, 0.05, 0.1]},
    #     'CatBoostRegressor': {'iterations': [200, 400, 600]},
    #     'XGBRegressor': {'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.05, 0.1]},
    #     'KNeighborsRegressor': {'n_neighbors': [3, 5, 7]},
    #     'ElasticNet': {'alpha': [0.01, 0.1, 1.0]},
    #     'DecisionTreeRegressor': {'max_depth': [None, 3, 6]},
    #     'RandomForestRegressor': {'n_estimators': [200, 400, 600]},
    #     'LGBMRegressor': {'num_leaves': [31, 50, 70], 'n_estimators': [200, 400, 600]},
    # }


    param_grids = {
        'BayesianRidge': {
            'alpha_1': np.logspace(-6, -4, 3),
            'alpha_2': np.logspace(-6, -4, 3),
            'lambda_1': np.logspace(-6, -4, 3),
            'lambda_2': np.logspace(-6, -4, 3)
        },
        'Ridge': {
            'alpha': np.logspace(-2, 1, 4)
        },
        'AdaBoostRegressor': {
            'n_estimators': [100, 200, 300],
            'learning_rate': np.logspace(-2, 0, 3)
        },
        'ExtraTreesRegressor': {
            'n_estimators': [200, 400, 600],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'SGDRegressor': {
            'alpha': np.logspace(-4, -2, 3),
            'penalty': ['l2', 'l1', 'elasticnet']
        },
        'LassoLars': {
            'alpha': np.logspace(-2, 1, 4)
        },
        'PassiveAggressiveRegressor': {
            'C': np.logspace(-2, 1, 4)
        },
        'OrthogonalMatchingPursuit': {
            'n_nonzero_coefs': [5, 10, 15]
        },
        'HuberRegressor': {
            'epsilon': [1.35, 1.5, 1.75],
            'alpha': np.logspace(-4, -2, 3)
        },
        'PLSRegression': {
            'n_components': [2, 4, 6, 8]
        },
        'HistGradientBoostingRegressor': {
            'max_iter': [200, 400, 600],
            'learning_rate': np.logspace(-2, -1, 3)
        },
        'ARDRegression': {
            'alpha_1': np.logspace(-6, -4, 3),
            'alpha_2': np.logspace(-6, -4, 3)
        },
        'QuantileRegressor': {
            'alpha': np.logspace(-1, 0, 3)
        },
        'LinearRegression': {
            'fit_intercept': [True, False]
        },
        'SVR': {
            'C': np.logspace(-1, 1, 3),
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'GradientBoostingRegressor': {
            'n_estimators': [200, 400, 600],
            'learning_rate': np.logspace(-2, -1, 3),
            'max_depth': [3, 4, 5]
        },
        'CatBoostRegressor': {
            'iterations': [200, 400, 600],
            'depth': [6, 8, 10],
            'learning_rate': np.logspace(-2, -1, 3)
        },
        'XGBRegressor': {
            'max_depth': [3, 5, 7],
            'learning_rate': np.logspace(-2, -1, 3),
            'n_estimators': [200, 400, 600]
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'ElasticNet': {
            'alpha': np.logspace(-2, 1, 4),
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'DecisionTreeRegressor': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForestRegressor': {
            'n_estimators': [200, 400, 600],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20]
        },
        'LGBMRegressor': {
            'num_leaves': [31, 50, 70],
            'n_estimators': [200, 400, 600],
            'learning_rate': np.logspace(-2, -1, 3)
        }
    }



# Define a function to perform grid search
    def perform_grid_search(model_name, model, param_grid, X_train, y_train):
        pipeline_steps = [('scaler', StandardScaler()), (model_name, model)]

        if model_name == 'StackingRegressor':
            pipeline_steps[-1] = (model_name, StackingRegressor(
                estimators=[('ridge', Ridge()), ('lasso', Lasso()), ('svr', SVR())],
                final_estimator=Ridge()
            ))

        pipeline = Pipeline(pipeline_steps)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={model_name + '__' + k: v for k, v in param_grid.items()},
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_



    models = {
        'BayesianRidge': BayesianRidge(),
        'Ridge': Ridge(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'SGDRegressor': SGDRegressor(),
        'LassoLars': LassoLars(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'HuberRegressor': HuberRegressor(),
        'PLSRegression': PLSRegression(),
        #'StackingRegressor': StackingRegressor(estimators=[('ridge', Ridge()), ('lasso', Lasso()), ('svr', SVR())]),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
        'ARDRegression': ARDRegression(),
        'QuantileRegressor': QuantileRegressor(),
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'CatBoostRegressor': cb.CatBoostRegressor(silent=True),
        'XGBRegressor': xgb.XGBRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'ElasticNet': ElasticNet(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'LGBMRegressor': lgbm.LGBMRegressor(),
    }

    # Perform hyperparameter tuning for each model
    results = {}
    for model_name, model in models.items():
        print(f"Performing grid search for {model_name}...")
        best_estimator, best_params, best_score = perform_grid_search(model_name, model, param_grids[model_name], X_train, y_train.ravel())
        results[model_name] = {
            'best_estimator': best_estimator,
            'best_params': best_params,
            'best_score': best_score
        }

    best_model_params = None
    best_model_name = None
    best_rmse = float('inf')
    best_y_pred = None
    best_r2 = None

    # Train and evaluate each model with the best hyperparameters
    for model_name, result in results.items():
        model_class = models[model_name].__class__
        cleaned_params = {k.split('__')[-1]: v for k, v in result['best_params'].items()}
        if best_model_name == 'StackingRegressor':
            model = StackingRegressor(estimators=[('ridge', Ridge()), ('lasso', Lasso()), ('svr', SVR())], final_estimator=Ridge(**cleaned_params))
        else:
            model = model_class(**cleaned_params)
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"{model_name}: RMSE = {rmse:.2f}, R2 = {r2:.2f}")

        # Check if this model has the best RMSE so far
        if rmse < best_rmse:
            best_model_name = model_name
            best_rmse = rmse
            best_y_pred = y_pred
            best_model_params = cleaned_params
            best_r2 = r2
    # Store actual and predicted values for the best model
    if best_model_name is not None:
        print(f"best_model_name:{best_model_name}")
        cell_lines = X_test_.index

        with open("./Output/11.ALL_More_than_1.5_actual_predicted.csv", 'a', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            if file.tell()==0:
                writer.writerow(["Drug", "Cell_Line", "Actual", "Predicted", "Difference"])
            for cell_line,actual, predicted in zip(cell_lines,y_test, best_y_pred):
                diff = actual - predicted
                print("{}, {}, {}, {},{}".format(drug,cell_line,actual, predicted, diff))
                writer.writerow([drug,cell_line,actual, predicted, diff])




    print("Best Model:",best_model_name,best_rmse,best_r2)
    return best_model_name, best_model_params,best_rmse,best_r2,top_genes_pc1

CTRPDrugPredictions = {}
output_file = "./Output/11.ALL_More_than_1.5_Best_MODEL.csv"
file_exists = os.path.isfile(output_file)
for index, drug in filtered_df['Unnamed: 0'].items():
    print(f"Drug at index {index}: {drug}")
    temp = CTRPv2_AUC[CTRPv2_AUC['cpd_name'] == drug]
    AUCs = temp['Avg_AUC']
    AUCs.index = temp['CCL_Name']
    column_names_first_part = [name.split('_')[0] for name in CTRPv2_RNAseq_TPM.columns]
    # Check for common values with AUCs.index
    common_cell_lines = [name for name in column_names_first_part if name in AUCs.index]
    # common_cell_lines = CTRPv2_RNAseq_TPM.columns[CTRPv2_RNAseq_TPM.columns.isin(AUCs.index)]

    AUCs_ord = AUCs[common_cell_lines]
    AUCs_ord = AUCs_ord[~AUCs_ord.index.duplicated(keep='first')]

    genes=CTRPv2_RNAseq_TPM["Unnamed: 0"]
    train_data_ord = CTRPv2_RNAseq_TPM.loc[:, common_cell_lines]
    train_data_ord.index=genes.values
    train_data_ord = train_data_ord[(train_data_ord.sum(axis=1) != 0)]
    print("train_data_ord ")
    print(train_data_ord.shape)
    result = calc_phenotype(drug=drug,training_expr_data_o=train_data_ord,
                            training_ptype=AUCs_ord,
                            remove_low_varying_genes=removeLowVaryingGenes,
                            print_output=printOutput)
    result_df = pd.DataFrame([result], columns=["Model", "Best_model_params", "RMSE", "R2", "Top_20_genes_pc1"], index=[drug])

    # Write the result to the file, appending if the file exists
    if not file_exists:
        result_df.to_csv(output_file, index=True, mode='w', header=True)
        file_exists = True
    else:
        result_df.to_csv(output_file, index=True, mode='a', header=False)


    print(f"Results for {drug} saved.")

#     CTRPDrugPredictions[i] = result
#
# CTRPDrugPredictions_df = pd.DataFrame((CTRPDrugPredictions))
# print(CTRPDrugPredictions_df)
# fifty_drugs=possibleDrugs
# CTRPDrugPredictions_df.columns = fifty_drugs
# CTRPDrugPredictions_df.index = ["Model","Best_model_params","RMSE", "R2","Top_20_genes_pc1"]
# CTRPDrugPredictions_df_T = CTRPDrugPredictions_df.T
# print(CTRPDrugPredictions_df_T.shape)
# print(CTRPDrugPredictions_df_T.head())
# matrix_output_name = f"./Output/11.ALL_Best_MODEL.csv"
# CTRPDrugPredictions_df_T.to_csv(matrix_output_name, index=True)
