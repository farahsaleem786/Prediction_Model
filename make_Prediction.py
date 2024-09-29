import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
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
import os
from sklearn.linear_model import QuantileRegressor
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm

import csv

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

# Ensure reproducibility
np.random.seed(12345)


# Constants
removeLowVaryingGenes = 0.2
printOutput = False

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, SGDRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit, HuberRegressor, ElasticNet, LinearRegression, LassoLars, ARDRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin


# Load data
CTRPv2_AUC = pd.read_csv("./DataIn/CTRPv2/CTRPv2_AUC_clean.txt", sep="\t")
CTRPv2_RNAseq_TPM = pd.read_csv("./DataIn/CTRPv2/CTRPv2_RNAseq_TPM_clean.txt", sep="\t")

best_params_df = pd.read_csv("./Output/ALL_DRUGS_ALL_MODELS.csv")

cpd_column = CTRPv2_AUC.get('cpd_name')
unique_drugs = cpd_column.dropna().unique()
possibleDrugs=unique_drugs
# possibleDrugs = np.unique(CTRPv2_AUC['cpd_name'])
def display_sample_file():
    st.subheader("Sample TSV File Format:")
    sample_data = {
        'gene_id': ["ENSG00000223972.4", 'ENSG00000223962.4', 'ENSG00000223972.9'],
        'gene_name': ['GeneA', 'GeneB', 'GeneC'],
        'Sample1': [10, 20, 30],
        'Sample2': [15, 25, 35],
        'Sample3': [18, 28, 38]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
    st.write("Upload a file with similar structure to proceed.")
# Function to convert RPKM to TPM
def RPKM_to_TPM(sample_RPKMs):
    return (sample_RPKMs / sample_RPKMs.sum()) * 10**6

# Function to preprocess uploaded file and return preprocessed data
def preprocess_uploaded_file(uploaded_file):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.tsv'):
                CCLE_RNAseq = pd.read_csv(uploaded_file, sep='\t')
            else:
                st.error("Invalid file type. Please upload a TSV file.")
                return None

            # Check if required columns are present
            if 'gene_name' not in CCLE_RNAseq.columns or len(CCLE_RNAseq.columns) <1:
                st.error("Invalid file format. Please make sure the file contains 'gene_id',gene_name' and sample columns.")
                return None

            st.write("Uploaded File:")
            st.dataframe(CCLE_RNAseq)

            convert_to_tpm = st.checkbox("Convert RPKM to TPM")

            # Convert RPKM to TPM if selected
            if convert_to_tpm:
                # Exclude duplicate genes from the filtered testing data
                CCLE_RNAseq_filtered = CCLE_RNAseq[~CCLE_RNAseq['gene_name'].duplicated()]
                CCLE_RNAseq_filtered.reset_index(drop=True, inplace=True)

                # Convert RPKM to TPM for each sample
                RPKM_values = CCLE_RNAseq_filtered.iloc[:,:].values
                TPM_values = np.apply_along_axis(RPKM_to_TPM, axis=0, arr=RPKM_values)

                # Create a DataFrame with TPM values for the filtered testing data
                test_data = pd.DataFrame(TPM_values, columns=CCLE_RNAseq_filtered.columns[2:])
                test_data.index = CCLE_RNAseq_filtered['gene_name']
                test_data.index.name = None
                return test_data

            else:
                # Use the original data without converting RPKM to TPM
                test_data = CCLE_RNAseq.iloc[:, 1:]  # Assuming the data starts from the second column
                test_data.index = CCLE_RNAseq['gene_name']
                test_data.index.name = None
                # Remove rows with all zeros
                test_data = test_data[test_data.sum(axis=1) != 0]
                return test_data

            # st.write("Preprocessed Data:")
            # st.dataframe(test_data)



    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
# Function to perform variable selection
def do_variable_selection(expr_mat, remove_low_varying_genes):
    vars = np.apply_along_axis(np.var, axis=1, arr=expr_mat)
    keep_rows = np.argsort(vars)[::-1][:int(expr_mat.shape[0] * (1 - remove_low_varying_genes))]
    return keep_rows

# Function to print top genes from feature importance
def print_feature_importance_with_names(feature_importance, gene_names):
    pc1_importance = feature_importance[0]
    sorted_indices = pc1_importance.argsort()[::-1]
    top_20_indices = sorted_indices[:20]
    top_genes = [gene_names[idx] for idx in top_20_indices]
    return top_genes

# Function to calculate phenotype
def calc_phenotype(drug, training_expr_data_o, test_data, training_ptype, best_model_params, model_name, remove_low_varying_genes=0.2, print_output=True):
    if not isinstance(training_expr_data_o, pd.DataFrame):
        raise ValueError("ERROR: 'training_expr_data_o' must be a DataFrame.")
    if not isinstance(training_ptype, pd.Series):
        raise ValueError("ERROR: 'training_ptype' must be a pandas Series.")
    if training_expr_data_o.shape[1] != len(training_ptype):
        raise ValueError("The training phenotype must be of the same length as the number of columns of the training expression matrix.")

    training_expr_data = training_expr_data_o.copy()

    # Perform gene selection based on variability
    if 0 < remove_low_varying_genes < 1:
        evaluable_genes = training_expr_data.index
        keep_rows_train = do_variable_selection(training_expr_data.loc[evaluable_genes, :], remove_low_varying_genes=remove_low_varying_genes)
        keep_rows = keep_rows_train
        number_genes_removed = training_expr_data.shape[0] - len(keep_rows)
        if print_output:
            st.write(f"\n{number_genes_removed} low variability genes filtered.")
    else:
        keep_rows = np.arange(training_expr_data.shape[0])

    # Subset the training data with selected genes
    hom_data_train_subset = training_expr_data.iloc[keep_rows, :].T
    print("hom_data_train_subset")
    print(hom_data_train_subset.head())
    # Reshape training phenotype data
    training_ptype_reshaped = training_ptype.values.reshape(-1, 1)

    # Create DataFrame for training phenotype
    training_ptype_df = pd.DataFrame(training_ptype_reshaped, index=training_ptype.index, columns=['Phenotype'])

    # Combine gene expression data with phenotype data
    combined_data = pd.concat([hom_data_train_subset, training_ptype_df], axis=1)

    # Prepare data for modeling
    print("test_data")
    print(test_data.T.head())
    print(test_data.T.shape)
    print(test_data.index)
    X_train_, X_test_, y_train, y_test = train_test_split(hom_data_train_subset, training_ptype, test_size=0.2, random_state=42)
    test_data=test_data.T
    common_columns = X_train_.columns.intersection(test_data.columns)
    print("common_columns")
    print(common_columns)
    # Filter test_data to include only common columns and make a copy to avoid chained indexing
    X_test_aligned = test_data[common_columns].copy()

    print("X_test_aligned = test_data[common_columns].copy()")
    print(X_test_aligned.head())
    print("Number of common columns:", len(common_columns))
    print("Number of columns in X_test_aligned after filtering:", len(X_test_aligned.columns))

    # If they still do not match, identify the discrepancies
    missing_in_X_test_aligned = set(common_columns) - set(X_test_aligned.columns)
    extra_in_X_test_aligned = set(X_test_aligned.columns) - set(common_columns)
    print("Extra columns in X_test_aligned:", extra_in_X_test_aligned)
    print("Columns missing in X_test_aligned:", missing_in_X_test_aligned)
    # Check for duplicated columns in X_test_aligned
    duplicated_columns = X_test_aligned.columns.duplicated()

    # Get the count of duplicated columns
    num_duplicated_columns = duplicated_columns.sum()

    # Print the number of duplicated columns
    print(f"Number of duplicated columns in X_test_aligned: {num_duplicated_columns}")

    # Optionally, you can get the names of the duplicated columns
    duplicated_column_names = X_test_aligned.columns[duplicated_columns]
    print("Duplicated column names in X_test_aligned:")
    print(duplicated_column_names)
    X_test_aligned = X_test_aligned.loc[:, ~duplicated_columns]

# Verify the number of columns after dropping duplicates
    print(f"Number of columns in X_test_aligned after removing duplicates: {X_test_aligned.shape[1]}")
    # if missing_in_X_test_aligned:
    #     print("Columns missing in X_test_aligned:", missing_in_X_test_aligned)
    # if extra_in_X_test_aligned:
    #     print("Extra columns in X_test_aligned:", extra_in_X_test_aligned)
    X_test_aligned = X_test_aligned.drop(columns=extra_in_X_test_aligned)
    print("X_test_aligned = X_test_aligned.drop(columns=extra_in_X_test_aligned)")
    print(X_test_aligned.head())

    # Add missing columns in X_test_aligned with the same names as X_train_
    missing_columns = X_train_.columns.difference(X_test_aligned.columns)
    if not missing_columns.empty:
        # Create a DataFrame with zeros for missing columns
        print("******************************X_test_aligned********************")
        print(X_test_aligned.index)
        missing_df = pd.DataFrame(0, index=X_test_aligned.index, columns=missing_columns)
        print("****************************** Missing_df  ********************")
        print(missing_df.head())
        # Concatenate X_test_aligned with the missing_df along axis=1 to add missing columns at once
        X_test_aligned = pd.concat([X_test_aligned, missing_df], axis=1)

        print("X_test_aligned = pd.concat([X_test_aligned, missing_df], axis=1)")
        print(X_test_aligned.head())

    # Reorder columns to match X_train_ if needed (optional)
    X_test_aligned = X_test_aligned[X_train_.columns]
    print("X_test_aligned = X_test_aligned[X_train_.columns]")
    print(X_test_aligned.head())
    # Perform PCA
    pca = PCA(n_components=0.90)  # Retain 90% of variance
    X_train_pca = pca.fit_transform(X_train_)
    print("X_train_")
    print(X_train_.head())
    print("X_test_aligned")
    print(X_test_aligned.head())
    X_test_pca = pca.transform(X_test_aligned)
    gene_names = combined_data.drop(columns=['Phenotype']).columns.tolist()

    # Get feature importance from PCA
    feature_importance = pca.components_

    # Print top genes in PC1
    top_genes_pc1 = print_feature_importance_with_names(feature_importance, gene_names)
    # st.write("Top genes in PC1:", top_genes_pc1)

    scaler_zscore = StandardScaler()
    X_train = scaler_zscore.fit_transform(X_train_pca)
    X_test = scaler_zscore.transform(X_test_pca)

    # Initialize the model with the best parameters
    best_params = best_model_params
    model = globals()[model_name](**best_params)

    # Fit the model
    model.fit(X_train, y_train.ravel())

    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = pd.DataFrame(columns=['Drug'] + list(test_data.index))

    # Predict on test set
    y_pred = model.predict(X_test)
    print("y_pred")
    print(y_pred)
    return y_pred

# Function to loop through drugs and get predictions
def loop_for_drugs(test_data):
    print("**********************************************test_data.index*******************")
    print(test_data.index)
    predictions = pd.DataFrame(columns=['Drug'] + list(test_data.columns))
    status_text = st.empty()
    table_placeholder = st.empty()

    # Loop through each drug and perform predictions
    for i, drug in enumerate(possibleDrugs):
        status_text.write(f"Predicting for Drug {i + 1}/{len(possibleDrugs)}: {drug}")
        temp = CTRPv2_AUC[CTRPv2_AUC['cpd_name'] == drug]
        AUCs = temp['Avg_AUC']
        AUCs.index = temp['CCL_Name']
        column_names_first_part = [name.split('_')[0] for name in CTRPv2_RNAseq_TPM.columns]
        common_cell_lines = [name for name in column_names_first_part if name in AUCs.index]

        AUCs_ord = AUCs[common_cell_lines]
        AUCs_ord = AUCs_ord[~AUCs_ord.index.duplicated(keep='first')]

        genes = CTRPv2_RNAseq_TPM["Unnamed: 0"]
        # print("test_data")
        # print(test_data)
        # genes_test=test_data["Unnamed: 0"]
        # test_data.index=genes_test.values
        # Load data subset from CTRPv2_RNAseq_TPM
        train_data_ord = CTRPv2_RNAseq_TPM.loc[:, common_cell_lines].astype(np.float32)

        train_data_ord.index = genes.values
        train_data_ord = train_data_ord[(train_data_ord.sum(axis=1) != 0)]

        # Load best parameters for the current drug from the CSV
        best_model_params = best_params_df.loc[best_params_df['Unnamed: 0'] == drug, 'Best_model_params'].iloc[0]
        best_model_params = eval(best_model_params)  # Convert string representation to dictionary
        model_name = best_params_df.loc[best_params_df['Unnamed: 0'] == drug, 'Model'].iloc[0]

        # Calculate phenotype and get predictions for the current drug
        y_pred = calc_phenotype(
            drug=drug,
            training_expr_data_o=train_data_ord,
            test_data=test_data,
            training_ptype=AUCs_ord,
            model_name=model_name,
            best_model_params=best_model_params,
            remove_low_varying_genes=removeLowVaryingGenes,
            print_output=printOutput
        )
        print("predictions before ")
        print(predictions)
        predictions.loc[len(predictions)] = [drug] + list(y_pred)
        print("predictions after")
        print(predictions)
        table_placeholder.dataframe(predictions)
    return predictions

def new_data():
    st.title("Drug Response Prediction App")
    display_sample_file()
    uploaded_file = st.file_uploader("Choose a file to analyze", type=["tsv"])
    print("uploaded_file")
    print(uploaded_file)
    if uploaded_file is not None:
        test_data = preprocess_uploaded_file(uploaded_file)

        if test_data is not None:

            st.write("Preprocessed Data:")
            st.dataframe(test_data)

            # Show button for making predictions
            if st.button("Make Predictions"):
                st.write("Generating predictions, please wait...")

                # Generate predictions for all drugs
                predictions = loop_for_drugs(test_data)
                st.write("Predictions Complete:")
                st.dataframe(predictions)
    elif uploaded_file is None:
        st.write("Please upload a  TSV file to proceed.")

