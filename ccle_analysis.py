import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
# Function to read and preprocess the data


def RPKM_to_TPM(rpkm):
    """Convert RPKM values to TPM"""
    rpkm_sum = np.sum(rpkm)
    tpm = (rpkm / rpkm_sum) * 1e6
    return tpm
def preprocess_data(file_path):
    # Read the file, skipping rows with errors
    df = pd.read_csv(file_path, sep="\t", skiprows=2)

    # Remove duplicate rows based on the 'Description' column
    initial_row_count = df.shape[0]
    df = df.drop_duplicates(subset="Description")
    duplicate_rows_removed = initial_row_count - df.shape[0]

    # Drop rows containing missing values
    df = df.dropna()
    missing_rows_dropped = initial_row_count - duplicate_rows_removed - df.shape[0]

    # Drop unnecessary columns and set 'Description' as the index
    df.drop(columns=["Name"], inplace=True)
    df.set_index("Description", inplace=True)

    # Extract cell line information
    cell_lines_information = list(df.columns)

    # Extract short-standard cell line names
    cell_lines_names = [col.split('_')[0] for col in df.columns]

    # Ensure unique column names
    unique_cell_lines_names = []
    counts = {}
    for name in cell_lines_names:
        if name in counts:
            counts[name] += 1
            unique_cell_lines_names.append(f"{name}_{counts[name]}")
        else:
            counts[name] = 0
            unique_cell_lines_names.append(name)
    df.columns = unique_cell_lines_names

    # Extract short-standard cell line tissues
    cell_lines_tissues = [col.split('_')[1] for col in cell_lines_information]

    # Count unique tissue names and their occurrences
    unique_tissue_names = pd.Series(cell_lines_tissues)
    tissue_occurrences = unique_tissue_names.value_counts()

    # Convert to NumPy array
    data_array = df.values

    # Convert RPKM to TPM
    df_tpm = np.empty_like(df)
    df_tpm[:] = np.nan
    df_tpm = pd.DataFrame(df_tpm, index=df.index, columns=df.columns)
    for i in range(df_tpm.shape[1]):
        df_tpm.iloc[:, i] = RPKM_to_TPM(df.iloc[:, i])

    return df_tpm, cell_lines_information, cell_lines_names, cell_lines_tissues, unique_tissue_names, tissue_occurrences, duplicate_rows_removed, missing_rows_dropped, data_array

# Function to display the first 10 rows of the data
def display_data_table(df):
    st.subheader('CCLE Data Table')
    st.dataframe(df.head(20))  # Show only the first 10 rows

# Function to plot tissue occurrences (Word Cloud)
def plot_tissue_occurrences_wordcloud(tissue_occurrences):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tissue_occurrences.to_dict())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Function to plot tissue occurrences (Sunburst Chart)
def plot_tissue_occurrences_sunburst(tissue_occurrences):
    tissue_occurrences = tissue_occurrences.rename_axis('Tissue').reset_index(name='Count')

        # Calculate the percentage of each tissue
    total_count = tissue_occurrences['Count'].sum()
    tissue_occurrences['Percentage'] = (tissue_occurrences['Count'] / total_count) * 100

    # Plot sunburst chart
    fig = px.sunburst(
        tissue_occurrences,
        path=['Tissue'],
        values='Count',
        title='Tissue Occurrences',
        color='Count',
        color_continuous_scale='viridis',  # Use a single color scheme
        hover_data={'Percentage': ':.2f', 'Count': True}  # Include both count and percentage in hover
    )

    # Update the hover template to show Tissue, Count, and Percentage
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{customdata[1]}<br>Percentage: %{customdata[0]:.2f}%')

    st.plotly_chart(fig)





def plot_gene_and_cell_line_count_horizontal_bar(df):
    gene_count = df.shape[0]
    cell_line_count = df.shape[1]

    fig = go.Figure(data=[
        go.Bar(name='Count', y=['Genes', 'Cell Lines'], x=[gene_count, cell_line_count], orientation='h', marker_color=['darkblue', 'yellow'])
    ])

    fig.update_layout(
        title='Number of Genes and Cell Lines',
        xaxis_title='Count',
        yaxis_title='Category',
        hovermode='y'
    )

    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )

    st.plotly_chart(fig)




# def plot_pca_scatter(df, cell_lines_tissues):
#     # Step 1: Standardize the Data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df.T)  # Transpose because PCA expects samples as rows
#
#     # Step 2: Optimal Number of Components
#     n_components = 0.95
#     reducer1 = PCA(n_components=n_components)
#     r = reducer1.fit_transform(X_scaled)
#
#     # Step 3: Create a More Informative Color Mapping
#     palette = sns.color_palette("husl", len(np.unique(cell_lines_tissues)))
#
#     # Step 4: Adjust Legend Placement and Marker Styles
#     plt.figure(figsize=(12, 8))
#     sns.scatterplot(x=r[:, 0], y=r[:, 1], hue=cell_lines_tissues, palette=palette,
#                     style=cell_lines_tissues, s=180, legend='full')
#     plt.title('PCA: Visualize Cell Lines by Tissue Type')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend(title='Tissue Type', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     st.pyplot(plt)
#
#
#
# def plot_singular_values(df):
#     # Step 1: Standardize the Data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df.T)
#
#     # Step 2: Optimal Number of Components
#     n_components = 0.95
#     reducer1 = PCA(n_components=n_components)
#     reducer1.fit_transform(X_scaled)
#
#     # Plot Singular Values
#     v = reducer1.singular_values_
#     plt.figure(figsize=(20, 5))
#     plt.plot(v, '*-')
#     plt.title('Singular Values')
#     plt.xlabel('Component')
#     plt.ylabel('Singular Value')
#     plt.grid(True)
#     st.pyplot(plt)
#
# def plot_explained_variance_ratio(df):
#     # Step 1: Standardize the Data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df.T)
#
#     # Step 2: Optimal Number of Components
#     n_components = 0.95
#     reducer1 = PCA(n_components=n_components)
#     reducer1.fit_transform(X_scaled)
#
#     # Step 5: Plot Explained Variance Ratio
#     explained_variance_ratio = reducer1.explained_variance_ratio_
#     plt.figure(figsize=(8, 6))
#     plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, color='skyblue')
#     plt.xlabel('Principal Component')
#     plt.ylabel('Explained Variance Ratio')
#     plt.title('Explained Variance Ratio by Principal Component')
#     plt.grid(True)
#     st.pyplot(plt)







def plot_pca_scatter(df, cell_lines_tissues):
    # Step 1: Standardize the Data
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(df.T)  # Transpose because PCA expects samples as rows
    X_scaled = (df.T)
    reducer1 = PCA()
    r = reducer1.fit_transform(X_scaled)

    # Create a More Informative Color Mapping
    palette = sns.color_palette("husl", len(np.unique(cell_lines_tissues)))

    # Plot PCA Scatter
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(x=r[:, 0], y=r[:, 1], hue=cell_lines_tissues, palette=palette,
                              style=cell_lines_tissues, s=180, legend='full')
    plt.title('PCA: Visualize Cell Lines by Tissue Type')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Tissue Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Annotate with key information
    explained_var = np.round(reducer1.explained_variance_ratio_[:2].sum() * 100, 2)
    plt.figtext(0.99, 0.01, f'Variance Explained by PC1 and PC2: {explained_var}%',
                horizontalalignment='right', verticalalignment='bottom', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    st.pyplot(plt)

    # Print key information
    print(f'PCA Scatter Plot:')
    print(f'- Variance Explained by PC1 and PC2: {explained_var}%')

def plot_singular_values(df):
    # Step 1: Standardize the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.T)

    # Step 2: Optimal Number of Components
    n_components = 0.95
    reducer1 = PCA(n_components=n_components)
    reducer1.fit_transform(X_scaled)

    # Plot Singular Values
    singular_values = reducer1.singular_values_
    plt.figure(figsize=(20, 5))
    plt.plot(singular_values, '*-', markersize=10)
    plt.title('Singular Values')
    plt.xlabel('Component')
    plt.ylabel('Singular Value')
    plt.grid(True)

    # Annotate with key information
    max_singular_value = np.max(singular_values)
    plt.figtext(0.99, 0.01, f'Max Singular Value: {max_singular_value:.2f}',
                horizontalalignment='right', verticalalignment='bottom', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    st.pyplot(plt)

    # Print key information
    print(f'Singular Values Plot:')
    print(f'- Max Singular Value: {max_singular_value:.2f}')

def plot_explained_variance_ratio(df):
    # Step 1: Standardize the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.T)

    # Step 2: Optimal Number of Components
    n_components = 0.95
    reducer1 = PCA(n_components=n_components)
    reducer1.fit_transform(X_scaled)

    # Plot Explained Variance Ratio
    explained_variance_ratio = reducer1.explained_variance_ratio_
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, color='skyblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.grid(True)

    # Annotate with key information
    total_variance = np.sum(explained_variance_ratio)
    plt.figtext(0.99, 0.01, f'Total Variance Explained by Top Components: {total_variance:.2f}',
                horizontalalignment='right', verticalalignment='bottom', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    st.pyplot(plt)

    # Print key information
    print(f'Explained Variance Ratio Plot:')
    print(f'- Total Variance Explained by Top Components: {total_variance:.2f}')

def plot_gene_expression(df):
    # Plot mean expression for cells]

    v_cells = df.mean(axis=0)
    x_values = v_cells.index
    y_values = v_cells.values

    # Print details
    max_mean_value = v_cells.max()
    min_mean_value = v_cells.min()
    max_mean_cell = v_cells.idxmax()
    min_mean_cell = v_cells.idxmin()
    median_value = v_cells.median()
    print("Highest mean expression value:", max_mean_value)
    print("Cell line with highest mean expression:", max_mean_cell)
    print("Lowest mean expression value:", min_mean_value)
    print("Cell line with lowest mean expression:", min_mean_cell)
    print("Median mean expression value:", median_value)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Mean Expression', marker=dict(color='skyblue')))
    fig1.update_layout(
        title='Mean Expression for Cells',
        xaxis=dict(
            title='Cells',
            title_standoff=25,  # Adjust the distance of the label from the axis
            showline=True,      # Show x-axis line
            showgrid=True       # Show grid lines
        ),
        yaxis=dict(
            title='Mean Expression',
            title_standoff=25,  # Adjust the distance of the label from the axis
            showline=True,      # Show y-axis line
            showgrid=True       # Show grid lines
        )
    )

    # Plot sorted mean expression for cells
    v_cells_sorted = np.sort(v_cells.values)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.arange(len(v_cells_sorted)), y=v_cells_sorted, mode='lines+markers',
                              name='Sorted Mean Expression', marker=dict(color='lightgreen')))
    fig2.update_layout(title='Sorted Mean Expression for Cells', xaxis_title='Ordered Cells', yaxis_title='Mean Expression')

    # Plot histogram of mean expression for cells
    fig3 = px.histogram(v_cells, nbins=50, title='Histogram of Mean Expression for Cells')

    # Plot mean expression for genes
    v_genes = df.mean(axis=1)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=v_genes.index, y=v_genes.values, mode='lines', name='Mean Expression',
                              marker=dict(color='skyblue')))
    fig4.update_layout(title='Mean Expression for Genes', xaxis_title='Genes', yaxis_title='Mean Expression')

    # Plot sorted mean expression for genes
    v_genes_sorted = np.sort(v_genes.values)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=np.arange(len(v_genes_sorted)), y=v_genes_sorted, mode='lines+markers',
                              name='Sorted Mean Expression', marker=dict(color='lightgreen')))
    fig5.update_layout(title='Sorted Mean Expression for Genes', xaxis_title='Ordered Genes', yaxis_title='Mean Expression')

    # Plot histogram of mean expression for genes
    fig6 = px.histogram(v_genes, nbins=50, title='Histogram of Mean Expression for Genes')

    # Top expressed genes
    top_genes = v_genes.sort_values(ascending=False).head(40)

    # Plot top expressed genes
    fig7 = go.Figure([go.Bar(x=top_genes.index, y=top_genes.values, marker_color='skyblue')])
    fig7.update_layout(title='Top Expressed Genes', xaxis_title='Genes', yaxis_title='Mean Expression')
    # Display all plots within the same app
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)
    st.plotly_chart(fig5)
    st.plotly_chart(fig6)
    st.plotly_chart(fig7)



def plot_correlation_matrix(df, title):
    cm = df.corr()

    list_names = list(df.columns)
    a, b = np.where(cm > 0.975)
    temp_list = []
    st_data = pd.DataFrame()
    for i in range(len(a)):
        if a[i] == b[i]: continue
        tmp_hash1 = str(a[i]) + '_' + str(b[i])
        tmp_hash2 = str(b[i]) + '_' + str(a[i])
        if (tmp_hash1 in temp_list) or (tmp_hash2 in temp_list): continue
        temp_list.append(tmp_hash1)
        st_data.loc[i, 'Name1'] = list_names[a[i]]
        st_data.loc[i, 'Name2'] = list_names[b[i]]
        st_data.loc[i, 'Correlation'] = cm.iloc[a[i], b[i]]
        st_data.loc[i, 'Abs Correlation'] = np.abs(cm.iloc[a[i], b[i]])

    st_data = st_data.sort_values('Abs Correlation', ascending=False)

    st.subheader(f'Top Correlated Pairs for {title}')
    st.dataframe(st_data)

    #Plot heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title(f'{title} Correlation Matrix Heatmap')
    st.pyplot(plt)





def display_ccle_data_analysis():
    st.title('CCLE Gene Expression Data')

    # Read and preprocess the data
    df, cell_lines_information, cell_lines_names, cell_lines_tissues, unique_tissue_names, tissue_occurrences, duplicate_rows_removed, missing_rows_dropped, data_array = preprocess_data("./DataIn/CCLE/CCLE_RPKM.gct")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Rows with missing values dropped: {missing_rows_dropped}")
    print(f"Formatted data as NumPy array: {data_array.shape}")
    print("First few rows of TPM data:")
    print(df.head())
    # Display the first 10 rows of the data by default
    display_data_table(df)

    # Sidebar radio button for view selection
    st.sidebar.subheader('Select Graph Type')
    view_option = st.sidebar.radio('Choose view', [

        'Data Information',
        'PCA Analysis',
        'Gene Expression',
        'Correlation Analysis'
    ])


    if view_option == 'Data Information':
            plot_gene_and_cell_line_count_horizontal_bar(df)
            plot_tissue_occurrences_sunburst(tissue_occurrences)
            plot_tissue_occurrences_wordcloud(tissue_occurrences)
    elif view_option=='PCA Analysis': # Call the function to visualize PCA
        plot_pca_scatter(df, cell_lines_tissues)
        plot_explained_variance_ratio(df)
        plot_singular_values(df)
    elif view_option=='Gene Expression':
        plot_gene_expression(df)
    elif view_option == 'Correlation Analysis':
        plot_correlation_matrix(df.T, "Cell Lines")
        plot_correlation_matrix(df, "Genes")

