import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to read CTRPv2 data
def read_ctrp_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    required_columns = ['cpd_name', 'CCL_Name', 'area_under_curve', 'Avg_AUC']
    data = data[required_columns]
    data.columns = ['Drug', 'Cell_Line', 'AUC', 'Avg_AUC']
    return data

# Function to plot AUC distribution
def plot_auc_distribution(ctrp_data):
    fig_auc = px.histogram(ctrp_data, x='AUC', nbins=50, title='Distribution of Area Under Curve (AUC)', color_discrete_sequence=['skyblue'])
    fig_auc.update_layout(
        title='Distribution of Area Under Curve (AUC)',
        xaxis_title='Area Under Curve (AUC)',
        yaxis_title='Frequency',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15)),
        yaxis=dict(title_font=dict(size=15))
    )
    max_auc = ctrp_data['AUC'].max()
    min_auc = ctrp_data['AUC'].min()

    fig_auc.add_annotation(
        x=max_auc, y=0,
        text=f'Highest AUC: {max_auc:.2f}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-50,
        bgcolor='yellow',
        bordercolor='black',
        borderwidth=2
    )

    fig_auc.add_annotation(
        x=min_auc, y=0,
        text=f'Lowest AUC: {min_auc:.2f}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-50,
        bgcolor='yellow',
        bordercolor='black',
        borderwidth=2
    )
    fig_auc.show()

# Function to plot Average AUC distribution
def plot_avg_auc_distribution(ctrp_data):
    fig_avg_auc = px.histogram(ctrp_data, x='Avg_AUC', nbins=50, title='Distribution of Average AUC', color_discrete_sequence=['green'])
    fig_avg_auc.update_layout(
        title='Distribution of Average AUC',
        xaxis_title='Average AUC',
        yaxis_title='Frequency',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15)),
        yaxis=dict(title_font=dict(size=15))
    )
    max_avg_auc = ctrp_data['Avg_AUC'].max()
    min_avg_auc = ctrp_data['Avg_AUC'].min()

    fig_avg_auc.add_annotation(
        x=max_avg_auc, y=0,
        text=f'Highest Avg_AUC: {max_avg_auc:.2f}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-50,
        bgcolor='yellow',
        bordercolor='black',
        borderwidth=2
    )

    fig_avg_auc.add_annotation(
        x=min_avg_auc, y=0,
        text=f'Lowest Avg_AUC: {min_avg_auc:.2f}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-50,
        bgcolor='yellow',
        bordercolor='black',
        borderwidth=2
    )
    fig_avg_auc.show()

# Function to perform linear regression analysis
def perform_regression_analysis(ctrp_data):
    X = ctrp_data[['Avg_AUC']]
    y = ctrp_data['AUC']

    model = LinearRegression()
    model.fit(X, y)

    # Predict AUC values using Avg_AUC
    ctrp_data['Predicted_AUC'] = model.predict(X)

    # Plot regression line with hover data
    fig_reg = px.scatter(
        ctrp_data,
        x='Avg_AUC',
        y='AUC',
        title='Regression Analysis: AUC vs. Avg_AUC',
        trendline='ols',
        hover_data=['Avg_AUC', 'AUC', 'Predicted_AUC']
    )
    fig_reg.update_layout(
        title='Regression Analysis: AUC vs. Avg_AUC',
        xaxis_title='Average AUC',
        yaxis_title='AUC',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15)),
        yaxis=dict(title_font=dict(size=15))
    )
    fig_reg.show()

# Function to perform cluster analysis
def perform_cluster_analysis(ctrp_data, num_clusters):
    # Select the features for clustering
    X = ctrp_data[['AUC', 'Avg_AUC']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    ctrp_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Add cluster labels to the dataframe for hover information
    ctrp_data['Cluster'] = ctrp_data['Cluster'].astype(str)

    # Plot clusters
    fig_cluster = px.scatter(
        ctrp_data,
        x='AUC',
        y='Avg_AUC',
        color='Cluster',
        hover_data={
            'Drug': True,        # Display drug names
            'Cell_Line': True,   # Display cell line names
            'AUC': ':.2f',       # Format AUC to two decimal places
            'Avg_AUC': ':.2f',   # Format Avg_AUC to two decimal places
            'Cluster': True      # Display cluster labels
        },
        title=f'Cluster Analysis: K-means Clustering (K={num_clusters})'
    )
    fig_cluster.update_layout(
        title=f'Cluster Analysis: K-means Clustering (K={num_clusters})',
        xaxis_title='AUC',
        yaxis_title='Avg_AUC',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15)),
        yaxis=dict(title_font=dict(size=15))
    )
    fig_cluster.show()

    # Save data for Cluster 1 to CSV
    cluster_1_data = ctrp_data[ctrp_data['Cluster'] == '1']  # Assuming Cluster 1 is labeled as '0'
    cluster_1_data.to_csv(f'cluster_1_data.csv', index=False)

    return fig_cluster


# Function to create heatmap analysis
def create_heatmap_analysis(ctrp_data, metric):
    try:
        # Get the pivot table
        z_data = get_heatmap_data(ctrp_data, metric)

        # Create heatmap using Plotly
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_data.values,
            x=z_data.columns,
            y=z_data.index,
            colorscale='Viridis',
            hovertemplate='Cell Line: %{y}<br>Drug: %{x}<br>' + metric + ': %{z}<br>',
            showscale=True  # Ensure the color scale is visible
        ))

        # Update heatmap layout
        fig_heatmap.update_layout(
            title=f'Heatmap Analysis: {metric}',
            xaxis_title='Cell Line',
            yaxis_title='Drug',
            title_font=dict(size=20, family='Arial', color='black'),
            xaxis=dict(title_font=dict(size=15)),
            yaxis=dict(title_font=dict(size=15))
        )

        # Display heatmap
        fig_heatmap.show()

    except ValueError as e:
        print(f"Error creating heatmap: {e}")

# Function to get heatmap data
def get_heatmap_data(ctrp_data, metric):
    if metric == 'AUC':
        z_data = ctrp_data.pivot_table(index='Drug', columns='Cell_Line', values='AUC', aggfunc='mean')
    elif metric == 'Avg_AUC':
        z_data = ctrp_data.pivot_table(index='Drug', columns='Cell_Line', values='Avg_AUC', aggfunc='mean')
    else:
        raise ValueError("Invalid metric. Supported metrics are 'AUC' and 'Avg_AUC'.")

    return z_data

# Function to perform dimensionality reduction using PCA
def perform_dimensionality_reduction(ctrp_data):
    X = ctrp_data[['AUC', 'Avg_AUC']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Create dataframe of principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Drug'] = ctrp_data['Drug']
    pca_df['Cell_Line'] = ctrp_data['Cell_Line']
    pca_df['Cluster'] = ctrp_data['Cluster']

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_

    # Plot PCA results
    fig_pca = px.scatter(
        pca_df, x='PC1', y='PC2', color='Cluster',
        hover_data={'Drug': True, 'Cell_Line': True},
        title='Dimensionality Reduction: PCA'
    )
    fig_pca.update_layout(
        title=f'Dimensionality Reduction: PCA (Explained Variance: PC1 = {explained_variance[0]*100:.2f}%, PC2 = {explained_variance[1]*100:.2f}%)',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15)),
        yaxis=dict(title_font=dict(size=15))
    )

    # Add annotations for cluster centroids
    cluster_centroids = pca_df.groupby('Cluster').mean().reset_index()
    for index, row in cluster_centroids.iterrows():
        fig_pca.add_annotation(
            x=row['PC1'], y=row['PC2'],
            text=f'Cluster {row["Cluster"]}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            bgcolor='yellow',
            bordercolor='black',
            borderwidth=2
        )

    fig_pca.show()


if __name__ == "__main__":
    # Define file path for CTRPv2 data
    file_path = './DataIn/CTRPv2/CTRPv2_AUC_clean.txt'

    # Read the data
    ctrp_data = read_ctrp_data(file_path)

    # Plot distributions
    plot_auc_distribution(ctrp_data)
    plot_avg_auc_distribution(ctrp_data)

    # Perform regression analysis
    perform_regression_analysis(ctrp_data)

    # Perform cluster analysis
    num_clusters = 5
    perform_cluster_analysis(ctrp_data, num_clusters)

    # Create heatmap analysis
    create_heatmap_analysis(ctrp_data, 'AUC')
    create_heatmap_analysis(ctrp_data, 'Avg_AUC')

    # Perform dimensionality reduction
    perform_dimensionality_reduction(ctrp_data)
