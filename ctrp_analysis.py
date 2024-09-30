import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
# Function to calculate number of unique cell lines per drug
def calculate_avg_auc_threshold_counts(ctrp_data, threshold):
    # Filter drugs based on the average AUC threshold
    filtered_data = ctrp_data[(ctrp_data['Avg_AUC'] >= threshold[0]) & (ctrp_data['Avg_AUC'] <= threshold[1])]

    # Count drugs within the threshold and get drug names
    drug_counts = filtered_data.groupby('Drug')['Avg_AUC'].count().reset_index()
    drug_counts.columns = ['Drug', 'Count']

    return drug_counts

# Function to plot the number of drugs within each threshold range
def plot_avg_auc_threshold_counts(ctrp_data, threshold):
    # Calculate drug counts within the threshold
    drug_counts = calculate_avg_auc_threshold_counts(ctrp_data, threshold)

    # Plot the bar chart
    fig = px.bar(drug_counts, x='Drug', y='Count',
                 title=f'Drugs with Average AUC between {threshold[0]} and {threshold[1]}',
                 labels={'Count': 'Number of Drugs', 'Drug': 'Drug Name'},
                 color='Drug', color_continuous_scale='inferno')

    fig.update_layout(
        xaxis_title='Drug',
        yaxis_title='Number of Drugs',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15), tickangle=45),
        yaxis=dict(title_font=dict(size=15)),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Assuming ctrp_data is a DataFrame loaded with the necessary data

def calculate_drug_cell_line_count(ctrp_data):
    drug_cell_line_count = ctrp_data.groupby('Drug')['Cell_Line'].nunique().reset_index()
    drug_cell_line_count.columns = ['Drug', 'Unique_Cell_Lines']
    return drug_cell_line_count

# Function to plot number of unique cell lines per drug
def plot_drug_cell_line_count(drug_cell_line_count):
    # Sort drug_cell_line_count to find smallest and largest values
    sorted_counts = drug_cell_line_count.sort_values(by='Unique_Cell_Lines')
    print("Number of unique cell lines per drug:")
    drugs_of_interest = [
        "AT7867", "AZD6482", "AZD7762", "AZD8055", "BMS-345541", "BMS-536924", "BMS-754807",
        "CHIR-99021", "GSK1059615", "IC-87114", "KU-55933", "MK-2206", "OSI-027", "OSI-930",
        "PHA-793887", "PI-103", "PIK-93", "PLX-4720", "SN-38", "SNX-2112", "SU11274", "TPCA-1",
        "UNC0638", "ZSTK474"
    ]

    filtered_data = drug_cell_line_count[drug_cell_line_count['Drug'].isin(drugs_of_interest)]

    # Print the filtered cell lines and their counts
    for index, row in filtered_data.iterrows():
        print(f"Drug: {row['Drug']}, Unique Cell Lines: {row['Unique_Cell_Lines']}")


    # Plot the bar chart
    fig = px.bar(drug_cell_line_count, x='Drug', y='Unique_Cell_Lines',
                 title='Number of Unique Cell Lines per Drug',
                 labels={'Unique_Cell_Lines': 'Number of Unique Cell Lines', 'Drug': 'Drug Name'},
                 color='Drug')

    fig.update_layout(
        xaxis_title='Drug',
        yaxis_title='Number of Unique Cell Lines',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15)),
        yaxis=dict(title_font=dict(size=15))
    )

    # Add annotations for smallest and largest values
    smallest = sorted_counts.iloc[0]
    largest = sorted_counts.iloc[-1]

    fig.add_annotation(x=smallest['Drug'], y=smallest['Unique_Cell_Lines'],
                       text=f'Smallest: {smallest["Unique_Cell_Lines"]}', showarrow=True,
                       arrowhead=1, ax=0, ay=-30, font=dict(color='red'))

    fig.add_annotation(x=largest['Drug'], y=largest['Unique_Cell_Lines'],
                       text=f'Largest: {largest["Unique_Cell_Lines"]}', showarrow=True,
                       arrowhead=1, ax=0, ay=-30, font=dict(color='blue'))

    st.plotly_chart(fig, use_container_width=True)


# Example usage:
# Load your CTRPv2 data into a DataFrame
# ctrp_data = pd.read_csv('path_to_your_file.csv')

# Calculate the drug cell line count
# drug_cell_line_count = calculate_drug_cell_line_count(ctrp_data)

# Plot the drug cell line count
# plot_drug_cell_line_count(drug_cell_line_count)


# Function to read CTRPv2 data
def read_ctrp_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    required_columns = ['cpd_name', 'CCL_Name', 'area_under_curve', 'Avg_AUC']
    data = data[required_columns]
    data.columns = ['Drug', 'Cell_Line', 'AUC', 'Avg_AUC']
    # data.to_csv('./DataIn/CTRPv2/CTRPv2_AUC_filtered.csv', index=False)
    # print("file saved")
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
    st.plotly_chart(fig_auc, use_container_width=True)

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
    st.plotly_chart(fig_reg, use_container_width=True)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# def perform_cluster_analysis(ctrp_data, num_clusters):
#     # Select the features for clustering
#     X = ctrp_data[['AUC', 'Avg_AUC']]
#
#     # Standardize the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#     ctrp_data['Cluster'] = kmeans.fit_predict(X_scaled)
#
#     # Plot clusters
#     fig_cluster = px.scatter(
#         ctrp_data,
#         x='AUC',
#         y='Avg_AUC',
#         color='Cluster',
#         title=f'Cluster Analysis: K-means Clustering (K={num_clusters})'
#     )
#     fig_cluster.update_layout(
#         title=f'Cluster Analysis: K-means Clustering (K={num_clusters})',
#         xaxis_title='AUC',
#         yaxis_title='Avg_AUC',
#         title_font=dict(size=20, family='Arial', color='black'),
#         xaxis=dict(title_font=dict(size=15)),
#         yaxis=dict(title_font=dict(size=15))
#     )
#     st.plotly_chart(fig_cluster, use_container_width=True)
# def perform_cluster_analysis(ctrp_data, num_clusters):
#     # Select the features for clustering
#     X = ctrp_data[['AUC', 'Avg_AUC']]
#
#     # Standardize the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#     ctrp_data['Cluster'] = kmeans.fit_predict(X_scaled)
#
#     # Add cluster labels to the dataframe for hover information
#     ctrp_data['Cluster'] = ctrp_data['Cluster'].astype(str)
#
#     # Compute silhouette score
#     silhouette_avg = silhouette_score(X_scaled, ctrp_data['Cluster'])
#     st.write(f"Silhouette Score: {silhouette_avg:.4f}")
#
#     # Define hover data
#     hover_data = {
#         'Drug': True,        # Display drug names
#         'Cell_Line': True,   # Display cell line names
#         'AUC': ':.2f',       # Format AUC to two decimal places
#         'Avg_AUC': ':.2f',   # Format Avg_AUC to two decimal places
#         'Cluster': True      # Display cluster labels
#     }
#
#     # Plot clusters
#     fig_cluster = px.scatter(
#         ctrp_data,
#         x='AUC',
#         y='Avg_AUC',
#         color='Cluster',
#         hover_data=hover_data,
#         title=f'Cluster Analysis: K-means Clustering (K={num_clusters})'
#     )
#     fig_cluster.update_layout(
#         title=f'Cluster Analysis: K-means Clustering (K={num_clusters})',
#         xaxis_title='AUC',
#         yaxis_title='Avg_AUC',
#         title_font=dict(size=20, family='Arial', color='black'),
#         xaxis=dict(title_font=dict(size=15)),
#         yaxis=dict(title_font=dict(size=15))
#     )
#
#     # Add interactive features
#     fig_cluster.update_traces(marker=dict(size=8, opacity=0.8))
#
#     # Display the plot using Streamlit
#     st.plotly_chart(fig_cluster, use_container_width=True)
#
#     # Optionally, display additional cluster information such as centroids and cluster sizes
#     st.subheader('Cluster Information')
#
#     # Compute and display cluster centroids
#     centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['AUC', 'Avg_AUC'])
#     centroids['Cluster'] = centroids.index.astype(str)
#     st.write('Cluster Centroids:')
#     st.write(centroids)
#
#     # Display cluster sizes
#     cluster_sizes = ctrp_data['Cluster'].value_counts().sort_index()
#     st.write('Cluster Sizes:')
#     st.write(cluster_sizes)

# Function to create heatmap analysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assume ctrp_data is your DataFrame containing the CTRP data

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

    # Define hover data
    hover_data = {
        'Drug': True,        # Display drug names
        'Cell_Line': True,   # Display cell line names
        'AUC': ':.2f',       # Format AUC to two decimal places
        'Avg_AUC': ':.2f',   # Format Avg_AUC to two decimal places
        'Cluster': True      # Display cluster labels
    }

    # Plot clusters
    fig_cluster = px.scatter(
        ctrp_data,
        x='AUC',
        y='Avg_AUC',
        color='Cluster',
        hover_data=hover_data,
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
    st.plotly_chart(fig_cluster, use_container_width=True)




# Function to generate cluster descriptions
def get_heatmap_data(ctrp_data, metric):
    if metric == 'AUC':
        z_data = ctrp_data.pivot_table(index='Drug', columns='Cell_Line', values='AUC', aggfunc='mean')
    elif metric == 'Avg_AUC':
        z_data = ctrp_data.pivot_table(index='Drug', columns='Cell_Line', values='Avg_AUC', aggfunc='mean')
    else:
        raise ValueError("Invalid metric. Supported metrics are 'AUC' and 'Avg_AUC'.")

    return z_data


# Function to display the heatmap and the underlying data
# Function to create heatmap analysis
def create_heatmap_analysis(ctrp_data, metric):
    try:
        # Get the pivot table
        z_data = get_heatmap_data(ctrp_data, metric)

        # Display pivot table (optional)
        st.dataframe(z_data)

        # Create heatmap using Plotly
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_data.values,
            x=z_data.columns,
            y=z_data.index,
            colorscale='Viridis',
            hovertemplate='Cell Line: %{y}<br>Drug: %{x}<br>AUC: %{z}<br>',
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

        # Display heatmap in Streamlit
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.subheader('Key Findings:')

        # # Identify drugs programmatically
        # high_efficacy_drugs, resistant_drugs = identify_drugs(ctrp_data)
        #
        # # Display findings
        # st.markdown('**High Efficacy Drugs:**')
        # st.write('\n'.join(high_efficacy_drugs))
        #
        # st.markdown('**Resistant Drugs:**')
        # st.write('\n'.join(resistant_drugs))

    except ValueError as e:
        st.error(f"Error creating heatmap: {e}")


# Function to perform dimensionality reduction using PCA
# def perform_dimensionality_reduction(ctrp_data):
#     X = ctrp_data[['AUC', 'Avg_AUC']]
#
#     # Standardize the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Perform PCA
#     pca = PCA(n_components=2)
#     principal_components = pca.fit_transform(X_scaled)
#
#     # Create dataframe of principal components
#     pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
#
#     # Plot PCA results
#     fig_pca = px.scatter(pca_df, x='PC1', y='PC2', title='Dimensionality Reduction: PCA')
#     fig_pca.update_layout(
#         title='Dimensionality Reduction: PCA',
#         xaxis_title='Principal Component 1',
#         yaxis_title='Principal Component 2',
#         title_font=dict(size=20, family='Arial', color='black'),
#         xaxis=dict(title_font=dict(size=15)),
#         yaxis=dict(title_font=dict(size=15))
#     )
#     st.plotly_chart(fig_pca, use_container_width=True)


########################################
def perform_dimensionality_reduction(ctrp_data):
    # Extract AUC and Avg_AUC columns
    X = ctrp_data[['AUC', 'Avg_AUC']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Create dataframe of principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Explained variance ratios
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()

    # Print key PCA information
    st.write(f"Variance explained by PC1: {explained_variance[0] * 100:.2f}%")
    st.write(f"Variance explained by PC2: {explained_variance[1] * 100:.2f}%")
    st.write(f"Total variance explained by first two components: {total_variance * 100:.2f}%")

    # Apply K-means clustering on the PCA results
    kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust number of clusters as needed
    pca_df['Cluster'] = kmeans.fit_predict(principal_components)

    # Merge PCA results with original data to get detailed information
    pca_df['Drug'] = ctrp_data['Drug']  # Adjust column names as necessary
    pca_df['Cell_Line'] = ctrp_data['Cell_Line']  # Adjust column names as necessary

    # Print cluster centers
    st.write("Cluster Centers in the PCA space:")
    st.write(pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2'], index=[f'Cluster {i}' for i in range(5)]))

    # Add Cluster Characteristics
    for i in range(5):
        st.write(f"\nCluster {i} Characteristics:")
        st.write(pca_df[pca_df['Cluster'] == i].describe())

        # Display specific drugs and cell lines in each cluster
        st.write(f"\nDrugs and Cell Lines in Cluster {i}:")
        st.write(pca_df[pca_df['Cluster'] == i][['Drug', 'Cell_Line']])

    # Plot PCA results with clusters
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                         title='Dimensionality Reduction: PCA with K-means Clustering',
                         color_continuous_scale=px.colors.qualitative.Set1)

    fig_pca.update_layout(
        title='Dimensionality Reduction: PCA with K-means Clustering',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15)),
        yaxis=dict(title_font=dict(size=15))
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    # Print PCA Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=['AUC', 'Avg_AUC'])
    st.write("PCA Loadings:")
    st.write(loadings)

    # Additional Statistics (optional)
    st.write("Silhouette Score for Clustering:")
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(principal_components, pca_df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    # Interpretations
    st.write("Interpretation of PCA Loadings:")
    st.write("PC1 represents a combination of AUC and Avg_AUC with equal contribution. This principal component explains the majority of the variance, indicating that AUC and Avg_AUC are strongly correlated.")
    st.write("PC2 captures the difference between AUC and Avg_AUC. Its minimal contribution suggests that variations between these metrics are relatively insignificant.")



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
    st.plotly_chart(fig_avg_auc, use_container_width=True)


# Main function to display CTRP data analysis with interactive options
def display_ctrp_data_analysis():
    #ctrp_data = read_ctrp_data("DataIn/CTRPv2/CTRPv2_AUC_clean.txt")
 

    file_path = "./DataIn/CTRPv2/CTRPv2_AUC_clean.txt"

    if os.path.exists(file_path):
        st.write("present")
        try:
            ctrp_data = pd.read_csv(file_path, sep='\t')  # Try reading directly with pandas
            st.write(ctrp_data.head())
        except Exception as e:
            st.error(f"Error while reading file: {e}")
    else:
        st.error(f"Error: File '{file_path}' not found.")

    st.title("CTRP Drug Response Data Analysis")
    st.dataframe(ctrp_data, height=400, width=1500)

    st.sidebar.header("Select Analysis to Perform")
    selected_analysis = st.sidebar.radio(
        "Select Analysis",
        [
            "AUC Distribution",
            "Average AUC Distribution",
            "Drug-Cell Line Interaction",
            "Drug Counts by Average AUC Threshold",
            "Regression Analysis",
            "Cluster Analysis",
            "Heatmap Analysis",
            "Dimensionality Reduction"

        ]
    )

    if selected_analysis == "AUC Distribution":
        plot_auc_distribution(ctrp_data)

    elif selected_analysis == "Average AUC Distribution":
        plot_avg_auc_distribution(ctrp_data)
    elif selected_analysis == "Drug Counts by Average AUC Threshold":
            st.sidebar.header('Set Average AUC Threshold Range')
            # min_threshold = st.sidebar.slider('Minimum Threshold', min_value=0.0, max_value=28.0, value=5.0, step=0.5)
            max_threshold = st.sidebar.slider('Maximum Threshold', min_value=0.07, max_value=29.95, value=5.0, step=0.5)
            threshold = (0.07, max_threshold)
            # Display filtered drug counts and plot
            st.header(f'Drugs with Average AUC between {threshold[0]} and {threshold[1]}')
            plot_avg_auc_threshold_counts(ctrp_data, threshold)


    elif selected_analysis == "Regression Analysis":
        perform_regression_analysis(ctrp_data)

    elif selected_analysis == "Cluster Analysis":
        st.sidebar.subheader("Cluster Analysis Parameters")
        num_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=10, value=3, step=1)
        if st.sidebar.button("Perform Clustering"):
            perform_cluster_analysis(ctrp_data, num_clusters)

    elif selected_analysis == "Heatmap Analysis":
        st.sidebar.subheader("Heatmap Analysis Options")
        metric = st.sidebar.selectbox("Select Metric", ['AUC', 'Avg_AUC'])
        if st.sidebar.button("Generate Heatmap"):
            create_heatmap_analysis(ctrp_data, metric)

    elif selected_analysis == "Dimensionality Reduction":
        perform_dimensionality_reduction(ctrp_data)
    elif selected_analysis == "Drug-Cell Line Interaction":
        drug_cell_line_count = calculate_drug_cell_line_count(ctrp_data)
        plot_drug_cell_line_count(drug_cell_line_count)


