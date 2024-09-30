from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
#from transformers import pipeline
import re
import numpy as np

from ccle_analysis import display_ccle_data_analysis
from ctrp_analysis import display_ctrp_data_analysis
from make_Prediction import new_data

# Function to load model results data
def load_model_results():
    return pd.read_csv("./Output/ALL_DRUGS_ALL_MODELS.csv")

# Function to load actual, predicted, and difference data
def load_actual_predicted_data():
    df = pd.read_csv("./Output/ALL_DRUGS_ALL_MODELS_ACTUAL_PREDICTED_Processed.csv")
    df['Difference'] = abs(df['Difference'])
    return df

# Function to display model results
def display_model_results(model_results_df):
    st.write("### Drug Model Results")
    display_df = model_results_df[['Unnamed: 0', 'RMSE', 'R2', 'Top_20_genes_pc1']]
    display_df.columns = ['Drug', 'RMSE', 'R2', 'Top 20 Genes']
    display_df.index = display_df.index + 1  # Start index from 1

    # Apply custom CSS for styling
    st.markdown("""
        <style>
        .dataframe {background-color: #f9f9f9; color: #333; border: 1px solid #ddd; border-radius: 4px; padding: 8px;}
        .dataframe th {background-color: #f1f1f1; font-weight: bold; text-align: center;}
        .dataframe td {text-align: center; padding: 8px;}
        .dataframe tr:nth-child(even) {background-color: #f9f9f9;}
        .dataframe tr:hover {background-color: #f1f1f1;}
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(display_df, height=400, width=800)

# Function to filter data based on RMSE selection
def filter_data_by_rmse(model_results_df, rmse_filter):
    if rmse_filter == "Less than 1":
        filtered_df = model_results_df[model_results_df['RMSE'] < 1]
    elif rmse_filter == "Less than 1.5":
        filtered_df = model_results_df[model_results_df['RMSE'] < 1.5]
    elif rmse_filter == "Less than 2":
        filtered_df = model_results_df[model_results_df['RMSE'] < 2]
    elif rmse_filter == "Between 0 to 1":
        filtered_df = model_results_df[(model_results_df['RMSE'] >= 0) & (model_results_df['RMSE'] < 1)]
    elif rmse_filter == "Between 1 to 1.5":
        filtered_df = model_results_df[(model_results_df['RMSE'] >= 1) & (model_results_df['RMSE'] < 1.5)]
    elif rmse_filter == "Between 1.5 to 2":
        filtered_df = model_results_df[(model_results_df['RMSE'] >= 1.5) & (model_results_df['RMSE'] < 2)]
    elif rmse_filter == "Greater than equal to 2":
        filtered_df = model_results_df[model_results_df['RMSE'] >= 2]
    else:
        filtered_df = model_results_df
    return filtered_df

# Function to display actual, predicted, and difference data for a selected drug
def display_actual_predicted_data(actual_predicted_df, model_results_df, selected_drug):
    st.write(f"### Data for {selected_drug}")

    filtered_df = actual_predicted_df[actual_predicted_df['Drug'] == selected_drug].reset_index(drop=True)
    filtered_df.index = filtered_df.index + 1  # Start index from 1

    rmse_value = model_results_df[model_results_df['Unnamed: 0'] == selected_drug]['RMSE'].values[0]
    st.write(f"**RMSE for {selected_drug}:** {rmse_value:.4f}")

    # Apply custom CSS for styling
    st.markdown("""
        <style>
        .dataframe {background-color: #f9f9f9; color: #333; border: 1px solid #ddd; border-radius: 4px; padding: 8px;}
        .dataframe th {background-color: #f1f1f1; font-weight: bold; text-align: center;}
        .dataframe td {text-align: center; padding: 8px;}
        .dataframe tr:nth-child(even) {background-color: #f9f9f9;}
        .dataframe tr:hover {background-color: #f1f1f1;}
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(filtered_df, height=400, width=800)
    return filtered_df

# Function to create scatter plot for RMSE vs R2
def create_rmse_r2_scatter_plot(df):
    fig = go.Figure(go.Scatter(
        x=df['RMSE'], y=df['R2'], mode='markers',
        marker=dict(size=10, color=df['R2'], colorscale='Viridis', line=dict(width=1, color='Black')),
        text=df['Unnamed: 0'],
        hovertemplate="<b>Drug</b>: %{text}<br><b>RMSE</b>: %{x}<br><b>R2</b>: %{y}<br>",
    ))
    fig.update_layout(
        title="RMSE vs R2 for All Drugs",
        xaxis_title="RMSE",
        yaxis_title="R2",
        showlegend=False,
        font=dict(family="Arial", size=12, color="black"),
    )
    st.plotly_chart(fig)

# Function to create bar plot for RMSE range distribution
def create_rmse_range_bar_plot(df):
    fig = px.histogram(df, x='RMSE', title='Distribution of RMSE Values', nbins=20, color_discrete_sequence=['#1f77b4'])
    fig.update_layout(bargap=0.1, xaxis_title='RMSE', yaxis_title='Number of Drugs', template='plotly_white')
    st.plotly_chart(fig)

# Function to check if a gene is mitochondrial
def is_mitochondrial(gene_name):
    return gene_name.startswith('MT-')

def create_genes_frequency_bar_plot(df, remove_mitochondrial):
    # Extract genes list from the dataframe
    genes_list = df['Top_20_genes_pc1'].apply(eval).tolist()
    all_genes = [gene for sublist in genes_list for gene in sublist]

    # Filter out mitochondrial genes if the checkbox is checked
    if remove_mitochondrial:
        all_genes = [gene for gene in all_genes if not is_mitochondrial(gene)]

    # Count the frequency of each gene
    gene_counts = Counter(all_genes)

    # Create a dataframe from the gene counts
    gene_df = pd.DataFrame(gene_counts.items(), columns=['Gene', 'Frequency'])

    # Create a bar plot using Plotly
    fig = px.bar(gene_df, x='Gene', y='Frequency', title='Frequency of Genes Across All Drugs', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Set3)

    # Add lines to bars
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)

    fig.update_layout(
        xaxis_title='Gene',
        yaxis_title='Frequency',
        showlegend=False
    )

    st.plotly_chart(fig)

# Function to create bar plot for frequency of models used
def create_model_frequency_bar_plot(df):
    # Count the frequency of each model
    model_counts = df['Model'].value_counts()

    # Create a dataframe from the model counts
    model_df = pd.DataFrame(model_counts).reset_index()
    model_df.columns = ['Model', 'Frequency']

    # Create a bar plot using Plotly
    fig = px.bar(model_df, x='Model', y='Frequency', title='Frequency of Models Used', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Set2)

    # Add lines to bars
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)

    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Frequency',
        showlegend=False
    )

    st.plotly_chart(fig)

# Function to create scatter plot for actual vs predicted values
def create_scatter_plot(filtered_df, selected_drug):
    fig = go.Figure(go.Scatter(
        x=filtered_df['Actual'], y=filtered_df['Predicted'], mode='markers',
        marker=dict(size=8, color=filtered_df['Predicted'], colorscale='Viridis', line=dict(width=1, color='Black')),
        text=filtered_df.index,
        hovertemplate="<b>Index</b>: %{text}<br><b>Actual</b>: %{x}<br><b>Predicted</b>: %{y}</b>: %{marker.color}<br>",
    ))
    fig.add_trace(go.Scatter(x=[filtered_df['Actual'].min(), filtered_df['Actual'].max()],
                             y=[filtered_df['Actual'].min(), filtered_df['Actual'].max()],
                             mode='lines', line=dict(color="Red", width=2, dash="dash")))
    fig.update_layout(title=f"Actual vs Predicted for Drug {selected_drug}", xaxis_title="Actual", yaxis_title="Predicted", showlegend=False)
    st.plotly_chart(fig)

# Function to create residuals histogram
def create_residuals_histogram(filtered_df, selected_drug):
    residuals = abs(filtered_df['Actual'] - filtered_df['Predicted'])

    fig = go.Figure(go.Histogram(x=residuals, nbinsx=20, marker=dict(color='teal'), opacity=0.7))
    fig.update_layout(title=f"Histogram of Residuals for Drug {selected_drug}", xaxis_title="Residuals", yaxis_title="Frequency", showlegend=False)
    st.plotly_chart(fig)

# Function to create bar plot for highest and lowest values
def create_highest_lowest_bar_plot(filtered_df, selected_drug):
    categories = ['Highest Actual', 'Lowest Actual', 'Highest Predicted', 'Lowest Predicted']
    values = [filtered_df['Actual'].max(), filtered_df['Actual'].min(), filtered_df['Predicted'].max(), filtered_df['Predicted'].min()]

    fig = go.Figure(go.Bar(x=categories, y=values, marker_color=['blue', 'blue', 'orange', 'orange']))
    fig.update_layout(title=f"Bar Plot: Highest and Lowest Values for Drug {selected_drug}", xaxis_title="Category", yaxis_title="Value", showlegend=False)
    st.plotly_chart(fig)

# Function to calculate difference ranges and return percentages
def calculate_difference_ranges(differences):
    ranges = {
        'Less than equal to 1': sum(1 for diff in differences if 0 <= diff <= 1),
        'Less than equal to 2': sum(1 for diff in differences if 1 < diff <= 2),
        'Greater than 2': sum(1 for diff in differences if diff > 2)
    }
    total = len(differences)
    percentages = {k: (v / total) * 100 for k, v in ranges.items()}
    return percentages

# Function to create bar plot for difference ranges
def create_difference_ranges_bar_plot(percentages):
    categories, percents = zip(*percentages.items())

    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize as needed
    bars = ax.bar(categories, percents, color=['#1f77b4', '#ff7f0e', '#d62728'])  # Use consistent colors
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', (bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage')
    ax.set_title('Distribution of Differences')
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# Function to display a bar chart of the filtered drugs and their RMSEs
def display_rmse_chart(filtered_df, rmse_filter):
    st.sidebar.write(f"Number of drugs in this selection: {len(filtered_df)}")
    if not filtered_df.empty:
        fig = px.bar(filtered_df, x='Unnamed: 0', y='RMSE', labels={'Unnamed: 0': 'Drug', 'RMSE': 'RMSE'}, template='plotly_white')
        fig.update_layout(title='Filtered Drugs and RMSE', xaxis_title='Drug', yaxis_title='RMSE')
        st.plotly_chart(fig)
    else:
        st.write("No drugs found for the selected RMSE range.")

# Function to handle user queries about specific drugs


def get_highest_lowest_values(filtered_df, selected_drug):
    highest_actual = filtered_df['Actual'].max()
    lowest_actual = filtered_df['Actual'].min()
    highest_predicted = filtered_df['Predicted'].max()
    lowest_predicted = filtered_df['Predicted'].min()
    return {
        'highest_actual': highest_actual,
        'lowest_actual': lowest_actual,
        'highest_predicted': highest_predicted,
        'lowest_predicted': lowest_predicted
    }
def create_actual_vs_predicted_scatter_plot(df, selected_drug):
    # Filter the DataFrame for the selected drug
    filtered_df = df[df['Drug'] == selected_drug]

    # Scatter plot of actual vs predicted values
    fig = go.Figure()

    # Add scatter plot for actual values
    fig.add_trace(go.Scatter(
        x=filtered_df['Predicted'], y=filtered_df['Actual'],
        mode='markers',
        name='Actual vs Predicted',
        marker=dict(color='blue')
    ))

    # Add regression line for actual values
    actual_fit = np.polyfit(filtered_df['Predicted'], filtered_df['Actual'], 1)
    actual_fit_fn = np.poly1d(actual_fit)
    fig.add_trace(go.Scatter(
        x=filtered_df['Actual'],
        y=actual_fit_fn(filtered_df['Actual']),
        mode='lines',
        name='Actual Regression Line',
        line=dict(color='blue', dash='dash')
    ))

    # Add regression line for predicted values
    predicted_fit = np.polyfit(filtered_df['Predicted'], filtered_df['Actual'], 1)
    predicted_fit_fn = np.poly1d(predicted_fit)
    fig.add_trace(go.Scatter(
        x=filtered_df['Predicted'],
        y=predicted_fit_fn(filtered_df['Predicted']),
        mode='lines',
        name='Predicted Regression Line',
        line=dict(color='red', dash='dash')
    ))

    # Set the layout of the figure
    fig.update_layout(
        title=f"Actual vs Predicted Values for {selected_drug}",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        legend=dict(x=0.1, y=1.1),
        font=dict(family="Arial", size=12, color="black"),
        yaxis=dict(range=[0, max(filtered_df['Actual'].max(), filtered_df['Predicted'].max())]),
        xaxis=dict(range=[0, max(filtered_df['Actual'].max(), filtered_df['Predicted'].max())])
    )

    # Plot the figure using Streamlit
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout="wide")

    selection = option_menu(
        menu_title=None,
        options=["Home", "CTRP Data Analysis", "CCLE Data Analysis", "Make Predictions"],
        icons=["house", "clipboard-data", "bar-chart-line", "tools"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selection == "Home":
        # st.markdown("<div class='title'>Drug Model Results and Predictions</div>", unsafe_allow_html=True)

        # Load data
        model_results_df = load_model_results()
        actual_predicted_df = load_actual_predicted_data()

        # Display model results section
        # st.header("Drug Model Results")
        display_model_results(model_results_df)
        create_rmse_r2_scatter_plot(model_results_df)
        create_rmse_range_bar_plot(model_results_df)
        remove_mitochondrial = st.checkbox("Remove Mitochondrial Genes")
        create_genes_frequency_bar_plot(model_results_df, remove_mitochondrial)
        create_model_frequency_bar_plot(model_results_df)

        # Sidebar filters
        st.sidebar.header("Filters")
        rmse_filter = st.sidebar.selectbox("Filter Drugs by RMSE", ["All", "Less than 1", "Less than 1.5", "Less than 2", "Between 0 to 1", "Between 1 to 1.5", "Between 1.5 to 2"], index=0)
        filtered_model_results_df = filter_data_by_rmse(model_results_df, rmse_filter)

        # Display filtered results
        st.header("Filtered Drugs and RMSE")
        display_rmse_chart(filtered_model_results_df, rmse_filter)

        # Selected Drug Analysis section
        st.header("Selected Drug Analysis")
        selected_drug = st.selectbox("Select a Drug", sorted(filtered_model_results_df['Unnamed: 0']), index=0)
        filtered_df = display_actual_predicted_data(actual_predicted_df, model_results_df, selected_drug)

        st.sidebar.subheader("Select Graph Type")
        graph_type = st.sidebar.radio("Drug Analysis using Graphs:", ["Scatter Plot","Actual vs Predicted Scatter Plot", "Residuals Histogram", "Highest/Lowest Values", "Difference Ranges"])

        if graph_type == "Scatter Plot":
            create_scatter_plot(filtered_df, selected_drug)
        elif graph_type == "Actual vs Predicted Scatter Plot":
            create_actual_vs_predicted_scatter_plot(actual_predicted_df, selected_drug)
        elif graph_type == "Residuals Histogram":
            create_residuals_histogram(filtered_df, selected_drug)
        elif graph_type == "Highest/Lowest Values":
            create_highest_lowest_bar_plot(filtered_df, selected_drug)
        elif graph_type == "Difference Ranges":
            differences = filtered_df['Difference']
            percentages = calculate_difference_ranges(differences)
            create_difference_ranges_bar_plot(percentages)


    elif selection == "CTRP Data Analysis":
        display_ctrp_data_analysis()

    elif selection == "CCLE Data Analysis":
        display_ccle_data_analysis()

    elif selection == "Make Predictions":
        new_data()

if __name__ == "__main__":
    main()
