import gradio as gr
import pandas as pd

def analyze_data(file):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(file)

    # Calculate basic statistics on the data
    mean = data.mean()
    median = data.median()
    mode = data.mode().iloc[0]
    std_dev = data.std()

    # Create a dictionary of the results
    results = {
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Standard Deviation": std_dev
    }

    # Convert the results to a pandas DataFrame and return it
    return pd.DataFrame(results)

# Define the Gradio interface
iface = gr.Interface(
    analyze_data,
    "file",
    "dataframe",
    title="Statistical Analysis",
    description="Upload a CSV file and get basic statistics on the data. The dataset must include results labelled with mean, median, mode, and standard deviation.",
    theme="compact"
)

# Launch the Gradio interface
iface.launch()