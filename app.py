import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- TOPSIS Logic ---
def topsis(df, weights, impacts):
    try:
        # Extract data (assume first column is ID/Name)
        data = df.iloc[:, 1:].values.astype(float)
        names = df.iloc[:, 0].values
        
        # Validation
        num_cols = data.shape[1]
        if len(weights) != num_cols or len(impacts) != num_cols:
            return None, f"Error: Input has {num_cols} columns, but you provided {len(weights)} weights and {len(impacts)} impacts."
        
        # 1. Normalize
        rss = np.sqrt(np.sum(data**2, axis=0))
        normalized_data = data / rss
        
        # 2. Weighted Normalization
        weighted_data = normalized_data * weights
        
        # 3. Ideal Best & Worst
        ideal_best = []
        ideal_worst = []
        for i in range(num_cols):
            if impacts[i] == '+':
                ideal_best.append(np.max(weighted_data[:, i]))
                ideal_worst.append(np.min(weighted_data[:, i]))
            else: # Impact is '-'
                ideal_best.append(np.min(weighted_data[:, i]))
                ideal_worst.append(np.max(weighted_data[:, i]))
                
        # 4. Euclidean Distance
        s_plus = np.sqrt(np.sum((weighted_data - np.array(ideal_best))**2, axis=1))
        s_minus = np.sqrt(np.sum((weighted_data - np.array(ideal_worst))**2, axis=1))
        
        # 5. Score
        total_dist = s_plus + s_minus
        scores = np.divide(s_minus, total_dist, out=np.zeros_like(s_minus), where=total_dist!=0)
        
        # Append results
        df['Topsis Score'] = scores
        df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)
        
        return df, "Success"
        
    except Exception as e:
        return None, str(e)

# --- Streamlit UI ---
st.set_page_config(page_title="TOPSIS Web Service", layout="wide")

st.title("📊 TOPSIS Decision Support System")
st.markdown("Upload your dataset, define criteria weights/impacts, and get the ranked results instantly.")

# 1. File Upload
uploaded_file = st.file_uploader("Upload CSV file (First column must be names/IDs)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())
    
    col_count = df.shape[1] - 1
    st.info(f"Detected {col_count} criteria columns.")

    # 2. User Inputs
    col1, col2 = st.columns(2)
    with col1:
        weights_input = st.text_input("Enter Weights (comma separated)", value=",".join(["1"]*col_count))
    with col2:
        impacts_input = st.text_input("Enter Impacts (+ or - separated by comma)", value=",".join(["+"]*col_count))

    # 3. Calculate Button
    if st.button("Run TOPSIS Analysis"):
        # Parse inputs
        try:
            weights = [float(w) for w in weights_input.split(',')]
            impacts = impacts_input.split(',')
            
            # Run Algorithm
            result_df, msg = topsis(df.copy(), weights, impacts)
            
            if result_df is not None:
                st.success("Analysis Complete!")
                
                # Display Results
                st.write("### Ranked Results")
                st.dataframe(result_df)
                
                # Visualizations
                st.write("### Visualization")
                
                # Bar Chart
                fig, ax = plt.subplots(figsize=(10, 5))
                sorted_df = result_df.sort_values(by="Topsis Score", ascending=False)
                ax.bar(sorted_df.iloc[:, 0], sorted_df['Topsis Score'], color='teal')
                ax.set_xlabel("Alternatives")
                ax.set_ylabel("TOPSIS Score")
                ax.set_title("Performance Ranking")
                st.pyplot(fig)
                
                # Download Button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Result CSV",
                    data=csv,
                    file_name="topsis_result.csv",
                    mime="text/csv",
                )
            else:
                st.error(msg)
        except ValueError:
            st.error("Error: Please ensure weights are numeric and impacts are strictly '+' or '-'.")

else:
    st.info("Awaiting CSV file upload...")