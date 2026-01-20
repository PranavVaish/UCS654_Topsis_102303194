import sys
import pandas as pd
import numpy as np
import os

def check_numeric(df):
    try:
        # Select all columns from index 1 to the end
        data_part = df.iloc[:, 1:]
        # Attempt to convert to float; this will raise an error if non-numeric data exists
        data_part.astype(float)
        return True
    except Exception:
        return False

def topsis(input_file, weights, impacts, result_file):
    try:
        # File Not Found Handler
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Error: The file '{input_file}' was not found.")

        # Read the CSV file
        df = pd.read_csv(input_file)

        # Check for minimum columns
        if df.shape[1] < 3:
            raise ValueError("Error: Input file must contain three or more columns.")

        # Check for non-numeric values
        if not check_numeric(df):
            raise ValueError("Error: From 2nd to last columns must contain numeric values only.")

        # Pre-process inputs
        # Convert weights string "1,1,1,2" to list of floats
        try:
            weight_list = [float(w) for w in weights.split(',')]
        except ValueError:
             raise ValueError("Error: Weights must be numeric and separated by commas.")
             
        # Convert impacts string "+,+,-,+" to list
        impact_list = impacts.split(',')

        # Dimensions of the numeric part of data
        num_cols = df.shape[1] - 1 # Subtracting the first name/ID column

        # Check parameter counts match
        if len(weight_list) != num_cols or len(impact_list) != num_cols:
            raise ValueError(f"Error: Number of weights ({len(weight_list)}), impacts ({len(impact_list)}), "
                             f"and columns ({num_cols}) must be the same.")

        # Check Impact Symbols are valid
        if not all(i in ['+', '-'] for i in impact_list):
            raise ValueError("Error: Impacts must be either '+' or '-'.")

        # TOPSIS ALGORITHM IMPLEMENTATION 
        
        # 1. Convert data to numpy array for easier calculation
        data = df.iloc[:, 1:].values.astype(float)
        
        # 2. Normalize the matrix
        # Square root of sum of squares for each column
        rss = np.sqrt(np.sum(data**2, axis=0))
        normalized_data = data / rss

        # 3. Weighted Normalization
        weighted_data = normalized_data * weight_list

        # 4. Ideal Best and Ideal Worst
        ideal_best = []
        ideal_worst = []

        for i in range(num_cols):
            if impact_list[i] == '+':
                ideal_best.append(np.max(weighted_data[:, i]))
                ideal_worst.append(np.min(weighted_data[:, i]))
            else: # Impact is '-'
                ideal_best.append(np.min(weighted_data[:, i]))
                ideal_worst.append(np.max(weighted_data[:, i]))
        
        ideal_best = np.array(ideal_best)
        ideal_worst = np.array(ideal_worst)

        # 5. Euclidean Distance
        # Distance from Best
        s_plus = np.sqrt(np.sum((weighted_data - ideal_best)**2, axis=1))
        # Distance from Worst
        s_minus = np.sqrt(np.sum((weighted_data - ideal_worst)**2, axis=1))

        # 6. Performance Score
        # Handle division by zero edge case if s_plus + s_minus is 0
        total_dist = s_plus + s_minus
        performance_score = np.divide(s_minus, total_dist, out=np.zeros_like(s_minus), where=total_dist!=0)

        # 7. Append to DataFrame
        df['Topsis Score'] = performance_score
        
        # 8. Rank (Higher score is better)
        df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)

        # 9. Save to output file
        df.to_csv(result_file, index=False)
        print(f"Success: TOPSIS results saved to '{result_file}'")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(val_error)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Check for Correct Number of Parameters
    if len(sys.argv) != 5:
        print("Error: Wrong number of parameters.")
        print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <OutputResultFileName>")
        print('Example: python topsis.py data.csv "1,1,1,2" "+,+,-,+" result.csv')
    else:
        # Parse arguments
        input_file = sys.argv[1]
        weights = sys.argv[2]
        impacts = sys.argv[3]
        result_file = sys.argv[4]
        
        topsis(input_file, weights, impacts, result_file)