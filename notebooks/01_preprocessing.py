import pandas as pd

# Load data from absolute paths
EEG = pd.read_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\EEG.csv', low_memory=False)
GSR = pd.read_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\GSR.csv', low_memory=False)
EYE = pd.read_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\EYE.csv', low_memory=False)
IVT = pd.read_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\IVT.csv', low_memory=False)
ENG = pd.read_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\ENG.csv', low_memory=False)


# Helper to detect timestamp column
def get_timestamp_column(df):
    for col in df.columns:
        if "time" in col.lower():  # matches Time, timestamp, datetime, etc.
            return col
    return None


# Preprocess function
def preprocess(df, name):
    ts_col = get_timestamp_column(df)
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {name}, available columns: {df.columns.tolist()}")

    # Convert to datetime
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df.set_index(ts_col, inplace=True)

    # 🔑 Drop duplicate timestamps
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        print(f"⚠️ {name}: {dup_count} duplicate timestamps found — keeping first occurrence")
        df = df[~df.index.duplicated(keep="first")]

    # Separate numeric and non-numeric
    numeric_df = df.select_dtypes(include="number")
    non_numeric_df = df.select_dtypes(exclude="number")

    # Resample numeric
    numeric_resampled = numeric_df.resample("1s").mean().interpolate()

    # Resample non-numeric (forward fill)
    if not non_numeric_df.empty:
        non_numeric_resampled = non_numeric_df.resample("1s").ffill()
        df_resampled = numeric_resampled.join(non_numeric_resampled, how="left")
    else:
        df_resampled = numeric_resampled

    print(f"✅ {name}: processed {len(df)} → {len(df_resampled)} rows")
    return df_resampled


# Run preprocessing
EEG = preprocess(EEG, "EEG")
GSR = preprocess(GSR, "GSR")
EYE = preprocess(EYE, "EYE")
IVT = preprocess(IVT, "IVT")
ENG = preprocess(ENG, "ENG")

# Save processed data
EEG.to_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\EEG_processed.csv')
GSR.to_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\GSR_processed.csv')
EYE.to_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\EYE_processed.csv')
IVT.to_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\IVT_processed.csv')
ENG.to_csv(r'C:\Users\Riya\IIT_EDA_Internship\Data\ENG_processed.csv')

print("🎉 All files processed and saved successfully!")
