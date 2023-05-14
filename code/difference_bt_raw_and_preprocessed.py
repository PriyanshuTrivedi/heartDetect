import train_test
import pandas as pd

# Mainly converted cp into 4 columns,thal to 4 and slope to 3 and dropping unnecessary variables

initial=train_test.raw_df
processed=train_test.processed_df

with pd.ExcelWriter("data.xlsx") as writer:
    initial.to_excel(writer,sheet_name="initial")
    processed.to_excel(writer,sheet_name="processed")