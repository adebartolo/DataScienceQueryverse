import pandas as pd
from io import StringIO

# One way to fake data

data = """
date,app_name,ad_slot,page_type,page_section,vendor,clicks,impressions,revenue,capacity
2026-02-01 00:00:00,mw-c,banner,article,news,GAM,10,1000,100,1005
2026-02-01 00:00:00,mw-c,banner,article,news,GAM,15,1200,200,1205
2026-02-01 00:00:00,mw-c,banner,article,news,GAM,20,1300,300,1305
2026-02-01 00:00:00,mw-c,banner,article,news,GAM,25,1400,400,1405
2026-02-01 00:00:00,mw-c,banner,article,news,GAM,30,1500,500,1505
2026-02-01 00:00:00,mw-t,banner,article,news,GAM,10,1000,100,1005
2026-02-01 00:00:00,mw-t,banner,article,news,GAM,15,1200,200,1205
"""

# Create DataFrame
df_example = pd.read_csv(StringIO(data), parse_dates=["date"])
print(df_example)

######

# Save df to file

df_example.to_excel("ab_test_results.xlsx", index=False)

print("Saved results to 'ab_test_results.xlsx'")

######

# Clean up column names

df = pd.DataFrame(columns=["1. App Name", "5. Total Imps", "Revenue $", "1000.garbage"])

# Clean column names
df.columns = (
    df.columns
      .str.strip()                     # remove leading/trailing spaces
      .str.replace(r'^\d+\.\s*', '', regex=True)  # remove leading numbers like "1. "
      .str.replace(r'\s+', '_', regex=True)      # replace spaces with underscore
      .str.lower()                     # lowercase
)

print(df.columns)
