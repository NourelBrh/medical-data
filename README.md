# medical-data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("medical_examination.csv")

# Task 1: Add an overweight column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Task 2: Normalize cholesterol and gluc columns
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Task 3: Convert the data into long format and create a chart
df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active'])

# Create the catplot
plt.figure(figsize=(10, 5))
cat_plot = sns.catplot(data=df_cat, kind='count', x='variable', hue='value', col='cardio')
cat_plot.set_axis_labels("Variable", "Total Count")
cat_plot.set_titles("{col_name} {col_var}")
plt.show()

# Task 4: Clean the data
df_cleaned = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975))]

# Task 5: Create a correlation matrix and plot it
corr_matrix = df_cleaned.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5, mask=mask)
plt.title("Correlation Matrix")
plt.show()
