import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

#---------------------------------------------------------------------------------------------------------------------------------------------------
#PRE-PROCESSING OF TRAINING DATA

# Define function to extract constituency type
def extract_constituency_type(constituency):
    if "(ST)" in constituency:
        return 0
    elif "(SC)" in constituency:
        return 1
    else:
        return 2

# Define function to extract extra information
def extract_extra_info(name):
    if "Dr." in name:
        return 1
    elif "Adv." in name:
        return 2
    else:
        return 0

# Data preprocessing
def preprocess_data(df):
    # Convert 'Total Assets' and 'Liabilities' columns to numeric values
    df['Total Assets'] = df['Total Assets'].astype(str).str.replace('+', '') \
                         .str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5') \
                         .str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
    df['Total Assets'] = pd.to_numeric(df['Total Assets'], errors='coerce')
    
    df['Liabilities'] = df['Liabilities'].astype(str).str.replace('+', '') \
                        .str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5') \
                        .str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
    df['Liabilities'] = pd.to_numeric(df['Liabilities'], errors='coerce')
    
    # Extract features from 'Constituency ∇' and 'Candidate' columns
    df['Constituency_Type'] = df['Constituency ∇'].apply(extract_constituency_type)
    df['Extra'] = df['Candidate'].apply(extract_extra_info)
    
    # One-hot encode categorical columns ('Party' and 'state')
    one_hot_encoded_party = pd.get_dummies(df['Party'])
    one_hot_encoded_state = pd.get_dummies(df['state'])
    df = pd.concat([df, one_hot_encoded_party, one_hot_encoded_state], axis=1)
    
    # Replace zero values in 'Total Assets' and 'Liabilities' columns with mean values
    df['Total Assets'].replace(0, df['Total Assets'].mean(), inplace=True)
    df['Liabilities'].replace(0, df['Liabilities'].mean(), inplace=True)
    
    # Drop unnecessary columns
    columns_to_drop = ['ID', 'Candidate', 'state', 'Party', 'Constituency ∇']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    return df

# Load data
df = pd.read_csv("train.csv")

# Preprocess data
df = preprocess_data(df)
#---------------------------------------------------------------------------------------------------------------------------------------------------

#Training the model

# Define features (X) and target variable (y)
X = df.drop(['Education'], axis=1)  # Features are all columns except 'Education'
y = df['Education']  # Target variable is 'Education'

clf = BernoulliNB(alpha=0.38, binarize=0.0, fit_prior=True, class_prior=None)
clf.fit(X, y)

#---------------------------------------------------------------------------------------------------------------------------------------------------

# Load test data
df2 = pd.read_csv("test.csv")

# Drop unnecessary columns
columns_to_drop = ['ID']
df2.drop(columns_to_drop, axis=1, inplace=True)

# Convert 'Total Assets' and 'Liabilities' columns to numeric values
df2['Total Assets'] = df2['Total Assets'].astype(str).str.replace('+', '') \
                     .str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5') \
                     .str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
df2['Total Assets'] = pd.to_numeric(df2['Total Assets'], errors='coerce')

df2['Liabilities'] = df2['Liabilities'].astype(str).str.replace('+', '') \
                    .str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5') \
                    .str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
df2['Liabilities'] = pd.to_numeric(df2['Liabilities'], errors='coerce')

# Extract features from 'Constituency ∇' and 'Candidate' columns
df2['Constituency_Type'] = df2['Constituency ∇'].apply(extract_constituency_type)
df2['Extra'] = df2['Candidate'].apply(extract_extra_info)

# One-hot encode categorical columns ('Party' and 'state')
one_hot_encoded_party_test = pd.get_dummies(df2['Party'])
one_hot_encoded_state_test = pd.get_dummies(df2['state'])
df2 = pd.concat([df2, one_hot_encoded_party_test, one_hot_encoded_state_test], axis=1)

# Drop unnecessary columns
columns_to_drop_test = ['Constituency ∇', 'state', 'Party', 'Candidate']
df2.drop(columns_to_drop_test, axis=1, inplace=True)

# Replace zero values in 'Total Assets' and 'Liabilities' columns with mean values
df2['Total Assets'].replace(0, df2['Total Assets'].mean(), inplace=True)
df2['Liabilities'].replace(0, df2['Liabilities'].mean(), inplace=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Making predictions

y_pred_clf = clf.predict(df2)

# Create a DataFrame for the predictions
answer = pd.DataFrame(y_pred_clf, columns=["Education"])

# Save the predictions to a CSV file
answer.to_csv("answer_clf.csv", index=True)

df3 = pd.read_csv("train.csv")

# ---------------------------------------------------------------------------------------------------------------------------------------------------

#Plotting the data

import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------------------------
education_counts = df3['Education'].value_counts()

# Select the top 4 education levels and aggregate the rest
top_4_education = education_counts.head(4).index
df3['Education'] = df3['Education'].apply(lambda x: x if x in top_4_education else 'Others')

# Group data by Party and Education, and count the occurrences
grouped_data = df3.groupby(['Party', 'Education']).size().unstack().fillna(0)

# Compute percentage for each education level within each party
party_totals = grouped_data.sum(axis=1)
percentage_data = grouped_data.divide(party_totals, axis=0) * 100

# Custom color palette
colors = ['#FF5733', '#FFC300', '#C70039', '#900C3F', '#581845', '#003366']

# Plotting
plt.figure(figsize=(10, 6))
ax = percentage_data.plot(kind='bar', stacked=True, color=colors)

plt.title('Top 5 Parties vs Top 4 Education (Percentage)', fontsize=16)
plt.xlabel('Party', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=0)

# Adjust legend position
plt.legend(title='Education', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)

# Add percentage labels inside the bars
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=10)

plt.tight_layout()

# Save the plot as an image file (e.g., PNG)
plt.savefig('party_education_plot.png')

plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------

grouped_data = df.groupby(['Constituency_Type', 'Education']).size().unstack().fillna(0)

# Compute percentage for each education level within each constituency type
constituency_totals = grouped_data.sum(axis=1)
percentage_data = grouped_data.divide(constituency_totals, axis=0) * 100

# Plotting
plt.figure(figsize=(10, 6))
ax = percentage_data.plot(kind='bar', stacked=False)

plt.title('Constituency Type vs Education (Percentage)', fontsize=16)
plt.xlabel('Constituency Type', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=0)

# Adjust legend position
plt.legend(title='Education', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)

# Add percentage labels inside the bars with some padding
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.1f}%', (x + width / 2, y + height + 0.5), ha='center', va='center', fontsize=6)

plt.tight_layout()
plt.savefig('constituencyType_education_plot.png')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------

threshold_cases = 10  
filtered_df = df3[df3['Criminal Case'] > threshold_cases]

party_counts = filtered_df['Party'].value_counts(normalize=True) * 100

top_7_parties = party_counts.head(7).index
other_percentage = party_counts[~party_counts.index.isin(top_7_parties)].sum()
party_counts = party_counts.head(7)
party_counts['Others'] = other_percentage  # Add 'Others' with the calculated percentage

plt.figure(figsize=(10, 6))
ax = party_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
plt.title('Percentage Distribution of Parties with Candidates Having the Most Criminal Records', fontsize=16)
plt.xlabel('Party', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines

# Annotate each bar with its corresponding percentage value
for i, percentage in enumerate(party_counts):
    ax.text(i, percentage + 0.5, f'{percentage:.2f}%', ha='center', fontsize=10)

plt.text(6.5, 32, f'Threshold for criminal cases: {threshold_cases}', ha='right', fontsize=12, color='red')

plt.tight_layout()
plt.savefig('Percentage Distribution of Parties with Candidates Having the Most Criminal Records.png')

plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------

df3['Total Assets'] = df3['Total Assets'].astype(str).str.replace('+', '') \
                     .str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5') \
                     .str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
df3['Total Assets'] = pd.to_numeric(df3['Total Assets'], errors='coerce')

df3['Liabilities'] = df3['Liabilities'].astype(str).str.replace('+', '') \
                    .str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5') \
                    .str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
df3['Liabilities'] = pd.to_numeric(df3['Liabilities'], errors='coerce')

# Extract features from 'Constituency ∇' and 'Candidate' columns
df3['Constituency_Type'] = df3['Constituency ∇'].apply(extract_constituency_type)
df3['Extra'] = df3['Candidate'].apply(extract_extra_info)

# Replace zero values in 'Total Assets' and 'Liabilities' columns with mean values
df3['Total Assets'].replace(0, df3['Total Assets'].mean(), inplace=True)
df3['Liabilities'].replace(0, df3['Liabilities'].mean(), inplace=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Calculate the mean of the asset column to determine the threshold
threshold_asset = df3['Total Assets'].mean()  

# Step 2: Filter the DataFrame based on the threshold
filtered_df = df3[df3['Total Assets'] > threshold_asset]

# Step 3: Calculate percentage distribution of parties
party_counts = filtered_df['Party'].value_counts(normalize=True) * 100

# Step 4: Plot the percentage distribution of parties
plt.figure(figsize=(10, 6))
ax = party_counts.plot(kind='bar', color='skyblue')
plt.title('Percentage Distribution of Parties with Candidates Having Assets Above Threshold', fontsize=16)
plt.xlabel('Party', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines

# Annotate each bar with its corresponding percentage value
for i, percentage in enumerate(party_counts):
    ax.text(i, percentage + 0.5, f'{percentage:.2f}%', ha='center', fontsize=10)

# Add text to indicate the threshold for assets at the top right corner
plt.text(len(party_counts) - 3, party_counts.max() -4, f'Threshold for considering wealthy: {threshold_asset:.2f}', ha='right', fontsize=12, color='red')
plt.savefig('Percentage Distribution of Parties with the Most Wealthy Candidates.png')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------