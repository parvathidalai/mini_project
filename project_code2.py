import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv('instamart2.csv', delimiter=';', on_bad_lines='skip', dtype={'BillNo': str}, low_memory=False)


# Clean and filter data
df = df[['BillNo', 'Itemname']].dropna()
df['BillNo'] = df['BillNo'].astype(str)
df['Itemname'] = df['Itemname'].astype(str)

# Remove canceled orders (BillNo starting with 'C')
df = df[~df['BillNo'].str.startswith('C')]

# Group by transaction
basket = df.groupby('BillNo')['Itemname'].apply(list)

# One-hot encoding
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

# Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Output results
print("Top Frequent Itemsets:\n", frequent_itemsets.sort_values(by='support', ascending=False).head(10))
print("\nTop Association Rules:\n", rules.sort_values(by='lift', ascending=False).head(10))
