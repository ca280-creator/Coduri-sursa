import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

#Citire date
file_path = "/content/sample_data/consumption_user.csv"
cols_to_use = ['SUBJECT', 'SURVEY_DAY', 'INGREDIENT_ENG']

df = pd.read_csv(file_path, usecols=cols_to_use)
df = df.dropna(subset=['INGREDIENT_ENG'])
df['INGREDIENT_ENG'] = df['INGREDIENT_ENG'].str.strip().str.lower()

#Sample mai mic pentru rapiditate
df = df.sample(n=30000, random_state=42) 
print(f"Dimensiune esantion analizat: {df.shape[0]} linii.")

#Top 50 cele mai frecvente ingrediente din acest esantion
#pentru a reduce complexitatea combinatoriala
top_items = df['INGREDIENT_ENG'].value_counts().head(50).index
df_filtered = df[df['INGREDIENT_ENG'].isin(top_items)]

transactions = df_filtered.groupby(['SUBJECT', 'SURVEY_DAY'])['INGREDIENT_ENG'].apply(list).tolist()
print(f"Nr tranzactii unice: {len(transactions)}")

#Preprocesare (One-Hot)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)
print(f"Matrice: {df_trans.shape}")

#FP-Growth (Executie)
#Suport 0.02 (2%)
MIN_SUPPORT = 0.02 

start_time = time.time()
frequent_itemsets = fpgrowth(df_trans, min_support=MIN_SUPPORT, use_colnames=True)
end_time = time.time()

print(f"Timp executie: {end_time - start_time:.4f} secunde")
print(f"Itemset-uri identificate: {len(frequent_itemsets)}")

#Reguli
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values(by='lift', ascending=False)
    
    print("\n TOP 5 REGULI GASITE")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
    
    #Salveaza pentru raport
    rules.to_csv('rezultate_finale.csv', index=False)
    print("\nFisierul 'rezultate_finale.csv' a fost salvat.")
else:
    print("Nu s-au gasit reguli.")