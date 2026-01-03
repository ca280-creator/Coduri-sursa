import pandas as pd
import time
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

#Citire date
file_path = "/content/sample_data/consumption_user.csv"
cols_to_use = ['SUBJECT', 'SURVEY_DAY', 'INGREDIENT_ENG']

df = pd.read_csv(file_path, usecols=cols_to_use)
df = df.dropna(subset=['INGREDIENT_ENG'])
df['INGREDIENT_ENG'] = df['INGREDIENT_ENG'].str.strip().str.lower()

#Esantionare (sampling)
#Sample mai mic pentru comparatie
df = df.sample(n=30000, random_state=42) 

#Top 50 cele mai frecvente ingrediente din acest esantion
#pentru a reduce complexitatea combinatoriala
top_items = df['INGREDIENT_ENG'].value_counts().head(50).index
df_filtered = df[df['INGREDIENT_ENG'].isin(top_items)]

#Creare tranzactii
transactions = df_filtered.groupby(['SUBJECT', 'SURVEY_DAY'])['INGREDIENT_ENG'].apply(list).tolist()
print(f"Nr tranzactii analizate: {len(transactions)}")

#Preprocesare (One-Hot)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)
print(f"Matricea One-Hot creata: {df_trans.shape}")

#Apriori
#Parametrii identici cu FP-Growth (min_support=0.02)
start_time = time.time()

#Functia apriori
frequent_itemsets = apriori(df_trans, min_support=0.02, use_colnames=True)

end_time = time.time()
execution_time = end_time - start_time

print(f"Timp de executie APRIORI: {execution_time:.4f} secunde")
print(f"Itemset-uri identificate: {len(frequent_itemsets)}")

#Adaugam coloana de lungime
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

#Generare reguli
if not frequent_itemsets.empty: 
    #Generare reguli bazate pe metric="lift"
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    #Calculare lungime antecedent
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    
    #Filtrare si sortare pentru afisare
    #Cautare reguli unde lift > 1.1 si confidence > 0.1
    best_rules = rules[ (rules['lift'] > 1.1) & (rules['confidence'] > 0.1) ]
    best_rules = best_rules.sort_values(by='lift', ascending=False)
    
    print(f"Nr reguli generate: {len(rules)}")
    print("\nTop 5 Reguli (Apriori):")
    print(best_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
    
    # Salvare pentru raport
    best_rules.to_csv('rezultate_apriori.csv', index=False)
    print("\nRezultatele au fost salvate in 'rezultate_apriori.csv'")

else:
    print("Nu s-au gasit itemset-uri.")