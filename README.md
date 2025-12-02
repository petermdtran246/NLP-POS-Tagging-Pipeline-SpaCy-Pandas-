📘 NLP POS Tagging Pipeline (SpaCy + Pandas)

A clean, optimized, production-style POS tagging pipeline using SpaCy & Pandas.

This project demonstrates how to:
✅ Tokenize text
✅ Extract POS tags from each token
✅ Build a clean DataFrame
✅ Count token–POS frequencies
✅ Identify top nouns (or any POS)
✅ Perform linguistic analysis with SpaCy


🚀 Features

✔️ Clean, commented, readable code

✔️ Method-chaining Pandas workflow

✔️ Fast list-comprehension token extraction

✔️ Human-readable POS tags (using SpaCy _ attributes)

✔️ Easy to extend to NER, dependency parsing, sentiment, etc.


🧠 Tech Used
| Library    | Purpose                    |
| ---------- | -------------------------- |
| **SpaCy**  | Tokenization + POS tagging |
| **Pandas** | Data analysis & grouping   |
| **Python** | Glue everything together   |


📄 Full Code (POS Tagging Pipeline)

🧩 Step 1 — Load Libraries & NLP Model
import spacy
import pandas as pd

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

📄 Step 2 — Raw Text Input
emma_ja = (
    "emma woodhouse handsome clever and rich with a comfortable home..."
)

⚙️ Step 3 — Run NLP Pipeline (Tokenization + POS Tagging)
spacy_doc = nlp(emma_ja)

# View first 10 tokens
for t in spacy_doc[:10]:
    print(t.text, t.lemma_, t.pos_, t.tag_)

🏗️ Step 4 — Build DataFrame of Token + POS
rows = [{'token': token.text, 'pos_tag': token.pos_} for token in spacy_doc]
pos_df = pd.DataFrame(rows)

📊 Step 5 — Count Unique Tokens per POS (Fast Method)
pos_df_counts = (
    pos_df
        .value_counts(['token', 'pos_tag'])
        .reset_index(name='counts')
        .sort_values('counts', ascending=False)
)

📈 Step 6 — Count How Many Unique Words Belong to Each POS
pos_df_poscounts = (
    pos_df_counts['pos_tag']
        .value_counts()
        .sort_values(ascending=False)
)

🔎 Step 7 — Filter Top Nouns
top_nouns = (
    pos_df_counts[pos_df_counts['pos_tag'] == 'NOUN']
        .head(10)
)



📊 Sample Output Visualization
🔹 POS Frequency Table

| POS  | Unique Words |
| ---- | ------------ |
| NOUN | 35           |
| VERB | 19           |
| ADJ  | 18           |
| ADV  | 18           |
| PRON | 9            |
| ADP  | 8            |



🔹 Top 10 Most Frequent Nouns

| Token     | POS  | Count |
| --------- | ---- | ----- |
| governess | NOUN | 3     |
| friends   | NOUN | 3     |
| mother    | NOUN | 2     |
| daughters | NOUN | 2     |
| sisters   | NOUN | 2     |


