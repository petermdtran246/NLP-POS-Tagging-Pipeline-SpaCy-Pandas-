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

# ============================================
# STEP 1 — LOAD LIBRARIES & NLP MODEL
# ============================================

import spacy
import pandas as pd

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')


# ============================================
# STEP 2 — RAW TEXT INPUT
# (text from Jane Austen — pre-lowercased + no punctuation)
# ============================================

emma_ja = (
    'emma woodhouse handsome clever and rich with a comfortable home and happy disposition ...'
)


# ============================================
# STEP 3 — RUN THE NLP PIPELINE (TOKENIZATION + POS TAGGING)
# ============================================

spacy_doc = nlp(emma_ja)

# View token + POS + lemma + tag (first 10 tokens)
for t in spacy_doc[:10]:
    print(t.text, t.lemma_, t.pos_, t.tag_)    # With "_" → human-readable
                                               # Without "_" → machine numeric ID


# ============================================
# STEP 4 — BUILD A DATAFRAME OF TOKEN + POS TAG
# (Approach 2 — FASTEST + MOST READABLE)
# ============================================

rows = [{'token': token.text, 'pos_tag': token.pos_} for token in spacy_doc]
pos_df = pd.DataFrame(rows)

# pos_df now looks like:
# token     pos_tag
# emma      PROPN
# woodhouse PROPN
# handsome  ADJ
# clever    ADJ
# and       CCONJ
# ...


# ============================================
# STEP 5 — COUNT UNIQUE TOKENS PER POS
# (Approach 2 — value_counts is faster)
# ============================================

pos_df_counts = (
    pos_df
        .value_counts(['token', 'pos_tag'])        # → counts per token+POS (creates MultiIndex)
        .reset_index(name='counts')               # → convert MultiIndex → DataFrame
        .sort_values('counts', ascending=False)   # → sort highest freq first
)

# pos_df_counts now contains:
# token      pos_tag     counts
# of         ADP         14
# her        PRON        9
# had        AUX         9
# ...


# ============================================
# STEP 6 — COUNT HOW MANY UNIQUE WORDS BELONG TO EACH POS
# ============================================

pos_df_poscounts = (
    pos_df_counts['pos_tag']
        .value_counts()                   # → counts how many unique tokens per POS
        .sort_values(ascending=False)
)

# Example output:
# NOUN    35
# VERB    19
# ADJ     18
# ADV     18
# ...


# ============================================
# STEP 7 — FILTER TOP NOUNS (OR ANY POS)
# ============================================

top_nouns = (
    pos_df_counts
        [pos_df_counts['pos_tag'] == 'NOUN']   # → filter NOUN
        .head(10)                              # → take top 10
)

# Result:
# governess   NOUN   3
# friends     NOUN   3
# mother      NOUN   2
# daughters   NOUN   2
# ...


# ============================================
# END OF PIPELINE
# ============================================



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


