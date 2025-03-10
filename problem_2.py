import nltk
import ssl
from transformers import pipeline
import certifi

# Configure SSL certificates
ssl._create_default_https_context = ssl._create_unverified_context

# Download the punkt tokenizer
nltk.download('punkt_tab')

# Load the rental agreement text
# rental_agreement = """
# 1. The lessee shall pay a monthly rent of $1000.
# 2. The lessor is responsible for all maintenance and repairs.
# 3. The lessee must pay for any repairs exceeding $200.
# 4. The lease can be terminated by either party with a 30-day notice.
# 5. The lease can be terminated by the lessor with a 60-day notice.
# """
file_name = "rental_agreement1.text"
with open(file_name) as f:
    rental_agreement = f.read()
print(rental_agreement)

# Tokenize the text into clauses
clauses = nltk.sent_tokenize(rental_agreement)

# Initialize sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Function to classify clauses
def classify_clause(clause):
    sentiment = classifier(clause)[0]
    if "lessee" in clause.lower():
        if sentiment['label'] == 'NEGATIVE':
            return "Favorable to Lessor", "Shifts responsibility to lessee"
        else:
            return "Favorable to Lessee", "Benefits the lessee"
    elif "lessor" in clause.lower():
        if sentiment['label'] == 'NEGATIVE':
            return "Favorable to Lessee", "Shifts responsibility to lessor"
        else:
            return "Favorable to Lessor", "Benefits the lessor"
    else:
        return "Neutral", "Equal terms for both parties"

# Classify each clause
classified_clauses = [(clause, *classify_clause(clause)) for clause in clauses]

# Print classified clauses
for clause, classification, reason in classified_clauses:
    print(f"Clause: {clause}\nClassification: {classification}\nReason: {reason}\n")

# Identify contradictions
def find_contradictions(clauses):
    contradictions = []
    for i, clause1 in enumerate(clauses):
        for j, clause2 in enumerate(clauses):
            if i != j and "terminate" in clause1.lower() and "terminate" in clause2.lower():
                if "30-day" in clause1.lower() and "60-day" in clause2.lower():
                    contradictions.append((clause1, clause2))
    return contradictions

# Find and print contradictions
contradictions = find_contradictions(clauses)
for clause1, clause2 in contradictions:
    print(f"Contradiction found between:\nClause 1: {clause1}\nClause 2: {clause2}\n")
