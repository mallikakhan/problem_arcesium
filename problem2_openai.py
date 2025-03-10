import openai
import nltk
import time
import ssl


# Configure SSL certificates
ssl._create_default_https_context = ssl._create_unverified_context

# Download the punkt tokenizer
nltk.download('punkt')

# Set your OpenAI API key
openai.api_key = ''
openai_model = 'gpt-4o-mini'
# Load the rental agreement text
rental_agreement = """
1. The lessee shall pay a monthly rent of $1000.
2. The lessor is responsible for all maintenance and repairs.
3. The lessee must pay for any repairs exceeding $200.
4. The lease can be terminated by either party with a 30-day notice.
5. The lease can be terminated by the lessor with a 60-day notice.
"""

# Tokenize the text into clauses
clauses = nltk.sent_tokenize(rental_agreement)

# Function to classify clauses using GPT-3.5-turbo with batch processing and retry mechanism
def classify_clauses(clauses):
    prompt = "Classify the following clauses as Favorable to Lessor, Favorable to Lessee, or Neutral, and provide the reason why:\n\n"
    for i, clause in enumerate(clauses):
        prompt += f"Clause {i+1}: {clause}\n"
    prompt += "\nProvide the classification and reason for each clause."

    retries = 3
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            classifications = response['choices'][0]['message']['content'].strip().split('\n')
            return classifications
        except openai.error.RateLimitError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

# Classify all clauses in a single call
classifications = classify_clauses(clauses)

# Print classified clauses
for i, (clause, classification) in enumerate(zip(clauses, classifications)):
    print(f"Clause {i+1}: {clause}\nClassification: {classification}\n")

# Function to identify contradictions using GPT-3.5-turbo with batch processing and retry mechanism
def find_contradictions(clauses):
    prompt = "Identify if there are any contradictions between the following clauses:\n\n"
    for i, clause1 in enumerate(clauses):
        for j, clause2 in enumerate(clauses):
            if i != j:
                prompt += f"Clause {i+1}: {clause1}\nClause {j+1}: {clause2}\n\n"

    retries = 3
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            contradictions = response['choices'][0]['message']['content'].strip().split('\n')
            return contradictions
        except openai.error.RateLimitError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

# Find and print contradictions in a single call
contradictions = find_contradictions(clauses)
for contradiction in contradictions:
    if contradiction:
        print(f"Contradiction found: {contradiction}\n")


""" OUTPUT from GPT
Clause 1: 
1.
Classification: Here is the classification and reasoning for each clause:

Clause 2: The lessee shall pay a monthly rent of $1000.
Classification: 

Clause 3: 2.
Classification: **Clause 1:** (No content, cannot classify)  

Clause 4: The lessor is responsible for all maintenance and repairs.
Classification: - **Classification:** Neutral  

Clause 5: 3.
Classification: - **Reason:** There is no content to analyze.

Clause 6: The lessee must pay for any repairs exceeding $200.
Classification: 

Clause 7: 4.
Classification: **Clause 2:** The lessee shall pay a monthly rent of $1000.  

Clause 8: The lease can be terminated by either party with a 30-day notice.
Classification: - **Classification:** Favorable to Lessor  

Clause 9: 5.
Classification: - **Reason:** This clause clearly specifies the financial obligation of the lessee (tenant) to pay rent, which is typically seen as a favorable condition for the lessor (landlord) since it guarantees income.

Clause 10: The lease can be terminated by the lessor with a 60-day notice.
Classification: 

Contradiction found: Based on the provided clauses, there are contradictions between some of them:

Contradiction found: 1. **Clause 8 vs. Clause 10**:

Contradiction found:    - Clause 8 states, "The lease can be terminated by either party with a 30-day notice."

Contradiction found:    - Clause 10 states, "The lease can be terminated by the lessor with a 60-day notice."

Contradiction found:    

Contradiction found:    Here lies a contradiction: If either party can terminate the lease with a 30-day notice, that would imply the lessor can also terminate with the shorter notice period. Therefore, the imposition of a longer 60-day notice for the lessor contradicts Clause 8.

Contradiction found: 2. **General Contradiction on Maintenance Responsibilities**:

Contradiction found:    - Clause 4 states, "The lessor is responsible for all maintenance and repairs."

Contradiction found:    - Clause 6 states, "The lessee must pay for any repairs exceeding $200."

Contradiction found:    

Contradiction found:    The two clauses can create a confusion in the interpretation of responsibility regarding repairs. If the lessor is fully responsible for maintenance and repairs, the clause about the lessee paying for repairs exceeding $200 may imply a shared responsibility or conditions under which the lessee is liable. However, without additional context on what constitutes "repairs exceeding $200," this can lead to confusion regarding the obligations of each party.

Contradiction found: In summary, the primary contradictions are between clauses regarding termination notice and maintenance responsibilities, which may lead to confusion or conflicting obligations.


Process finished with exit code 0

"""