import os
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

os.environ['OPENAI_API_KEY'] = "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"

openai_api_key = os.getenv("OPENAI_API_KEY")
print("openai_api_key_____",openai_api_key)
openai.api_key=openai_api_key

def descrption(field, customer_name, value):
    # print(type(value))
    match field:
        case "Unique ID":
            return "The Unique ID for "+ customer_name+" is "+value
        case "Created on":
            return "This project was created at "+value
        # case "Created by":
        #     if(value=="Projects"):
        #         return ""
        #     else:
        #         return ""
        case "Customer Full Name":
            return "The full name of customer is "+value
        case "Date Created":
            return "The project is created at" + value
        case "Estimated Install Date set at sale - start":
            return ""
        case default:
            return "The "+field+" for "+customer_name+" is "+value



################################################################################
### Step 1
################################################################################

headers=[]
df=pd.read_csv('processed/initial.csv', low_memory=False)
headers = next(df.iterrows())[0]


df=pd.read_csv('processed/initial.csv', low_memory=False, header=1)

texts=[]
for row in df.iterrows():
    text=''

    for index, header_cell in enumerate(headers):
        if(header_cell != 'Customer Full Name'):
            value = str(row[1][header_cell])
            customer_fullname= str(row[1]['Customer Full Name'])
            text += descrption(header_cell, customer_fullname, value) +"."
            # text += header_cell+": "+str(row[1][header_cell])
            # if(index != len(headers)-1):
            #     text += "; "

    texts.append((str(row[1][headers[0]]),text))


# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = [headers[0], 'text'])
df['text'] = df[headers[0]] + ". " + df.text
df.to_csv('processed/scraped.csv')
df.head()

print('end scrapping')
################################################################################
### Step 2
################################################################################

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = [headers[0], 'text']
# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

################################################################################
### Step 3
################################################################################

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

shortened = []

# Loop through the dataframe
for index, row in enumerate(df.iterrows()):
    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )

################################################################################
### Step 4
################################################################################

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()

################################################################################
### Step 5
################################################################################

# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits
print('start embedding')
df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
