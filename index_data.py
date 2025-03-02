# read data from cleaned_markdown_results and create a vector index
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# read in all files in cleaned_markdown_results
files = os.listdir("cleaned_markdown_results")
# create a directory called vector_indices if it doesn't exist
if not os.path.exists("vector_indices_e5"):
    os.makedirs("vector_indices_e5")

# read in each file and create a vector index
for file in files:
    with open(f"cleaned_markdown_results/{file}", "r") as f:
        data = f.read()
        # create a vector index using E5
        
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        embeddings = model.encode(data)
        # save the vector index
        np.save(f"vector_indices_e5/{file}.npy", embeddings)
        
