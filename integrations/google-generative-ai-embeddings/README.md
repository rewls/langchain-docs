# Google Generative AI Embeddings

## Installation

```python
%pip install --upgrade --quiet langchain-google-genai
```

## Usage

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("hello, wordl!")
vector[:5]
```
