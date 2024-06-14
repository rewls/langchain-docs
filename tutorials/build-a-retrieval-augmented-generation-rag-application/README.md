# Build a Retrieval Augmented Generation (RAG) App

- This tutorial will show how to build a simple Q&A application over a text data source.

- We'll also see how LangSmith can help us trace and understand our application.

## What is RAG?

- RAG is technique for augmenting LLM knowledge with additional data.

- LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on.

- The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

- LangChain has a number of components designed to help build Q&A applications, and RAG application more generally.

## Concepts

- A typical RAG application has two main components:

    - Indexing: a pipeline for ingesting data from a source and indexing it.

        - This usually happens offline.

    - Retrieval and generation: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

### Indexing

1. Load: First we need to load our data.

    - This is done with DocumentLoaders.

2. Split: Text splitters break large `Documents` into smaller chunks.

    - This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won't fit in a model's finite context window.

3. Store: We need somewhere to store and index out splits, so that they can later be searched over.

    - This is often done using a VectorStore and Embeddings model.

### Retrieval and generation

4. Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.

5. Generate: A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data.

## Setup

- Jupyter Notebook

### Installation

```shell
$ pip install langchain
```

### LangSmith

- Many of the applications you build with LangChain will contain multiple steps with multiple invocation of LLM calls.

- As these applications get more and more complex, it becomes crucial to be able to inspect what eactly is going on inside your chain or agent.

- The best way to do this is with LangSmith.

- After you sign up at the link above, make sure to set your environment variables to start logging traces:

    ```shell
    export LANGCHAIN_TRACING_V2="true"
    export LANGCHAIN_API_KEY="..."
    ```

## Preview

- In this guide we'll build a QA app over as website.

- The specific website we will use is the LLM Powered Autonomous Agents blog post by Lilian Weng, which allows us to ask questions about the contents of the post.

- We can create a simple indexing pipeline and RAG chain to do this in ~20 lines of code.

```shell
$ pip install --upgrade --quiet  langchain-google-genai pillow
```

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
```

```python
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
        web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
                )
            ),
        )
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=GoogleGenerativeAIEmbeddings(
                                        model="models/embedding-001"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )

rag_chain.invoke("What is Task Decomposition?")
```

```python
# cleanup
vectorstore.delete_collection()
```

- Check out the [LangSmith trace](https://smith.langchain.com/public/1c6ca97e-445b-4d00-84b4-c7befcbc59fe/r).

## Detailed wolkthrough

### 1. Indexing: Load

- We can use DocumentLoaders for this, which are objects that load in data from a source and return a list of Documents.

- A `Document` is an object with some `page_cotent` (str) and `metadata` (dict).

- In this case we'll use the WebBaseLoader, which uses `urllib` to load HTML from web URLs and `BeautifulSoup` to parse it to text.

- We can customize the HTML -> text parsing by passing in parameters to the `BeautifulSoup` parser via `bs__kwargs` (see BeautifulSoup docs).

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

len(docs[0].page_content)
```

```python
print(docs[0].page_content[:500])
```

## 2. Indexing: Split

- Even for those models that could fit the full post in their context window, models can struggle to find information in very long inputs.

- To handle this we'll split the `Document` into chunks for embedding and vector storage.

- This should help us retrieve only the most relevant bits of the blog post at run time.

- In this case we'll split out documents into chunks of 1000 characters with 200 characters of overlap between chunks.

- The overlap helps mitigate the possibility of separating a statement from important context related to it.

- We use the RecursiveCharacterTextSplitter, which will recursively split the document using common separators like new lines until each chunk is the appropriate size.

- This is the recommended text splitter for generic text use cases.

- We set `add_start_index=True` so that the character index at which each split Document starts within the initial Document is preserved as metadata attribute "start_index".

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)
```

```python
len(all_splits[0].page_content)
```

```python
all_splits[10].metadata
```

## 3. Indexing: Store

- Now we need to index out 66 text chunks so that we can search over them at runtime.

- The most common way to do this is to embed the contents of each document split and insert these embeddings into a vector database (or vector store).

- When we want to search over out splits, we take a text search query, embed it, and perform some sort of "similarity" search to identify the stored splits with the most similar embeddings to out query embedding.

- The simplest similarity measure is cosine similarity -- we measure the cosine of the angle between each pair of embeddings (which are high dimensional vectors).

- We can embed and store all of our document splits in a single command using the Chroma vector store and GoogleGenerativeAIEmbeddings model.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=GoogleGenerativeAIEmbeddings(
                                        model="models/embedding-001"))
```

## 4. Retrieval and Generation

- First we need to define out logic for searching over documents.

- LangChain defines a Retriever interface which wraps an index that cn return relevant `Documents` given a string query.

- The most common type of `Retriever` is the VectorStoreRetriever, which uses the similarity search capabilities of a vector store to facilitate retrieval.

- Any `VectorStore` can easily be turned into a `Retriever` with `VectorStore.as_retriever()`:

```python
retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 6})

retrieved_docs = retriever.invoke(
        "What are the approaches to Task Decomposition?")

len(retrieved_docs)
```

## 5. Retrieval and Generation: Generate

- Let's put it all together into a chain that takes a question, retrieves relevant documents, constructs a prompt passes that to a model, and parses the output.

```python
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
        {"context": "filler context", "question": "filler question"}
        ).to_messages()

example_messages
```

```python
print(example_messages[0].content)
```

- We'll use the LCEL Runnable protocol to define the chain, allowing us to

    - pipe together components and functions in a transparent way

    - automatically trace our chain in LangSmith

    - get streaming, async, and batched calling out of the box.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
```

- Let's dissect the LCEL to understand what's going on.

- First: each of these components are instances of Runnable.

- This means that they implement the same methods which makes them easier to connect together.

- They can be connected into a Runnable Sequence via the `|` operator.

- LangChain will automatically cast certain objects to runnables when met with the `|` operator.

- Here, `format_docs` is cast to a RunnableLambda, and the dict with `"context"` and `"question"` is cast to a RunnableParallel.

- Let's trace how the input question flows through the above runnables.

- The input to `prompt` is expected to be a dict with keys `"context"` and `"question"`.

- So the first element of this chain builds runnables that will calculate both of these from the input question:

    - `retriever | format_docs` passes the question through the retriever, generating Document objects, and then to `format_docs` to generate strings;

    - `RunnablePassthrough()` passes through the input question unchanged.

- The last steps of the chain are `llm`, which runs the inference, and `StrOutputParser()`, which just plucks the string content out of the LLM's output message.

- You can analyze the individual steps of this chain via its [LangSmith trace](https://smith.langchain.com/public/1799e8db-8a6d-4eb2-84d5-46e8d7d5a99b/r).
