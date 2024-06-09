# Google AI chat models

```ipython
%pip install --upgrade --quiet  langchain-google-genai pillow
```

```ipython
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
```

## Example usage

```ipython
from langchain_google_genai import ChatGoogleGenerativeAI
```

```ipython
llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("Write a ballad about LangChain")
print(result.content)
```
