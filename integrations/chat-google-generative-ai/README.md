# Google AI chat models

```python
%pip install --upgrade --quiet  langchain-google-genai pillow
```

```python
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
```

## Example usage

```python
from langchain_google_genai import ChatGoogleGenerativeAI
```

```python
llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("Write a ballad about LangChain")
print(result.content)
```

- Gemini doesn't support `SystemMessage` at the moment, but it can be added to the first human message in the row.

- If you want such behavior, just set the `convert_system_message_to_human` to True:

```python
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
model(
    [
        SystemMessage(content="Answer only yes or no."),
        HumanMessage(content="Is apple a fruit?"),
    ]
)
```
