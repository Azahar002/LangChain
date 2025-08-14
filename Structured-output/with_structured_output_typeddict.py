from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatOpenAI(temperature=0)


class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

structured_model = model.with_structured_output(Review)
result = structured_model.invoke(
    """
    The hardware is great, bu thte sofware feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks Outdated compared to other brands. hoping for a software update to fix this.
    """
)

print(result)
# print(result['summary'])
# print(result['sentiment'])