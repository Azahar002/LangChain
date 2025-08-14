from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(temperature=0)


class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key theems discussed in the review in a list")
    summary: str = Field(description = "A brief summary of the review")
    sentiment: Literal["pros", "neg"] = Field(desciption = "Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description = "Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description = "Write down all the cons inside a list")
    name: Optional[list[str]] = Field(default=None, description = "Write the name of the reviewer")


structured_model = model.with_structured_output(Review)
result = structured_model.invoke(
    """
    The hardware is great, bu thte sofware feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks Outdated compared to other brands. hoping for a software update to fix this.
    """
)

print(result)
# print(result['summary'])
# print(result['sentiment'])