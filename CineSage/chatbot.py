from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = "mistral-small-2506")
prompt = ChatPromptTemplate.from_messages([
        (
            "system",
        """
        You are an expert movie information extractor Assistant.

        Analyze the given movie paragraph and extract all important information that would be useful for a movie database or recommendation system.

        Given a movie paragraph, extract:
        - Movie Name
        - IMDB rating
        - Genre
        - Director
        - Cast
        - Main Characters
        - Plot Summary
        - Themes
        - Keywords
        - Notable Features
        - Quick Summary (2-3 lines)

        If any information is not available, mention 'Not Available'.
        """
    ),
    ("human",
     """
    Exract information from this paragraph:
    {paragraph}
    """)
])

para = input("Give your paragraph : ")

final_prompt = prompt.invoke(
    {"paragraph" : para}
)

response = model.invoke(final_prompt)
print(response.content)