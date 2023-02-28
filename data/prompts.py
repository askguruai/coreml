COMPLETIONS_PROMPT_OPENAI = (
    lambda info, query: f"""Using the following text, answer the following question. If the answer is not contained within the text, say that you are not able to answer because relevant information is not present.

Text:
\"\"\"
{info}
\"\"\"

Question: {query}

Answer:"""
)

COMPLETIONS_PROMPT_OPENAI_NO_INFO = (
    lambda query: f"""You are helpful AI assistant created by Alex and Sergei which goal is to answer questions based on given information. You was asked question, answer it in detail.

Question: {query}

Answer:"""
)

# EMBEDDING_INSTRUCTION = (
#     "Represent the knowledge base search query for retrieving relevant information."
# )

EMBEDDING_INSTRUCTION = (
    "Represent the text snippet for similarity search."
)

COMPLETIONS_PROMPT_CUSTOM = (
    lambda info, query: f"""Answer following question in detail based on given text.

{info}

{query}"""
)
