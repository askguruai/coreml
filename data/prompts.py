from collections import defaultdict

COMPLETIONS_PROMPT_OPENAI_SYSTEM = {
    "general": "You are helpful assistant which follows given instructions",
    "support": "You are a customer support agent who is seeking to provide a complete, simple and helpful answer to a customer in a friendly manner. If text does not provide relevant information, say that you are not able to help because knowledge base does not contain necessary information.",
}

COMPLETIONS_PROMPT_OPENAI = (
    lambda info, query: f"""Using the following information from a knowledge base, answer the customer's question. If text provides links then copy them to output.

Information from knowledge base:
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

EMBEDDING_INSTRUCTION = "Represent the text snippet for similarity search."

COMPLETIONS_PROMPT_CUSTOM = (
    lambda info, query: f"""Answer following question in detail based on given text.

Text:
\"\"\"
{info}
\"\"\"

Question: {query}"""
)
