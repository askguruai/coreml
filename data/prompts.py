EMBEDDING_INSTRUCTION = (
    "Represent the knowledge base search query for retrieving relevant information."
)

PROMPT_GENERAL = (
    lambda info, query: f"""Answer following question in detail based on given text. If no relevant information present, say \" no relevant info present, can't answer\".

{info}

{query}"""
)

PROMPT_NO_INFO = (
    lambda query: f"""You are helpful AI assistant created by Alex and Sergei which goal is to answer questions based on given information. You was asked question, answer it in detail.

Question: {query}

Answer:"""
)
