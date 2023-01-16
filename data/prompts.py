PROMPT_GENERAL = (
    lambda info, query: f"""Using the following text, answer the following question. If the answer is not contained within the text, say that you are not able to answer because relevant information is not present.

Text:
\"\"\"
{info}
\"\"\"

Question: {query}

Answer:"""
)

PROMPT_NO_INFO = (
    lambda query: f"""You are helpful AI assistant created by Alex and Sergei which goal is to answer questions based on given information. You was asked question, answer it in detail.

Question: {query}

Answer:"""
)
