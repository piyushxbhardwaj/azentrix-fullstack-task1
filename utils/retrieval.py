from langchain_google_genai import ChatGoogleGenerativeAI
import os

def generate_answer(question, vector_store):
    """
    Retrieves context and generates an answer for the given question.
    """
    # Retrieve top k similar documents
    docs = vector_store.similarity_search(question, k=4)
    
    # Combine the document content to create context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Prompt template to prevent hallucination
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not contained in the context, you MUST return exactly this string:
    "This information is not available in the document"
    Do NOT guess or try to make up an answer.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    
    # Format the prompt
    formatted_prompt = prompt_template.format(context=context, question=question)
    
    # Initialize the LLM
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Generate the response
    response = model.invoke(formatted_prompt)
    
    return response.content
