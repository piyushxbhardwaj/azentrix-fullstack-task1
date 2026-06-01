from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_conversational_chain():
    """
    Sets up the QA chain with a custom prompt to prevent hallucination.
    """
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

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def generate_answer(question, vector_store):
    """
    Retrieves context and generates an answer for the given question.
    """
    # Retrieve top k similar documents
    docs = vector_store.similarity_search(question, k=4)
    
    # Generate answer using the QA chain
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    
    return response["output_text"]
