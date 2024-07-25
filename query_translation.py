from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from operator import itemgetter
from langchain.load import dumps, loads
from tempfile import NamedTemporaryFile
from langchain import hub
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

#Langsmith Tracing
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

 # LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


#Session States
def initialize_session_state():
    #dataframes
    if 'docs' not in st.session_state:
        st.session_state.docs = ''
    #sesion state for example questions
    if "db" not in st.session_state:
        st.session_state.db = ''

#Function to process text and create vector db
def get_vectorstore(pdf_docs):
    #Handle Streamlit uploaded File
    with NamedTemporaryFile(delete=False) as temp:
                        # Write the user's uploaded file to the temporary file.
                        with open(temp.name, "wb") as temp_file:
                            temp_file.write(pdf_docs.read())
    loader = PyPDFLoader(temp.name)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    # Index
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    return docs,retriever

def main():
    st.set_page_config("RAG:Query Translation 📁")
    st.header("Query Translation in RAG Pipelines")
    
    #Uploading the PDF Files:
    with st.sidebar:
        st.title("Configuration:🛠")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process File Button", accept_multiple_files=True)

        #Processing the uploaded pdf
        if st.button("Process File"):
            with st.spinner("Processing..."):
                st.session_state.docs,st.session_state.db = get_vectorstore(pdf_docs[0])
                st.success("Done")
        
        #Selecting the Multiselect Button
        option = st.selectbox("Select Query Translation Technique",
                              ("None","Multi-Query","RAG Fusion","Decomposition"),
                              index=0,
                              placeholder="Select query translation technique")

    user_question = st.text_input("Ask your question here")
    if user_question:
        with st.spinner("Generating Response..."):
            if option=="None":
                naive_query(st.session_state.db,user_question)
            elif option=="Multi-Query":
                multi_query(st.session_state.db,user_question)
            elif option=="RAG Fusion":
                rag_fusion(st.session_state.db,user_question)
            elif option=="Decomposition":
                query_decomposition(st.session_state.db,user_question)

  
#Function for Naive Query -- No Query Translation
def naive_query(retriever,user_question):
    # Prompt
    prompt = ChatPromptTemplate.from_template(

    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
    )

    #Creating the document chain
    document_chain = create_stuff_documents_chain(llm,prompt)
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    docs = retriever.get_relevant_documents(user_question)

    # Question
    response = retrieval_chain.invoke({"input":user_question})
    st.write(response['answer'])
    #st.write(response['context'])

    st.header("Document Sources")
    st.write(f"Number of Retrieved Documents: {len(docs)}")
    #Using Streamlit Expander
    with st.expander("Document Similarity Search"):
        #Find relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------------")

#Multi Query Technique
def multi_query(retriever,user_question):
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)


    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":user_question})
    
    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
    
    )

    response = final_rag_chain.invoke({"question":user_question})
    st.write(response.content)
    #st.write(len(docs))
    #st.write(docs)

    st.header("Document Sources")
    st.write(f"Number of Retrieved Documents: {len(docs)}")
    #Using Streamlit Expander
    with st.expander("Document Similarity Search"):
        #Find relevant chunks
        for i, doc in enumerate(docs):
            st.write(doc.page_content)
            st.write("---------------------------")

#RAG Fusion:
def rag_fusion(retriever,user_question):
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({"question": user_question})
   
    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        
    )

    response = final_rag_chain.invoke({"question":user_question})
    st.write(response.content)
 
    

    #Using Streamlit Expander
    st.header("Document Sources")
    st.write(f"Number of Retrieved Documents: {len(docs)}")
    with st.expander("Document Similarity Search"):
        #Find relevant chunks
        docs_1 = []
        for doc in docs:
            docs_1.append(doc[0])
        for i, doc in enumerate(docs_1):
            st.write(doc.page_content)
            st.write("---------------------------")

#Query Decomposition
def query_decomposition(retriever,user_question):
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    questions = generate_queries_decomposition.invoke({"question":user_question})


    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        
        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()



    q_a_pairs = ""
    docs_1 = []
    for q in questions:
        
        rag_chain = (
        {"context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())
        docs = retriever.invoke(q)
        docs_1.extend(docs)
        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    
    st.write("Generated Questions From Decomposition")
    st.write(questions)
    st.write("Generated Answer based on retrieved documents")
    st.write(answer)

    #Fixing Duplicates
    #docs_1 = list(set(docs_1))

    #Using Streamlit Expander
    st.header("Document Sources")
    st.write(f"Number of Retrieved Documents: {len(docs_1)}")
    with st.expander("Document Similarity Search"):
        #Find relevant chunks
        for i, doc in enumerate(docs_1):
            st.write(doc.page_content)
            st.write("---------------------------")

if __name__ == "__main__":
    main()