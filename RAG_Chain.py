from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline

from Load_Model import my_pipeline

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = HuggingFaceEmbeddings(model_name='vinai/phobert-base-v2')
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db


db = create_db_from_files()

retriever = db.as_retriever()

template1 = """<|im_start|>system
Dựa vào ngữ cảnh sau để trả lời câu hỏi\n{context}\n . Ngữ cảnh trên thuộc Luật hôn nhân và gia đình Việt Nam. Nếu câu hỏi không liên quan đến ngữ cảnh này, hãy đưa ra thông báo.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt_qa = PromptTemplate(template=template1, input_variables=["context","question"])

llm_chain = LLMChain(prompt=prompt_qa,
                     llm=my_pipeline
                     )

rag_chain = RetrievalQA.from_chain_type(
        llm = my_pipeline,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt_qa},
    )

question = "Ở Việt Nam, nữ giới đủ bao nhiêu tuổi thì được kết hôn?"
response = rag_chain.invoke({"query": question})
print(response)