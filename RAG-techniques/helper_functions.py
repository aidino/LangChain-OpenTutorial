from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import PromptTemplate
from openai import RateLimitError
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum

def replace_t_with_space(list_of_documents):
    """
    Thay thế tất cả các ký tự tab ('\t') bằng dấu cách trong nội dung trang của mỗi tài liệu

    Args:
        list_of_documents: Danh sách các đối tượng tài liệu, mỗi đối tượng có thuộc tính 'page_content'.

    Returns:
        Danh sách tài liệu đã được sửa đổi với các ký tự tab được thay thế bằng dấu cách.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def text_wrap(text, width=120):
    """
    Bao bọc văn bản đầu vào theo chiều rộng được chỉ định.

    Args:
        text (str): Văn bản đầu vào cần bao bọc.
        width (int): Chiều rộng để bao bọc văn bản.

    Returns:
        str: Văn bản đã được bao bọc.
    """
    return textwrap.fill(text, width=width)

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Mã hóa PDF document thành một vector store sử dụng embeddings của OpenAI.

    Args:
        path: Đường dẫn đến tệp PDF.
        chunk_size: Kích thước mong muốn của mỗi đoạn văn bản.
        chunk_overlap: Lượng chồng lấp giữa các đoạn liên tiếp.

    Returns:
        Một vector store FAISS chứa nội dung sách đã được mã hóa.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """
    Mã hóa một chuỗi thành một vector store sử dụng embeddings của OpenAI.

    Args:
        content (str): Nội dung văn bản cần được mã hóa.
        chunk_size (int): Kích thước của mỗi đoạn văn bản.
        chunk_overlap (int): Độ chồng lấp giữa các đoạn.

    Returns:
        FAISS: Một vector store chứa nội dung đã được mã hóa.

    Raises:
        ValueError: Nếu nội dung đầu vào không hợp lệ.
        RuntimeError: Nếu có lỗi trong quá trình mã hóa.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0

        # Generate embeddings and create the vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"An error occurred during the encoding process: {str(e)}")

    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Truy xuất ngữ cảnh liên quan và các URL duy nhất cho một câu hỏi nhất định bằng cách sử dụng retriever truy vấn các đoạn văn bản.

    Args:
        question: Câu hỏi mà ngữ cảnh và URL cần được truy xuất.

    Returns:
        Một tuple chứa:
        - Một chuỗi với nội dung được nối của các tài liệu liên quan.
        - Một danh sách các URL duy nhất từ metadata của các tài liệu liên quan.
    """

    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]

    return context


class QuestionAnswerFromContext(BaseModel):
    """
    Mô hình tạo ra câu trả lời cho một truy vấn dựa trên ngữ cảnh được cung cấp.
    
    Attributes:
        answer_based_on_content (str): Câu trả lời được tạo ra dựa trên ngữ cảnh.
    """
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain(llm):
    # Initialize the ChatOpenAI model with specific parameters
    question_answer_from_context_llm = llm

    # Define the prompt template for chain-of-thought reasoning
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # Create a chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_llm.with_structured_output(
        QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Trả lời một câu hỏi bằng cách sử dụng ngữ cảnh đã cho bằng cách gọi một chuỗi suy luận.

    Args:
        question: Câu hỏi cần được trả lời.
        context: Ngữ cảnh được sử dụng để trả lời câu hỏi.

    Returns:
        Một từ điển chứa câu trả lời, ngữ cảnh và câu hỏi.
    """
    input_data = {
        "question": question,
        "context": context
    }
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def show_context(context):
    """
    Hiển thị nội dung của danh sách ngữ cảnh được cung cấp.

    Args:
        context (list): Một danh sách các mục ngữ cảnh cần hiển thị.

    In ra từng mục ngữ cảnh trong danh sách với tiêu đề chỉ vị trí của nó.
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


def read_pdf_to_string(path):
    """
    Đọc tài liệu PDF từ đường dẫn được chỉ định và trả về nội dung của nó dưới dạng chuỗi.

    Args:
        path (str): Đường dẫn tệp đến tài liệu PDF.

    Returns:
        str: Nội dung văn bản được nối của tất cả các trang trong tài liệu PDF.

    Hàm này sử dụng thư viện 'fitz' (PyMuPDF) để mở tài liệu PDF, lặp qua từng trang,
    trích xuất nội dung văn bản từ mỗi trang và nối nó vào một chuỗi duy nhất.
    """
    # Open the PDF document located at the specified path
    doc = fitz.open(path)
    content = ""
    # Iterate over each page in the document
    for page_num in range(len(doc)):
        # Get the current page
        page = doc[page_num]
        # Extract the text content from the current page and append it to the content string
        content += page.get_text()
    return content


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """
    Thực hiện truy xuất BM25 và trả về k đoạn văn bản đã được làm sạch hàng đầu.

    Args:
    bm25 (BM25Okapi): Chỉ mục BM25 đã được tính toán trước.
    cleaned_texts (List[str]): Danh sách các đoạn văn bản đã được làm sạch tương ứng với chỉ mục BM25.
    query (str): Chuỗi truy vấn.
    k (int): Số lượng đoạn văn bản cần truy xuất.

    Returns:
    List[str]: k đoạn văn bản đã được làm sạch hàng đầu dựa trên điểm số BM25.
    """
    # Tokenize the query
    query_tokens = query.split()

    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top k scores
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Retrieve the top k cleaned text chunks
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts


async def exponential_backoff(attempt):
    """
    Triển khai backoff lũy thừa với jitter.
    
    Args:
        attempt: Số lần thử lại hiện tại.
        
    Chờ một khoảng thời gian trước khi thử lại thao tác.
    Thời gian chờ được tính là (2^attempt) + một phần ngẫu nhiên của giây.
    """
    # Calculate the wait time with exponential backoff and jitter
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")

    # Asynchronously sleep for the calculated wait time
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Thử lại một coroutine bằng cách sử dụng backoff lũy thừa khi gặp lỗi RateLimitError.
    
    Args:
        coroutine: Coroutine cần được thực thi.
        max_retries: Số lần thử lại tối đa.
        
    Returns:
        Kết quả của coroutine nếu thành công.
        
    Raises:
        Ngoại lệ gặp phải cuối cùng nếu tất cả các lần thử lại đều thất bại.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to execute the coroutine
            return await coroutine
        except RateLimitError as e:
            # If the last attempt also fails, raise the exception
            if attempt == max_retries - 1:
                raise e

            # Wait for an exponential backoff period before retrying
            await exponential_backoff(attempt)

    # If max retries are reached without success, raise an exception
    raise Exception("Max retries reached")


# Enum class representing different embedding providers
class EmbeddingProvider(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"

# Enum class representing different model providers
class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"


def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Trả về một nhà cung cấp embedding dựa trên nhà cung cấp được chỉ định và ID mô hình.

    Args:
        provider (EmbeddingProvider): Nhà cung cấp embedding để sử dụng.
        model_id (str): Tùy chọn - ID mô hình embedding cụ thể để sử dụng.

    Returns:
        Một instance nhà cung cấp embedding LangChain.

    Raises:
        ValueError: Nếu nhà cung cấp được chỉ định không được hỗ trợ.
    """
    if provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")