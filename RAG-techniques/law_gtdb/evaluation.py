"""
RAG Evaluation Script

Đoạn script này đánh giá hiệu suất của hệ thống **Retrieval-Augmented Generation (RAG)** 
bằng cách sử dụng các chỉ số khác nhau từ thư viện deepeval.


Dependencies:
- deepeval
- langchain_openai
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import json
from typing import List, Tuple, Dict, Any

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import sys
import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from helper import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:
    """
    Tạo một danh sách các đối tượng LLMTestCase để đánh giá.

    Args:
        questions (List[str]): Danh sách các câu hỏi đầu vào.
        gt_answers (List[str]): Danh sách các câu trả lời thực tế (ground truth).
        generated_answers (List[str]): Danh sách các câu trả lời được tạo ra.
        retrieved_documents (List[str]): Danh sách các tài liệu được truy xuất.

    Returns:
        List[LLMTestCase]: Danh sách các đối tượng LLMTestCase.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model="gpt-3.5-turbo",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model="gpt-3.5-turbo",
    include_reason=True
)

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Đánh giá một hệ thống RAG bằng cách sử dụng các câu hỏi kiểm tra và các chỉ số được xác định trước.
    
    Args:
        retriever: Thành phần retriever cần đánh giá
        num_questions: Số lượng câu hỏi kiểm tra cần tạo
    
    Returns:
        Dict chứa các chỉ số đánh giá
        
        1.  **Mức độ liên quan (Relevance)**: Thông tin được truy xuất liên quan đến câu hỏi đến mức nào?
        2.  **Tính đầy đủ (Completeness)**: Ngữ cảnh có chứa tất cả thông tin cần thiết không?
        3.  **Tính cô đọng (Conciseness)**: Ngữ cảnh được truy xuất có tập trung và không có thông tin không liên quan không?

    """
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.
    
    Question: {question}
    Retrieved Context: {context}
    
    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance: How relevant is the retrieved information to the question?
    2. Completeness: Does the context contain all necessary information?
    3. Conciseness: Is the retrieved context focused and free of irrelevant information?
    
    Provide ratings in JSON format:
    """)
    
    # Create evaluation chain
    eval_chain = (
        eval_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")
    
    # Evaluate each question
    results = []
    for question in questions:
        # Get retrieval results
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Evaluate results
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)
    
    return {
        "questions": questions,
        "results": results,
        "average_scores": calculate_average_scores(results)
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average scores across all evaluation results."""
    # Implementation depends on the exact format of your results
    pass

if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass