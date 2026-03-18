import json
import logging
import re
from typing import Any

from src.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)

class QuestionAnalyzer:
    """
    Analyzes questions to determine complexity, decomposes complex questions
    into sub-questions, and synthesizes final reports from multiple results.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    async def analyze(self, question: str) -> dict[str, Any]:
        """
        Analyze a question to determine if it's SIMPLE or COMPLEX.
        If COMPLEX, also provides a list of sub-questions.
        """
        prompt = f"""
        Analyze the following question and determine its complexity for a RAG system that queries a database.

        Complexity Criteria:
        - SIMPLE: A question that can be answered by a single SQL query or a direct data lookup (e.g., "How many users are there?", "What is the price of product X?").
        - COMPLEX: A question that requires multiple steps, comparisons across different data sets, or aggregation of disparate information (e.g., "Compare sales in 2023 vs 2024 by region", "What are the top 3 products and who are their main buyers?").

        If the question is COMPLEX, decompose it into exactly the necessary sub-questions to answer the original question.

        Return ONLY a JSON object with this structure:
        {{
            "complexity": "SIMPLE" | "COMPLEX",
            "sub_questions": ["sub-question 1", "sub-question 2", ...] (only if COMPLEX)
        }}

        Question: {question}
        """

        try:
            # vanna llm services usually have a generate method
            response = await self.llm.generate(prompt)

            if not response or not response.text:
                logger.warning("Empty response from LLM for question analysis.")
                return {"complexity": "SIMPLE", "sub_questions": []}

            response_text = response.text
            metadata = response.metadata

            # 1. Clean the response text from common LLM artifacts
            clean_text = (response_text.replace("\\'", "'")
                                       .replace("\\n", "\n")
                                       .replace("\\r", "\r")
                                       .replace("\\t", "\t"))

            # 2. Robust extraction: find the first { and last }
            start_index = clean_text.find('{')
            end_index = clean_text.rfind('}')

            if start_index != -1 and end_index != -1 and end_index >= start_index:
                extracted_json = clean_text[start_index:end_index+1]
            else:
                extracted_json = clean_text.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(extracted_json)
            except json.JSONDecodeError:
                try:
                    repaired_json = re.sub(r',\s*([\]\}])', r'\1', extracted_json)
                    result = json.loads(repaired_json)
                except json.JSONDecodeError as e:
                    hex_dump = ' '.join(hex(ord(c)) for c in extracted_json[:100])
                    logger.error("JSON decoding failed for question: %s. Error: %s. Extracted text: %r. Hex: %s",
                                 question, e, extracted_json, hex_dump)
                    return {"complexity": "SIMPLE", "sub_questions": [], "metadata": metadata}

            # Capture metadata
            if metadata:
                result["metadata"] = metadata

            logger.info("Analyzed question: %s -> %s", question, result.get("complexity"))
            return result

        except Exception as e:
            logger.error("Analysis failed unexpectedly: %s", e)
            # Fallback to SIMPLE on error
            return {"complexity": "SIMPLE", "sub_questions": []}


    async def synthesize(self, original_question: str, results: list[dict[str, Any]]) -> str:
        """
        Synthesize a final report from the original question and the results of sub-questions.
        Each result in `results` should contain the 'question' and its 'response'.
        """
        formatted_results = ""
        for i, res in enumerate(results):
            formatted_results += f"Sub-question {i+1}: {res['question']}\nResponse: {res['response']}\n\n"

        prompt = f"""
        You are a data analyst. Synthesize a final, professional, and clear response to the original question based on the results of several sub-questions.

        Original Question: {original_question}

        Intermediate Results:
        {formatted_results}

        Provide a comprehensive and well-structured answer that directly addresses the original question. Use markdown for better presentation.
        """

        try:
            response = await self.llm.generate(prompt)
            return response.text
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            return "Error: Could not synthesize final report."
