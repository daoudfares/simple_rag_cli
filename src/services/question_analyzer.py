import json
import logging
import re
from typing import Any

from src.llm.providers.base import BaseLLM

logger = logging.getLogger(__name__)

# Maximum number of sub-questions allowed for a COMPLEX question.
# Prevents runaway decomposition and excessive LLM/DB calls.
MAX_SUB_QUESTIONS = 5


class QuestionAnalyzer:
    """
    Analyzes questions to determine complexity, decomposes complex questions
    into sub-questions, and synthesizes final reports from multiple results.

    Uses two (optionally different) LLMs:
    - ``llm``: lightweight/fast model for routing and classification.
    - ``synthesis_llm``: heavier/smarter model for producing the final
      synthesis report.  Falls back to ``llm`` when not provided.
    """

    def __init__(self, llm: BaseLLM, synthesis_llm: BaseLLM | None = None):
        self.llm = llm  # Used for routing/analysis (lighter model)
        self.synthesis_llm = synthesis_llm or llm  # Used for final synthesis (heavier model for quality)

    async def analyze(self, question: str) -> dict[str, Any]:
        """
        Analyze a question to determine if it's SIMPLE or COMPLEX.
        If COMPLEX, also provides a list of sub-questions (capped at MAX_SUB_QUESTIONS).
        """
        prompt = f"""
        Analyze the following question and determine its complexity for a RAG system that queries a database.

        Complexity Criteria:
        - SIMPLE: A question that can be answered by a single SQL query or a direct data lookup (e.g., "How many users are there?", "What is the price of product X?").
        - COMPLEX: A question that requires multiple steps, comparisons across different data sets, or aggregation of disparate information (e.g., "Compare sales in 2023 vs 2024 by region", "What are the top 3 products and who are their main buyers?").

        If the question is COMPLEX, decompose it into exactly the necessary sub-questions to answer the original question.
        IMPORTANT: Generate at most {MAX_SUB_QUESTIONS} sub-questions. If the question requires more, consolidate related aspects into fewer sub-questions.

        Return ONLY a JSON object with this structure:
        {{
            "complexity": "SIMPLE" | "COMPLEX",
            "sub_questions": ["sub-question 1", "sub-question 2", ...] (only if COMPLEX, maximum {MAX_SUB_QUESTIONS})
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

            # Enforce sub-question limit
            if "sub_questions" in result and len(result["sub_questions"]) > MAX_SUB_QUESTIONS:
                logger.warning(
                    "LLM generated %d sub-questions, truncating to %d",
                    len(result["sub_questions"]),
                    MAX_SUB_QUESTIONS,
                )
                result["sub_questions"] = result["sub_questions"][:MAX_SUB_QUESTIONS]

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

        Each result in ``results`` should contain:
        - ``question``: the sub-question text
        - ``response``: textual response
        - ``dataframes`` (optional): list of dicts with ``title``, ``columns``, ``rows``
        - ``error`` (optional): error message if the sub-question failed

        Uses ``self.synthesis_llm`` (the heavier model) for higher-quality output.
        """
        formatted_results = ""
        for i, res in enumerate(results):
            section = f"Sub-question {i+1}: {res['question']}\n"

            if res.get("error"):
                section += f"Status: FAILED — {res['error']}\n"
            else:
                section += f"Response: {res['response']}\n"

                # Include structured DataFrame data for richer synthesis
                if res.get("dataframes"):
                    for df in res["dataframes"]:
                        section += f"\nData ({df.get('title', 'Results')}):\n"
                        columns = df.get("columns", [])
                        section += f"  Columns: {', '.join(columns)}\n"
                        rows = df.get("rows", [])
                        # Show first 20 rows to keep the prompt manageable
                        for row in rows[:20]:
                            section += f"  {row}\n"
                        if len(rows) > 20:
                            section += f"  ... and {len(rows) - 20} more rows\n"

            section += "\n"
            formatted_results += section

        prompt = f"""
        You are a data analyst. Synthesize a final, professional, and clear response to the original question based on the results of several sub-questions.

        Original Question: {original_question}

        Intermediate Results:
        {formatted_results}

        Instructions:
        - Provide a comprehensive and well-structured answer that directly addresses the original question.
        - If some sub-questions failed, work with the available results and clearly mention the gaps.
        - Use markdown for better presentation (tables, headers, bold, etc.).
        - Include key numbers and comparisons when available from the data.
        """

        try:
            response = await self.synthesis_llm.generate(prompt)
            return response.text
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            return "Error: Could not synthesize final report."
