from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.providers.base import LLMResponse
from src.services.question_analyzer import QuestionAnalyzer


@pytest.fixture
def analyzer():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return QuestionAnalyzer(llm)

@pytest.mark.asyncio
async def test_robustness_french_escaped_quotes(analyzer):
    # The actual case that failed for the user
    raw_text = r'{"complexity": "COMPLEX", "sub_questions": ["combien de lignes contient chacune d\'elles\u202f?"]}'
    analyzer.llm.generate.return_value = LLMResponse(text=raw_text)

    result = await analyzer.analyze("French test")

    assert result["complexity"] == "COMPLEX"
    assert "chacune d'elles" in result["sub_questions"][0]

@pytest.mark.asyncio
async def test_robustness_nested_with_text(analyzer):
    # Text surrounding a nested object
    raw_text = 'Here is the result: {"complexity": "COMPLEX", "meta": {"step": 1}} and some footer.'
    analyzer.llm.generate.return_value = LLMResponse(text=raw_text)

    result = await analyzer.analyze("Nested text test")

    assert result["complexity"] == "COMPLEX"
    assert result["meta"]["step"] == 1

@pytest.mark.asyncio
async def test_robustness_trailing_commas_complex(analyzer):
    # Multiple trailing commas in arrays and objects
    raw_text = '{"complexity": "COMPLEX", "sub_questions": ["q1", "q2",], "other": {"a": 1,},}'
    analyzer.llm.generate.return_value = LLMResponse(text=raw_text)

    result = await analyzer.analyze("Trailing comma complex test")

    assert result["complexity"] == "COMPLEX"
    assert len(result["sub_questions"]) == 2
    assert result["other"]["a"] == 1

@pytest.mark.asyncio
async def test_robustness_multiple_brackets(analyzer):
    # Handles text containing brackets before the actual JSON
    raw_text = 'Notes [1, 2, 3]: {"complexity": "SIMPLE"}'
    analyzer.llm.generate.return_value = LLMResponse(text=raw_text)

    result = await analyzer.analyze("Multiple brackets test")

    assert result["complexity"] == "SIMPLE"

@pytest.mark.asyncio
async def test_robustness_literal_escapes_yo(analyzer):
    # This simulates the "yo" case where the LLM might return literal \n
    raw_text = r'{\n    "complexity": "SIMPLE"\n}'
    analyzer.llm.generate.return_value = LLMResponse(text=raw_text)

    result = await analyzer.analyze("yo")

    assert result["complexity"] == "SIMPLE"

@pytest.mark.asyncio
async def test_robustness_escaped_newlines(analyzer):
    # LLM might return literal \n in the string
    raw_text = '{\n  "complexity": "SIMPLE"\n}'
    analyzer.llm.generate.return_value = LLMResponse(text=raw_text)

    result = await analyzer.analyze("Newline test")

    assert result["complexity"] == "SIMPLE"
