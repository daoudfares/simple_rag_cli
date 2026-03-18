from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.providers.base import LLMResponse
from src.services.question_analyzer import QuestionAnalyzer


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm

@pytest.fixture
def analyzer(mock_llm):
    return QuestionAnalyzer(mock_llm)

@pytest.mark.asyncio
async def test_analyze_simple_question(analyzer, mock_llm):
    mock_llm.generate.return_value = LLMResponse(text='{"complexity": "SIMPLE"}')

    result = await analyzer.analyze("How many users are there?")

    assert result["complexity"] == "SIMPLE"
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_complex_question(analyzer, mock_llm):
    mock_llm.generate.return_value = LLMResponse(text='{"complexity": "COMPLEX", "sub_questions": ["q1", "q2"]}')

    result = await analyzer.analyze("Compare sales in 2023 and 2024.")

    assert result["complexity"] == "COMPLEX"
    assert len(result["sub_questions"]) == 2
    assert result["sub_questions"] == ["q1", "q2"]

@pytest.mark.asyncio
async def test_analyze_fallback_on_error(analyzer, mock_llm):
    mock_llm.generate.side_effect = Exception("API Error")

    result = await analyzer.analyze("Some question")

    assert result["complexity"] == "SIMPLE"

@pytest.mark.asyncio
async def test_analyze_escaped_quotes(analyzer, mock_llm):
    # Simulate the actual error reported by the user
    # Note: Using double backslash to represent a literal backslash in the string
    mock_llm.generate.return_value = LLMResponse(text=r'{"complexity": "COMPLEX", "sub_questions": ["Quels index ou contraintes d\'optimisation existent?"]}')

    result = await analyzer.analyze("fais moi une analyse complete de ma base de données")

    assert result["complexity"] == "COMPLEX"
    assert "optimisation" in result["sub_questions"][0]

@pytest.mark.asyncio
async def test_analyze_nested_object(analyzer, mock_llm):
    # Test nested object support
    mock_llm.generate.return_value = LLMResponse(text='Some text before {"complexity": "COMPLEX", "details": {"nested": true}} and text after')

    result = await analyzer.analyze("Nested test")

    assert result["complexity"] == "COMPLEX"
    assert result["details"]["nested"] is True

@pytest.mark.asyncio
async def test_analyze_trailing_comma(analyzer, mock_llm):
    # Test trailing comma recovery
    mock_llm.generate.return_value = LLMResponse(text='{"complexity": "SIMPLE",}')

    result = await analyzer.analyze("Trailing comma test")

    assert result["complexity"] == "SIMPLE"

@pytest.mark.asyncio
async def test_synthesize_report(analyzer, mock_llm):
    mock_llm.generate.return_value = LLMResponse(text="Final report content")
    results = [
        {"question": "q1", "response": "r1"},
        {"question": "q2", "response": "r2"}
    ]

    report = await analyzer.synthesize("Original question", results)

    assert report == "Final report content"
    mock_llm.generate.assert_called_once()
