from synaptix.router import classify


def test_simple_query_is_local():

    route = classify(
        "Explain recursion"
    )

    assert route.needs_web is False


def test_current_query_uses_web():

    route = classify(
        "What are the latest AI models in 2026?"
    )

    assert route.needs_web is True


def test_research_is_deep():

    route = classify(
        "Research the latest local AI model landscape"
    )

    assert route.deep_research is True