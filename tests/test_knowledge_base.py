import knowledge_base


def test_search_returns_relevant_local_faq():
    result = knowledge_base.search_knowledge('What reference do I need to cancel my appointment?')
    assert result['ok'] is True
    assert result['results']
    assert any('reference' in row['text'].lower() for row in result['results'])
    assert all(row['source'].endswith(('.md', '.txt', '.pdf')) for row in result['results'])


def test_search_empty_query_is_rejected():
    result = knowledge_base.search_knowledge('   ')
    assert result['ok'] is False
    assert result['results'] == []
