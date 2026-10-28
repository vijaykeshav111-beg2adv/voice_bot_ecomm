from pathlib import Path
import knowledge_base


def test_ingestion_indexes_markdown_and_pdf():
    report = knowledge_base.ingest_documents(force=True)
    assert report['ok'] is True
    assert report['documents_indexed'] >= 2
    assert report['chunks_indexed'] >= 2
    assert any(name.lower().endswith('.pdf') for name in report['sources'])


def test_policy_query_retrieves_grounded_store_passage():
    result = knowledge_base.search_knowledge('Does ShopMate process real payments or collect card details?')
    assert result['ok'] is True
    assert result['results']
    assert 'payment' in result['results'][0]['text'].lower() or 'card' in result['results'][0]['text'].lower()


def test_direct_policy_answer_skips_llm_when_retrieval_is_relevant():
    answer = knowledge_base.answer_from_knowledge('What is the return and refund policy?')
    assert answer is not None
    assert answer['grounded'] is True
    assert answer['source'].lower().endswith(('.md', '.pdf', '.txt'))
    assert '7 calendar days' in answer['answer'].lower() or '5–7 business days' in answer['answer'].lower() or '5-7 business days' in answer['answer'].lower()


def test_unrelated_question_does_not_match_rag_policy_fast_path():
    assert knowledge_base.answer_from_knowledge('What is Python?') is None


def test_shopmate_scope_redirect_does_not_call_llm(monkeypatch):
    import core
    monkeypatch.setattr(core, '_tts_clip', lambda text: {'audio': b'audio', 'first_byte': 0.0,
                                                         'seconds': 0.0, 'done_at': core.time.perf_counter()})
    def forbidden(*args, **kwargs):
        raise AssertionError('LLM must not be called for unrelated questions')
    monkeypatch.setattr(core, '_open_llm_stream', forbidden)
    rec = core.new_record('scope-test', source='text')
    events = list(core.run_turn(core.new_conversation(), rec, text='What is Python?'))
    assert rec['answer_route'] == 'scope_redirect'
    assert rec['llm_seconds'] == 0
    assert any('shopping assistant' in e.get('text', '').lower() for e in events)


def test_policy_fast_path_does_not_call_llm(monkeypatch):
    import core
    monkeypatch.setattr(core, '_tts_clip', lambda text: {'audio': b'audio', 'first_byte': 0.0,
                                                         'seconds': 0.0, 'done_at': core.time.perf_counter()})
    def forbidden(*args, **kwargs):
        raise AssertionError('LLM must not be called for a high-confidence document answer')
    monkeypatch.setattr(core, '_open_llm_stream', forbidden)
    rec = core.new_record('rag-test', source='text')
    events = list(core.run_turn(core.new_conversation(), rec, text='What is the return and refund policy?'))
    assert rec['answer_route'] == 'rag_direct'
    assert rec['llm_seconds'] == 0
    assert any('7 calendar days' in e.get('text', '').lower() or '5–7 business days' in e.get('text', '').lower() or '5-7 business days' in e.get('text', '').lower() for e in events)


def test_refund_timing_question_retrieves_sample_policy_without_llm():
    answer = knowledge_base.answer_from_knowledge('When will I get a refund?')
    assert answer is not None
    assert answer['grounded'] is True
    assert 'business days' in answer['answer'].lower()


def test_return_window_question_retrieves_sample_policy():
    answer = knowledge_base.answer_from_knowledge('How long do I have to return an item?')
    assert answer is not None
    assert '7 calendar days' in answer['answer'].lower()
