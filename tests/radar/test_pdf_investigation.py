"""Question-driven investigation through real PydanticAI, without live models."""

import unittest

from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from tests.radar.pdf_support import empty_notes_model, make_pdf_document
from radar.agent.research_team import research_candidates
from radar.processing.evidence import attach_evidence
from radar.schema.papers import CollectedWork
from radar.provider.strata import StrataError


def sources(count=1, late="Overlap is required."):
    papers = [CollectedWork(openalex_id=f"https://openalex.org/W{i}", title=f"Paper {i}",
                            abstract="An identification result.") for i in range(count)]
    return papers, [make_pdf_document(p.openalex_id, ("Introduction.", late)) for p in papers]


def tool_returns(messages):
    return [part.content for message in messages for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "read_pdf_passage"]


def dossier(passage_id, **changes):
    payload = dict(
        paper_index=0, supporting_pages=[2], source_passage_ids=[passage_id],
        core_problem="When does the identification argument fail?",
        reported_contribution="The stated boundary requires overlap.",
        reasoning="Without overlap the proposed inference is not identified.",
        significance="Distinguishes an assumption from an empirical guarantee.",
        assumptions_limits=["Overlap is required."], connections=[],
        study_tasks=[dict(kind="counterexample", objective="Remove overlap.",
                         success_criterion="Construct indistinguishable observed distributions.",
                         missing_evidence="A counterexample is proposed, not executed.")],
        open_questions=["Which target populations satisfy overlap?"],
    )
    return payload | changes


class TestPDFInvestigation(unittest.TestCase):
    def test_a_read_co_emitted_with_final_answer_does_not_count_as_consulted_yet(self):
        papers, documents = sources()
        attempts = []

        def respond(messages, info):
            if not info.function_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
            attempts.append(1)
            if not tool_returns(messages):
                guessed = f"W0:{documents[0].sha256}:p2:0-{len(documents[0].pages[1].text)}"
                return ModelResponse(parts=[
                    ToolCallPart("read_pdf_passage", {"paper_index": 0, "page": 2}),
                    ToolCallPart(info.output_tools[0].name, {"learning_dossiers": [
                        dossier(guessed, reasoning="Guessed before receiving the source.")]})])
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                "learning_dossiers": [dossier(tool_returns(messages)[-1]["passage_id"],
                    reasoning="The returned source states overlap is required.")]} )])

        result = research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                     document_model=empty_notes_model())
        self.assertEqual(len(attempts), 2)
        self.assertEqual(result.draft.learning_dossiers[0].reasoning,
                         "The returned source states overlap is required.")

    def test_page_continuation_preserves_exact_text_and_distinct_passage_ids(self):
        from radar.config.documents import DOCUMENT_CHUNK_CHARS

        late = "α" * DOCUMENT_CHUNK_CHARS + "\n```quoted Markdown```\nThe boundary fails."
        papers, documents = sources(late=late)

        def respond(messages, info):
            if not info.function_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
            returns = tool_returns(messages)
            if not returns or returns[-1]["next_offset"] is not None:
                offset = returns[-1]["next_offset"] if returns else 0
                return ModelResponse(parts=[ToolCallPart("read_pdf_passage", {
                    "paper_index": 0, "page": 2, "offset": offset})])
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                "learning_dossiers": [dossier(returns[0]["passage_id"],
                    source_passage_ids=[p["passage_id"] for p in returns])]} )])

        result = research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                     document_model=empty_notes_model())
        self.assertEqual("".join(p.text for p in result.source_passages), late)
        self.assertEqual(result.source_passages[1].start, DOCUMENT_CHUNK_CHARS)
        self.assertIsNone(result.source_passages[1].next_offset)
        from radar.output.markdown import render_markdown

        report = attach_evidence(result.draft, papers, document_readings=list(result.document_readings),
                                 source_passages=list(result.source_passages))
        self.assertIn("````text\n" + result.source_passages[1].text + "\n````", render_markdown(report))

    def test_final_investigation_remains_inside_the_team_deadline(self):
        import asyncio

        papers, documents = sources()
        cancelled = []

        async def respond(messages, info):
            if not info.function_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

        with self.assertRaisesRegex(StrataError, "overall .* deadline"):
            research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                document_model=empty_notes_model(), analysis_timeout_s=0.1)
        self.assertEqual(cancelled, [True])

    def test_unread_cross_paper_and_mismatched_page_citations_fail_closed(self):
        for variant in ("unread", "other_paper", "wrong_page", "empty", "duplicate", "bool"):
            with self.subTest(variant=variant):
                papers, documents = sources(2)
                attempts = []

                def respond(messages, info):
                    if not info.function_tools:
                        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
                    returns = tool_returns(messages)
                    if not returns and variant != "unread":
                        return ModelResponse(parts=[ToolCallPart("read_pdf_passage", {
                            "paper_index": 1 if variant == "other_paper" else 0, "page": 2})])
                    # A correctly shaped but unread ID is still not evidence.
                    passage_id = returns[-1]["passage_id"] if returns else (
                        f"W0:{documents[0].sha256}:p2:0-20")
                    changes = {}
                    if variant == "wrong_page":
                        changes["supporting_pages"] = [1]
                    if variant in ("empty", "duplicate", "bool"):
                        changes["source_passage_ids"] = ({"empty": [], "duplicate": [passage_id] * 2,
                                                          "bool": [True]})[variant]
                    attempts.append(1)
                    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                        {"learning_dossiers": [dossier(passage_id, **changes)]})])

                with self.assertRaises(StrataError):
                    research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                        document_model=empty_notes_model())
                self.assertEqual(len(attempts), 2)

    def test_passage_addressing_rejects_booleans_strings_and_out_of_range_values(self):
        for changes in ({"paper_index": True}, {"page": True}, {"offset": True},
                        {"page": "2"}, {"paper_index": 1}, {"page": 3},
                        {"offset": -1}, {"offset": 100}):
            with self.subTest(changes=changes):
                papers, documents = sources()
                calls = []

                def respond(messages, info):
                    if not info.function_tools:
                        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
                    calls.append(1)
                    return ModelResponse(parts=[ToolCallPart("read_pdf_passage",
                        {"paper_index": 0, "page": 2, "offset": 0} | changes)])

                with self.assertRaises(StrataError):
                    research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                        document_model=empty_notes_model())
                self.assertEqual(len(calls), 2)

    def test_six_read_turns_and_final_repair_honor_eight_request_and_token_budgets(self):
        papers, documents = sources()
        final_calls, limits = [], []

        def respond(messages, info):
            limits.append(info.model_settings["max_tokens"])
            if not info.function_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
            final_calls.append(1)
            returns = tool_returns(messages)
            if len(returns) < 6:
                return ModelResponse(parts=[ToolCallPart("read_pdf_passage", {"paper_index": 0, "page": 2})])
            payload = dossier(returns[-1]["passage_id"], supporting_pages=[1] if len(final_calls) == 7 else [2])
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"learning_dossiers": [payload]})])

        result = research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                     document_model=empty_notes_model(), max_tokens=4096)
        self.assertEqual(len(final_calls), 8)
        self.assertEqual(limits, [4096] * 11)
        self.assertEqual(len(result.source_passages), 1)  # duplicate reads do not duplicate citations

    def test_seventh_passage_call_is_blocked(self):
        papers, documents = sources()
        calls = []

        def respond(messages, info):
            if not info.function_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
            calls.append(len(tool_returns(messages)))
            return ModelResponse(parts=[ToolCallPart("read_pdf_passage", {"paper_index": 0, "page": 2})])

        with self.assertRaises(StrataError):
            research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                document_model=empty_notes_model())
        self.assertEqual(calls, list(range(7)))

    def test_intact_tool_history_overflow_stops_before_another_request(self):
        from radar.config.documents import DOCUMENT_CHUNK_CHARS

        # One initial chunk: long notes must not fail an unrelated reduction.
        papers, documents = sources(late="a" * (DOCUMENT_CHUNK_CHARS - 100))
        calls = []

        def notes(messages, info):
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                "summary": "s" * 20000, "methods": "method", "results": "result",
                "limitations": "limits", "evidence": []})])

        def respond(messages, info):
            if not info.function_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])
            returns = tool_returns(messages)
            for passage in returns:
                self.assertEqual(passage["text"], "a" * (DOCUMENT_CHUNK_CHARS - 100))
            calls.append(len(returns))
            return ModelResponse(parts=[ToolCallPart("read_pdf_passage", {"paper_index": 0, "page": 2})])

        with self.assertRaisesRegex(StrataError, "history exceeds the prompt budget"):
            research_candidates(papers, documents=documents, model=FunctionModel(respond),
                                document_model=FunctionModel(notes))
        self.assertGreater(len(calls), 1)
        self.assertLess(len(calls), 7)

    def test_investigation_prompt_requires_source_revisiting_and_question_focus(self):
        from radar.prompts.catalog import pdf_investigation_prompt

        instructions = pdf_investigation_prompt().instructions
        for requirement in ("learning question", "read_pdf_passage", "source_passage_ids",
                            "untrusted", "counterevidence", "not semantic verification"):
            self.assertIn(requirement, instructions)

    def test_final_investigation_revisits_late_page_missing_from_reading_notes(self):
        paper = CollectedWork(openalex_id="https://openalex.org/W0", title="Identification",
                              abstract="A method for identification.")
        late = "The identification theorem requires overlap; without it the target is not identified."
        document = make_pdf_document(paper.openalex_id, ("Introduction only.", late))
        served = []

        def respond(messages, info):
            if not info.function_tools:
                # Three unchanged specialist stages.
                if "final synthesizer" in info.instructions:
                    self.fail("Final investigation has no passage tool")
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                         {"next_move": "Inspect the assumption."})])
            self.assertEqual([tool.name for tool in info.function_tools], ["read_pdf_passage"])
            returns = [part.content for message in messages for part in message.parts
                       if isinstance(part, ToolReturnPart) and part.tool_name == "read_pdf_passage"]
            if not returns:
                return ModelResponse(parts=[ToolCallPart("read_pdf_passage",
                    {"paper_index": 0, "page": 2, "offset": 0})])
            served.append(returns[-1])
            self.assertEqual(returns[-1]["text"], late)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                {"learning_dossiers": [dossier(returns[-1]["passage_id"])]})])

        result = research_candidates([paper], documents=[document], model=FunctionModel(respond),
                                     document_model=empty_notes_model())
        self.assertEqual(len(served), 1)
        self.assertNotIn(late, result.document_readings[0].notes.model_dump_json())
        self.assertEqual(result.document_readings[0].notes.evidence, [])
        self.assertEqual(result.source_passages[0].text, late)
        from radar.config.searches import search_policy

        for goal in search_policy().learning_goals:
            self.assertIn(goal.question, result.learning_question)
        report = attach_evidence(result.draft, [paper], document_readings=list(result.document_readings),
                                 source_passages=list(result.source_passages),
                                 learning_question=result.learning_question)
        self.assertEqual(report.learning_dossiers[0].source_pages, [2])
        self.assertEqual(report.learning_dossiers[0].source_passages[0].text, late)
        from radar.output.markdown import render_markdown

        rendered = render_markdown(report)
        self.assertIn(result.learning_question, rendered)
        self.assertIn(late, rendered)
        self.assertIn(served[0]["passage_id"], rendered)
