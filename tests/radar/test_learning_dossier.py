"""Learning dossier CORE slice: typed contracts, full-abstract prompts,
grounded evidence, rendering, and storage guards. No network or real models."""

from __future__ import annotations

import tempfile
import unittest

from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from radar.schema.opportunities import OpportunityDraft, RadarDraft, RadarReport
from radar.schema.papers import CollectedWork


def _work(i: int = 0, abstract: str = "Learning mechanisms and human incentives.") -> CollectedWork:
    return CollectedWork(
        openalex_id=f"https://openalex.org/W{i}",
        title=f"Paper {i}",
        abstract=abstract,
        primary_url=f"https://example.org/paper-{i}",
    )


def _dossier_dict(index: int = 0, **overrides):
    base = {
        "paper_index": index,
        "core_problem": "What problem does the paper address?",
        "reported_contribution": "What the abstract reports as its contribution.",
        "reasoning": "Key reasoning or mechanism sketched in the abstract.",
        "significance": "Why this matters for advanced study (interpretation).",
        "assumptions_limits": ["Assumes the abstract summary is faithful."],
        "connections": ["Hypothesis: transfer the mechanism to a new evaluation."],
        "study_tasks": [{
            "kind": "derivation",
            "objective": "Re-derive the central claim from stated assumptions.",
            "success_criterion": "Derivation matches the reported claim or exposes a gap.",
            "missing_evidence": "Full text needed to check omitted steps.",
        }],
        "open_questions": ["What detail is missing from the abstract?"],
    }
    base.update(overrides)
    return base


class TestLearningSchema(unittest.TestCase):
    def test_new_contracts_preserve_shape_and_forbid_extra(self):
        from radar.schema.learning import LearningDossierDraft, StudyTask

        task = StudyTask(
            kind="derivation", objective="o" * 300,
            success_criterion="s" * 300, missing_evidence="m" * 300)
        self.assertEqual(task.kind, "derivation")
        with self.assertRaises(ValidationError):
            StudyTask(kind="proof", objective="o", success_criterion="s", missing_evidence="m")
        with self.assertRaises(ValidationError):
            StudyTask(kind="derivation", objective="o", success_criterion="s",
                      missing_evidence="m", evidence_level="abstract")
        good = LearningDossierDraft.model_validate(_dossier_dict())
        self.assertEqual(good.paper_index, 0)
        with self.assertRaises(ValidationError):
            LearningDossierDraft.model_validate(_dossier_dict(paper_index=True))
        with self.assertRaises(ValidationError):
            LearningDossierDraft.model_validate(_dossier_dict(core_problem=""))
        with self.assertRaises(ValidationError):
            LearningDossierDraft.model_validate(
                _dossier_dict(study_tasks=[_dossier_dict()["study_tasks"][0]] * 3))
        with self.assertRaises(ValidationError):
            LearningDossierDraft.model_validate(_dossier_dict(open_questions=["q"] * 4))
        with self.assertRaises(ValidationError):
            LearningDossierDraft.model_validate(_dossier_dict(evidence_level="abstract"))
        with self.assertRaises(ValidationError):
            LearningDossierDraft.model_validate(_dossier_dict(connections=["not labeled"]))

    def test_connections_can_be_empty_without_manufacturing_a_transfer(self):
        from radar.schema.learning import LearningDossierDraft

        draft = LearningDossierDraft.model_validate(_dossier_dict(connections=[]))
        self.assertEqual(draft.connections, [])

    def test_required_study_content_cannot_be_whitespace_only(self):
        from radar.schema.learning import LearningDossierDraft, StudyTask

        for field in ("core_problem", "reported_contribution", "reasoning", "significance"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                LearningDossierDraft.model_validate(_dossier_dict(**{field: " \n "}))
        for field in ("objective", "success_criterion", "missing_evidence"):
            payload = _dossier_dict()["study_tasks"][0] | {field: " \n "}
            with self.subTest(field=field), self.assertRaises(ValidationError):
                StudyTask.model_validate(payload)

    def test_resolved_source_must_match_primary_index(self):
        from radar.schema.opportunities import ResolvedLearningDossier

        with self.assertRaises(ValidationError):
            ResolvedLearningDossier.model_validate({
                "dossier": _dossier_dict(0),
                "source": {"index": 1, "openalex_id": "https://openalex.org/W1"},
            })

    def test_legacy_drafts_validate_without_learning_output(self):
        from radar.schema.opportunities import LearningRadarDraft

        legacy = {"opportunities": [{"title": "T", "evidence": [0]}],
                  "ignore": [], "next_move": "N"}
        draft = LearningRadarDraft.model_validate(legacy)
        self.assertEqual(draft.opportunities[0].title, "T")
        self.assertEqual(draft.learning_dossiers, [])
        # Specialist output is exactly RadarDraft (no dossier field required).
        self.assertIsInstance(RadarDraft.model_validate(legacy), RadarDraft)

    def test_old_stored_reports_read_with_empty_dossiers(self):
        old = {"opportunities": [], "ignore": [], "next_move": "N"}
        report = RadarReport.model_validate(old)
        self.assertEqual(report.learning_dossiers, [])
        self.assertEqual(report.model_dump()["learning_dossiers"], [])


class TestLearningEvidence(unittest.TestCase):
    def test_valid_dossier_attaches_code_owned_abstract_source(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence

        draft = LearningRadarDraft.model_validate(
            {"opportunities": [], "next_move": "N",
             "learning_dossiers": [_dossier_dict(1)]})
        report = attach_evidence(draft, [_work(0), _work(1)])
        self.assertEqual(len(report.learning_dossiers), 1)
        resolved = report.learning_dossiers[0]
        self.assertEqual(resolved.source.index, 1)
        self.assertEqual(resolved.source.openalex_id, "https://openalex.org/W1")
        self.assertEqual(resolved.evidence_level, "abstract")
        self.assertEqual(resolved.dossier.paper_index, 1)

    def test_invalid_dossier_index_is_never_silently_dropped(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence

        draft = LearningRadarDraft.model_validate(
            {"opportunities": [], "next_move": "N",
             "learning_dossiers": [_dossier_dict(7)]})
        with self.assertRaises(ValueError):
            attach_evidence(draft, [_work(0)])

    def test_abstract_dossier_cannot_claim_pdf_page_provenance(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence

        draft = LearningRadarDraft.model_validate({
            "learning_dossiers": [_dossier_dict(supporting_pages=[1])]})
        with self.assertRaises(ValueError):
            attach_evidence(draft, [_work()])

    def test_duplicate_dossier_indices_are_rejected(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence

        with self.assertRaises(ValidationError):
            LearningRadarDraft.model_validate(
                {"opportunities": [], "next_move": "N",
                 "learning_dossiers": [_dossier_dict(0), _dossier_dict(0)]})

    def test_metadata_without_abstract_cannot_support_an_abstract_dossier(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence

        draft = LearningRadarDraft.model_validate({"learning_dossiers": [_dossier_dict()]})
        for abstract in ("", " \n "):
            with self.subTest(abstract=repr(abstract)), self.assertRaises(ValueError):
                attach_evidence(draft, [_work(abstract=abstract)])


class TestCompleteAbstracts(unittest.TestCase):
    def test_default_prompt_still_truncates_for_old_callers(self):
        from radar.agent.opportunity_analysis import build_prompt

        tail = "TAIL-MARKER-XYZ"
        prompt = build_prompt([_work(0, abstract="head " + ("word " * 500) + " " + tail)])
        self.assertNotIn("TAIL-MARKER-XYZ", prompt)

    def test_opt_in_keeps_full_abstract_tail(self):
        from radar.agent.opportunity_analysis import build_prompt

        tail = "TAIL-MARKER-XYZ"
        abstract = "head " + ("word " * 150) + " " + tail
        prompt = build_prompt([_work(0, abstract=abstract)], complete_abstracts=True)
        self.assertIn(tail, prompt)

    def test_prompt_discloses_actual_abstract_coverage(self):
        from radar.agent.opportunity_analysis import build_prompt

        self.assertIn("abstract excerpt", build_prompt([_work()]))
        self.assertIn("complete abstract", build_prompt([_work()], complete_abstracts=True))

    def test_oversized_leading_source_yields_zero_honest_coverage(self):
        from radar.agent import research_team

        pool = [_work(0, abstract="word " * 3000)]
        included = research_team.select_for_prompt(pool, 1)
        self.assertEqual(included, [])


class TestResearchSynthesis(unittest.TestCase):
    def test_substantive_narratives_survive_final_synthesis_rendering_and_sqlite(self):
        from radar.agent.research_team import research_candidates
        from radar.output.markdown import render_markdown
        from radar.processing.evidence import attach_evidence
        from radar.prompts.catalog import opportunity_analysis_prompt
        from radar.storage.sqlite import SQLiteStore

        # Synthetic boundary data checks retention, not scientific quality or
        # a model's ability to produce an actual postgraduate analysis.
        reasoning = "Condition on the stated assumptions; distinguish the claim from its interpretation. " * 12
        task_text = "Compare the assumption-dependent derivation with a counterexample and report the unresolved step. " * 5
        hypothesis = "Hypothesis: " + "This transfer remains unverified until the contrasting boundary conditions are tested. " * 5
        opportunity_text = ("This proposed check is not a reported result and requires the specified comparison. " * 30).strip()
        dossier = _dossier_dict(
            core_problem=reasoning, reported_contribution=reasoning,
            reasoning=reasoning, significance=reasoning,
            assumptions_limits=[task_text], connections=[hypothesis],
            open_questions=[task_text], study_tasks=[{
                "kind": "counterexample", "objective": task_text,
                "success_criterion": task_text, "missing_evidence": task_text}])
        calls = []

        def respond(messages, info):
            calls.append(info.instructions)
            payload = {"opportunities": [], "next_move": "Inspect the assumptions."}
            if info.instructions == opportunity_analysis_prompt().instructions:
                payload = {"learning_dossiers": [dossier], "next_move": opportunity_text,
                           "ignore": [task_text.strip()],
                           "opportunities": [{"title": "Discriminating check", "wow": opportunity_text,
                                              "investigate": opportunity_text, "reproduce": opportunity_text,
                                              "evidence": [0]}]}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

        result = research_candidates([_work()], model=FunctionModel(respond))
        self.assertEqual(len(calls), 4)
        report = attach_evidence(result.draft, list(result.included))
        self.assertEqual(report.learning_dossiers[0].dossier.model_dump(
            exclude={"supporting_pages", "source_passage_ids"}), dossier)
        for field in ("wow", "investigate", "reproduce"):
            self.assertEqual(getattr(report.opportunities[0].draft, field), opportunity_text)
        self.assertEqual(report.next_move, opportunity_text)
        self.assertEqual(report.ignore, [task_text.strip()])
        markdown = render_markdown(report)
        for text in (reasoning, task_text, hypothesis, opportunity_text):
            self.assertIn(text, markdown)
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = store.begin_run("analyze", "fixture")
            store.save_pool(run_id, [_work()])
            store.save_report(run_id, report, 1)
            stored = store.projection()[1][0][1]
            self.assertEqual(stored, report)
            self.assertEqual(render_markdown(stored), markdown)

    def test_final_synthesis_still_rejects_blank_unlabeled_hypotheses_and_invalid_indices(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import opportunity_analysis_prompt
        from radar.provider.strata import StrataError

        invalid = (
            _dossier_dict(reasoning=" \n "),
            _dossier_dict(assumptions_limits=[" \n "]),
            _dossier_dict(connections=["Unlabeled transfer remains unsupported."]),
            _dossier_dict(study_tasks=[{**_dossier_dict()["study_tasks"][0], "objective": " \n "}]),
            _dossier_dict(paper_index=True),
            _dossier_dict(paper_index=99),
        )
        for dossier in invalid:
            with self.subTest(dossier=dossier):
                def respond(messages, info):
                    payload = {"opportunities": [], "next_move": "Inspect the assumptions."}
                    if info.instructions == opportunity_analysis_prompt().instructions:
                        payload["learning_dossiers"] = [dossier]
                    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

                with self.assertRaises(StrataError):
                    research_candidates([_work()], model=FunctionModel(respond))

    def test_final_prompt_requires_depth_without_arbitrary_narrative_targets(self):
        from radar.prompts.catalog import opportunity_analysis_prompt

        prompt = opportunity_analysis_prompt()
        for requirement in ("assumption-dependent reasoning", "reported claims", "interpretations",
                            "discriminating", "uncertainty", "fictional explanation"):
            self.assertIn(requirement, prompt.instructions)
        self.assertNotIn("250-400", prompt.instructions)
        self.assertNotIn("<=2 sentences", prompt.task_template)

    def test_synthesis_accepts_legacy_opportunity_only_draft(self):
        from radar.agent.research_team import research_candidates

        def respond(messages, info):
            draft = RadarDraft(opportunities=[OpportunityDraft(title="T", evidence=[0])],
                               next_move="N")
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                     draft.model_dump())])

        result = research_candidates([_work(0), _work(1)], model=FunctionModel(respond))
        self.assertEqual(result.draft.opportunities[0].title, "T")

    def test_synthesis_rejects_out_of_range_dossier_index(self):
        from radar.agent.research_team import research_candidates
        from radar.provider.strata import StrataError

        def respond(messages, info):
            payload = {"opportunities": [], "next_move": "N",
                       "learning_dossiers": [_dossier_dict(99)]}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

        with self.assertRaises(StrataError):
            research_candidates([_work(0)], model=FunctionModel(respond))

    def test_all_stages_see_complete_abstracts(self):
        from radar.agent.research_team import research_candidates

        tail = "TAIL-MARKER-XYZ"
        pool = [_work(i, abstract=f"intro {'word ' * 150}{tail} end {i}") for i in range(2)]

        def respond(messages, info):
            prompt = next(p.content for m in messages for p in m.parts
                          if isinstance(p, UserPromptPart))
            self.assertIn(tail, prompt)
            draft = RadarDraft(next_move="N")
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                     draft.model_dump())])

        result = research_candidates(pool, model=FunctionModel(respond))
        self.assertEqual(len(result.included), 2)

    def test_final_synthesis_produces_a_dossier_without_an_extra_model_call(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import opportunity_analysis_prompt
        from radar.processing.evidence import attach_evidence

        calls = []

        def respond(messages, info):
            calls.append(info.instructions)
            payload = RadarDraft(next_move="Inspect the assumptions.").model_dump()
            if info.instructions == opportunity_analysis_prompt().instructions:
                payload["learning_dossiers"] = [_dossier_dict(0, connections=[])]
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

        result = research_candidates([_work()], model=FunctionModel(respond))
        self.assertEqual(len(calls), 4)
        report = attach_evidence(result.draft, list(result.included))
        self.assertEqual(report.learning_dossiers[0].source.openalex_id, _work().openalex_id)
        self.assertEqual(report.learning_dossiers[0].dossier.study_tasks[0].kind, "derivation")


class TestRenderingAndStorage(unittest.TestCase):
    def test_markdown_renders_grounded_dossier_as_hypothesis(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence
        from radar.output.markdown import render_markdown

        draft = LearningRadarDraft.model_validate(
            {"opportunities": [], "next_move": "N",
             "learning_dossiers": [_dossier_dict(0)]})
        md = render_markdown(attach_evidence(draft, [_work(0)]))
        self.assertIn("Learning dossier", md)
        self.assertIn("Hypothesis:", md)
        self.assertIn("https://example.org/paper-0", md)
        self.assertIn("abstract", md.lower())
        self.assertLess(md.index("Learning dossier"), md.index("## Ignore"))

    def test_opportunity_only_report_rendering_is_unchanged(self):
        from radar.processing.evidence import attach_evidence
        from radar.output.markdown import render_markdown

        md = render_markdown(attach_evidence(RadarDraft(next_move="N"), []))
        self.assertNotIn("Learning dossier", md)

    def test_sqlite_rejects_dossier_source_outside_pool(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence
        from radar.storage.sqlite import SQLiteStore, StorageError

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = store.begin_run("analyze", "OpenAlex")
                store.save_pool(run_id, [_work(0)])
                bad = LearningRadarDraft.model_validate(
                    {"opportunities": [], "next_move": "N",
                     "learning_dossiers": [_dossier_dict(0)]})
                # Resolve inside the run, then point the source outside the pool.
                report = attach_evidence(bad, [_work(0)])
                report.learning_dossiers[0].source.openalex_id = "https://openalex.org/W999"
                with self.assertRaises(StorageError):
                    store.save_report(run_id, report, 1)

    def test_sqlite_accepts_grounded_dossier(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence
        from radar.storage.sqlite import SQLiteStore

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = store.begin_run("analyze", "OpenAlex")
                store.save_pool(run_id, [_work(0)])
                draft = LearningRadarDraft.model_validate(
                    {"opportunities": [], "next_move": "N",
                     "learning_dossiers": [_dossier_dict(0)]})
                store.save_report(run_id, attach_evidence(draft, [_work(0)]), 1)
                _, reports = store.projection()
                self.assertEqual(len(reports[0][1].learning_dossiers), 1)

    def test_sqlite_rejects_dossier_outside_reported_analyzed_index_range(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence
        from radar.storage.sqlite import SQLiteStore, StorageError

        papers = [_work(0), _work(1)]
        draft = LearningRadarDraft.model_validate({"learning_dossiers": [_dossier_dict(1)]})
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = store.begin_run("analyze", "fixture")
            store.save_pool(run_id, papers)
            for selected in (0, 1):
                with self.subTest(selected=selected), self.assertRaises(StorageError):
                    store.save_report(run_id, attach_evidence(draft, papers), selected)
            self.assertEqual(store.projection()[1], [])

    def test_sqlite_uses_run_pool_abstract_not_the_supplied_draft_or_current_catalogue(self):
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence
        from radar.storage.sqlite import SQLiteStore, StorageError

        draft = LearningRadarDraft.model_validate({"learning_dossiers": [_dossier_dict()]})
        report = attach_evidence(draft, [_work()])
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            original = store.begin_run("analyze", "fixture")
            store.save_pool(original, [_work(abstract="")])
            later = store.begin_run("collect", "fixture")
            store.save_pool(later, [_work()])
            with self.assertRaises(StorageError):
                store.save_report(original, report, 1)
            self.assertEqual(store.projection()[1], [])

    def test_falkor_preflights_dossier_before_deletion_and_projects_grounded_edges(self):
        from radar.storage.falkor import FalkorGraph
        from radar.schema.opportunities import LearningRadarDraft
        from radar.processing.evidence import attach_evidence

        try:
            import redis  # noqa: F401
            import redislite  # noqa: F401
        except ImportError:
            self.skipTest("bundled Falkor engine unavailable")
            return
        if not (getattr(redislite, "__redis_executable__", None)):
            self.skipTest("bundled Falkor engine unavailable")
            return
        import os as _os
        if not _os.path.isfile(redislite.__redis_executable__):
            self.skipTest("bundled Falkor engine unavailable")
            return
        paper = _work(0)
        good = attach_evidence(LearningRadarDraft.model_validate(
            {"opportunities": [], "next_move": "N",
             "learning_dossiers": [_dossier_dict(0)]}), [paper])
        bad = good.model_copy(deep=True)
        bad.learning_dossiers[0].source.openalex_id = "https://openalex.org/W999"
        with tempfile.TemporaryDirectory() as directory:
            with FalkorGraph(directory) as graph:
                graph.rebuild([paper], [("run-1", good)])
                dossiers = graph.query(
                    "MATCH (r:Run)-[:HAS_LEARNING_DOSSIER]->(d:LearningDossier) "
                    "RETURN r.id, d.evidence_level")
                self.assertEqual(dossiers, [["run-1", "abstract"]])
                links = graph.query(
                    "MATCH (d:LearningDossier)-[:ABOUT_PAPER]->(p:Paper) "
                    "RETURN p.openalex_id")
                self.assertEqual(links, [[paper.openalex_id]])
                with self.assertRaises(Exception):
                    graph.rebuild([paper], [("bad-run", bad)])
                self.assertEqual(
                    graph.query("MATCH (p:Paper) RETURN p.openalex_id"),
                    [[paper.openalex_id]])


if __name__ == "__main__":
    unittest.main()
