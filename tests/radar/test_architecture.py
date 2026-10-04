"""Architecture boundary tests: leaf modules stay service-free and the
dependency direction holds (adapters never import CLI/pipeline/output)."""

from __future__ import annotations

import ast
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import radar

_PACKAGE = pathlib.Path(radar.__file__).resolve().parent


def _forbidden_imports(
    source: str, package: tuple[str, ...], banned: tuple[str, ...],
) -> list[tuple[int, str]]:
    """Resolve static imports only; docstrings and ordinary data aren't edges."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else ()
            module = ".".join((*base, node.module or "")).rstrip(".")
            targets = [f"{module}.{alias.name}" for alias in node.names]
        else:
            continue
        found.extend((node.lineno, target) for target in targets
                     if any(target == name or target.startswith(name + ".") for name in banned))
    return found


class TestGuardCoverage(unittest.TestCase):
    def test_guards_target_the_imported_application_package(self):
        self.assertEqual(_PACKAGE, pathlib.Path(radar.__file__).resolve().parent)

    def test_empty_scans_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                sys.modules[__name__], "_PACKAGE", pathlib.Path(tmp)):
            for method in ("test_adapters_never_import_cli_pipeline_output",
                           "test_output_never_imports_pipeline_or_services"):
                with self.subTest(method=method):
                    case = TestDependencyDirection(method)
                    with self.assertRaises(AssertionError):
                        getattr(case, method)()

    def test_relative_and_root_imports_are_detected(self):
        sources = ("from radar import pipeline", "from .. import output",
                   "from ..cli import main", "import radar.pipeline as orchestration")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for dirname in ("source", "provider", "agent", "processing", "storage"):
                (root / dirname).mkdir()
                (root / dirname / "__init__.py").write_text("", encoding="utf-8")
            path = root / "agent" / "nested" / "bad.py"
            path.parent.mkdir()
            for source in sources:
                # Nested relative imports need one more parent level.
                path.write_text(source.replace("from ..", "from ..."), encoding="utf-8")
                with self.subTest(source=source), mock.patch.object(
                        sys.modules[__name__], "_PACKAGE", root):
                    with self.assertRaises(AssertionError):
                        TestDependencyDirection().test_adapters_never_import_cli_pipeline_output()

    def test_comments_docstrings_and_values_are_not_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "output").mkdir()
            (root / "output" / "markdown.py").write_text(
                '\"\"\"Compare with radar.pipeline and radar.provider.\"\"\"\n'
                '# import httpx\nVALUE = "radar.cli"\n'
                'from radar.schema import opportunities\n', encoding="utf-8")
            with mock.patch.object(sys.modules[__name__], "_PACKAGE", root):
                TestDependencyDirection().test_output_never_imports_pipeline_or_services()

    def test_output_service_and_adapter_imports_are_detected(self):
        sources = ("import httpx2", "from pydantic_ai import Agent", "import openai",
                   "from radar import source", "from ..provider import freetoken")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "output").mkdir()
            path = root / "output" / "bad.py"
            for source in sources:
                path.write_text(source, encoding="utf-8")
                with self.subTest(source=source), mock.patch.object(
                        sys.modules[__name__], "_PACKAGE", root):
                    with self.assertRaises(AssertionError):
                        TestDependencyDirection().test_output_never_imports_pipeline_or_services()


class TestServiceFreeLeaves(unittest.TestCase):
    def test_schema_and_config_import_without_services(self):
        import os
        import subprocess

        code = (
            "import sys;"
            "sys.modules['httpx'] = None; sys.modules['httpx2'] = None;"
            "sys.modules['pydantic_ai'] = None; sys.modules['openai'] = None;"
            "import radar.schema.papers, radar.schema.opportunities,"
            " radar.config.runtime, radar.config.interests, radar.config.searches,"
            " radar.schema.configuration, radar.prompts.catalog;"
            "from radar.config.interests import default_profile;"
            "from radar.config.searches import build_query_plan;"
            "from radar.prompts.catalog import screening_questions;"
            "build_query_plan(default_profile()); screening_questions();"
            "print('service-free-ok')"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_PACKAGE.parent) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env,
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("service-free-ok", proc.stdout, proc.stderr[-2000:])


class TestDependencyDirection(unittest.TestCase):
    def offenders_in(self, directory: str, banned: tuple[str, ...]) -> list[str]:
        paths = sorted((_PACKAGE / directory).rglob("*.py"))
        self.assertTrue(paths, f"No application modules scanned in {directory}")
        offenders = []
        for path in paths:
            relative = path.relative_to(_PACKAGE)
            package = ("radar", *relative.parent.parts)
            for line, target in _forbidden_imports(path.read_text(encoding="utf-8"), package, banned):
                offenders.append(f"{relative}:{line}: {target}")
        return offenders

    def test_adapters_never_import_cli_pipeline_output(self):
        adapter_dirs = ("source", "provider", "agent", "processing", "storage")
        offenders: list[str] = []
        for dirname in adapter_dirs:
            offenders.extend(self.offenders_in(dirname, ("radar.cli", "radar.pipeline", "radar.output")))
        self.assertEqual(offenders, [])

    def test_output_never_imports_pipeline_or_services(self):
        offenders = self.offenders_in("output", (
            "radar.cli", "radar.pipeline", "radar.source", "radar.provider",
            "radar.agent", "radar.storage", "httpx", "httpx2", "pydantic_ai", "openai"))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
