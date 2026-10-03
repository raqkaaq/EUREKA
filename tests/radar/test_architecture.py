"""Architecture boundary tests: leaf modules stay service-free and the
dependency direction holds (adapters never import CLI/pipeline/output)."""

from __future__ import annotations

import pathlib
import sys
import unittest

_PACKAGE = pathlib.Path(__file__).resolve().parent.parent


class TestServiceFreeLeaves(unittest.TestCase):
    def test_schema_and_config_import_without_services(self):
        import os
        import subprocess

        code = (
            "import sys;"
            "sys.modules['httpx'] = None; sys.modules['httpx2'] = None;"
            "sys.modules['pydantic_ai'] = None; sys.modules['openai'] = None;"
            "import radar.schema.papers, radar.schema.opportunities,"
            " radar.config.runtime, radar.config.interests;"
            "print('service-free-ok')"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_PACKAGE.parent.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        )
        self.assertIn("service-free-ok", proc.stdout, proc.stderr[-2000:])


class TestDependencyDirection(unittest.TestCase):
    def test_adapters_never_import_cli_pipeline_output(self):
        adapter_dirs = ("source", "provider", "agent", "processing", "storage")
        offenders: list[str] = []
        for dirname in adapter_dirs:
            for path in (_PACKAGE / dirname).glob("*.py"):
                text = path.read_text(encoding="utf-8")
                for lineno, line in enumerate(text.splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for banned in ("radar.cli", "radar.pipeline", "radar.output"):
                        if banned in stripped and "output/markdown" not in stripped:
                            offenders.append(f"{path.name}:{lineno}: {stripped}")
        self.assertEqual(offenders, [])

    def test_output_never_imports_pipeline_or_services(self):
        offenders: list[str] = []
        for path in (_PACKAGE / "output").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for banned in ("radar.cli", "radar.pipeline", "radar.source",
                               "radar.provider", "radar.agent", "radar.storage",
                               "httpx", "pydantic_ai"):
                    if banned in stripped:
                        offenders.append(f"{path.name}:{lineno}: {stripped}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
