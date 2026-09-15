"""Python wrapper for the METEOR 1.5 Java implementation."""

import atexit
import os
import subprocess
import threading
from pathlib import Path


class Meteor:
    def __init__(self):
        self.lock = threading.Lock()
        self.meteor_p = None

        try:
            import pycocoevalcap.meteor as meteor_package
        except ImportError as exc:
            raise RuntimeError(
                "METEOR evaluation requires pycocoevalcap==1.2"
            ) from exc

        module_dir = Path(meteor_package.__file__).resolve().parent
        meteor_jar = module_dir / "meteor-1.5.jar"
        meteor_paraphrase = module_dir / "data" / "paraphrase-en.gz"
        missing_assets = [
            path
            for path in (meteor_jar, meteor_paraphrase)
            if not path.is_file()
        ]
        if missing_assets:
            missing_text = ", ".join(str(path) for path in missing_assets)
            raise FileNotFoundError(
                f"METEOR runtime assets not found in pycocoevalcap: {missing_text}"
            )

        self.module_dir = module_dir
        memory = os.environ.get("METEOR_JAVA_MEMORY", "2G")
        command = [
            "java",
            f"-Xmx{memory}",
            "-jar",
            meteor_jar.name,
            "-",
            "-",
            "-stdio",
            "-l",
            "en",
            "-norm",
        ]
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        try:
            self.meteor_p = subprocess.Popen(
                command,
                cwd=self.module_dir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Java is required for METEOR evaluation but was not found on PATH"
            ) from exc
        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.meteor_p is not None:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def compute_score(self, gts, res):
        if gts.keys() != res.keys():
            raise ValueError("references and hypotheses must have identical keys")

        image_ids = list(gts)
        scores = []
        eval_line = "EVAL"
        with self.lock:
            for image_id in image_ids:
                if len(res[image_id]) != 1:
                    raise ValueError("each sample must contain exactly one hypothesis")
                stat = self._stat(res[image_id][0], gts[image_id])
                eval_line += f" ||| {stat}"

            self._write_line(eval_line)
            for _ in image_ids:
                scores.append(float(self._read_line()))
            score = float(self._read_line())
        return score, scores

    def _stat(self, hypothesis, references):
        hypothesis = hypothesis.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(
            ("SCORE", " ||| ".join(references), hypothesis)
        )
        self._write_line(score_line)
        return self._read_line()

    def _write_line(self, value):
        if self.meteor_p is None or self.meteor_p.stdin is None:
            raise RuntimeError("METEOR process is not running")
        try:
            self.meteor_p.stdin.write((value + "\n").encode("utf-8"))
            self.meteor_p.stdin.flush()
        except BrokenPipeError as exc:
            stderr = b""
            if self.meteor_p.stderr is not None:
                stderr = self.meteor_p.stderr.read()
            detail = stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"METEOR process terminated: {detail}") from exc

    def _read_line(self):
        if self.meteor_p is None or self.meteor_p.stdout is None:
            raise RuntimeError("METEOR process is not running")
        value = self.meteor_p.stdout.readline()
        if not value:
            stderr = b""
            if self.meteor_p.stderr is not None:
                stderr = self.meteor_p.stderr.read()
            detail = stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"METEOR returned no output: {detail}")
        return value.decode("utf-8").strip()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
