"""ScoreEvent: the one message shape between the machinery and a display.
StageSchedule: trial-stage labels per volume (docs/session_api.md).

Every message on the score queue is a dict with at least `kind` and `source`;
`ScoreEvent.from_message` does no guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

KINDS = ("tick", "score", "scorer", "error", "end")


@dataclass
class ScoreEvent:
    kind: str                            # tick | score | scorer | error | end
    source: str                          # "pipeline", an extra process's name, or "session"
    volume_idx: Optional[int] = None
    timestamp: Optional[float] = None    # when the score was written
    watchdog_timestamp: Optional[float] = None   # when the DICOM was picked up
    analysis_timestamp: Optional[float] = None   # right after scoring
    score_raw: Optional[float] = None    # the decoder score (pipeline only)
    score_z: Optional[float] = None
    score_original: Optional[float] = None
    reg_ready: Optional[bool] = None     # always set on kind="score"
    stage: Optional[str] = None          # trial-stage label (scores.csv event_type)
    meta: dict = field(default_factory=dict)   # everything else (scorer values, error text, exit code)

    _FIELDS = ("volume_idx", "timestamp", "watchdog_timestamp", "analysis_timestamp", "score_raw", "score_z",
               "score_original", "reg_ready", "stage")

    @classmethod
    def from_message(cls, message: dict) -> "ScoreEvent":
        if not isinstance(message, dict) or "kind" not in message or "source" not in message:
            raise ValueError(f"score queue message must be a dict with 'kind' and 'source': {message!r}")
        kind = str(message["kind"])
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        m = dict(message)
        kwargs: dict[str, Any] = {"kind": kind, "source": str(m.pop("source"))}
        m.pop("kind")
        for key in cls._FIELDS:
            if key in m:
                v = m.pop(key)
                if key == "volume_idx":
                    v = None if v is None else int(v)
                elif key == "reg_ready":
                    v = None if v is None else bool(v)
                elif key != "stage":
                    v = None if v is None else float(v)
                kwargs[key] = v
        if kind == "score" and kwargs.get("reg_ready") is None:
            raise ValueError("kind='score' messages must carry reg_ready")
        kwargs["meta"] = m
        return cls(**kwargs)

    def to_message(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind, "source": self.source}
        for key in self._FIELDS:
            v = getattr(self, key)
            if v is not None:
                out[key] = v
        out.update(self.meta)
        return out


# ----------------------------------------------------------------------------- stages
@dataclass
class StageSchedule:
    """Volume v (1-based): pre_label while v <= skip_first_trs; baseline_label for
    the next baseline_trs volumes; then `cycle` repeats n_cycles times (None =
    forever); post_label afterwards. Empty cycle: only pre/baseline are labelled."""

    skip_first_trs: int = 0
    baseline_trs: int = 0
    cycle: list = field(default_factory=list)   # [(name, n_trs), ...]
    n_cycles: Optional[int] = None
    pre_label: str = "background"
    baseline_label: str = "background"
    post_label: str = "post_task"

    @classmethod
    def from_config(cls, cfg, skip_first_trs: int = 0) -> Optional["StageSchedule"]:
        if cfg is None:
            return None
        return cls(skip_first_trs=int(skip_first_trs), baseline_trs=int(cfg.baseline_trs),
                   cycle=[(str(n), int(t)) for n, t in cfg.cycle], n_cycles=cfg.n_cycles,
                   pre_label=cfg.pre_label, baseline_label=cfg.baseline_label, post_label=cfg.post_label)

    def to_metadata(self) -> dict:
        return {"version": 1, "skip_first_trs": int(self.skip_first_trs), "baseline_trs": int(self.baseline_trs),
                "cycle": [{"name": str(n), "trs": int(t)} for n, t in self.cycle],
                "n_cycles": None if self.n_cycles is None else int(self.n_cycles),
                "pre_label": self.pre_label, "baseline_label": self.baseline_label, "post_label": self.post_label}

    @property
    def cycle_length(self) -> int:
        return sum(max(0, int(t)) for _, t in self.cycle)

    def label_for_volume(self, volume_idx: int) -> Optional[str]:
        if volume_idx <= self.skip_first_trs:
            return self.pre_label
        exp_tr = volume_idx - self.skip_first_trs
        if exp_tr <= self.baseline_trs:
            return self.baseline_label
        expanded: list[str] = []
        for name, t in self.cycle:
            expanded.extend([str(name)] * max(0, int(t)))
        if not expanded:
            return None
        tr_after = exp_tr - self.baseline_trs
        cycle_idx = (tr_after - 1) // len(expanded)
        if self.n_cycles is not None and cycle_idx >= int(self.n_cycles):
            return self.post_label
        return expanded[(tr_after - 1) % len(expanded)]

    def windows(self) -> list[dict]:
        """[{cycle, stage, start_tr, end_tr}] in experiment TRs (after skip_first_trs), baseline as cycle 0."""
        out, end = [], 0
        if self.baseline_trs > 0:
            end += self.baseline_trs
            out.append({"cycle": 0, "stage": self.baseline_label, "start_tr": 1, "end_tr": end})
        n = self.n_cycles or 0
        for c in range(1, n + 1):
            for name, t in self.cycle:
                start = end + 1
                end += max(0, int(t))
                out.append({"cycle": c, "stage": str(name), "start_tr": start, "end_tr": end})
        return out
