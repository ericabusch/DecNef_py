"""Session API: start the real-time machinery, consume scores, stop cleanly.

A *launcher* (a study's script) owns the display and trial structure; this
module owns everything else that the ``rt_*_parallel.py`` launchers used to
copy by hand: settings loading, ``session_metadata.json``, the pipeline child
process, the optional BIOPAC listener process, extra scorer processes (e.g.
the resting-state PCA scorer), the score queue, and shutdown.

    from rt_session import SessionConfig, StageSchedule, start_session

    cfg = SessionConfig(sub="00086", day="3", run="4", base_data=..., incoming_root=...,
                        stages=StageSchedule(skip_first_trs=0, baseline_trs=20,
                                             cycle=[("iti", 3), ("cue", 4), ("scans", 3), ("delay", 3), ("feedback", 3)],
                                             n_cycles=20))
    with start_session(cfg) as session:           # spawns pipeline (+ physio, extras)
        for event in session.scores(timeout=0.02):  # display runs here, in the main process
            if event.kind == "score": ...
            if done: break
    # leaving the block stops every child, in order, even on Ctrl-C or an exception

Wire format: processes still exchange plain dicts on the score queue (so
unported launchers and ``rs_pca_runtime`` are unaffected); ``ScoreEvent``
is built on the consumer side. See ``docs/session_api.md`` for the schemas.
"""
from __future__ import annotations

import atexit
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterator, Optional

log = logging.getLogger(__name__)

STAGES_SCHEMA_VERSION = 1
PHYS_REG_COUNTS = {"RICOR8": 8, "RVT5": 5, "RVT+RICOR13": 13}


class SessionError(RuntimeError):
    """The pipeline (or an extra process) exited with an error."""


# --------------------------------------------------------------------------- schemas
@dataclass
class ScoreEvent:
    """One message from the session, in a fixed shape.

    kind: "tick" (volume processed, no score), "score" (decoder score from the
    pipeline), "scorer" (message from an extra process, e.g. the PCA scorer),
    "error" (a process reported or hit an error), "end" (pipeline exited; last event).
    Fields missing from a message are None (``reg_ready`` None means "the message
    had no reg_ready key"; the old launchers defaulted it differently, so the
    consumer decides). Every other key of the message is kept verbatim in ``meta``.
    """

    kind: str
    volume_idx: Optional[int] = None
    timestamp: Optional[float] = None
    watchdog_timestamp: Optional[float] = None
    analysis_timestamp: Optional[float] = None
    score_raw: Optional[float] = None
    score_z: Optional[float] = None
    score_original: Optional[float] = None
    reg_ready: Optional[bool] = None
    stage: Optional[str] = None          # the pipeline's event_type (trial-stage label)
    source: str = "pipeline"             # "pipeline" | ProcessSpec.name | "session"
    meta: dict = field(default_factory=dict)

    _KNOWN = ("volume_idx", "timestamp", "watchdog_timestamp", "analysis_timestamp", "score_raw",
              "score_z", "score_original", "reg_ready", "event_type", "kind")

    @classmethod
    def from_message(cls, message: dict, source: str = "pipeline") -> "ScoreEvent":
        """Build an event from a queue dict (the pipeline's, a scorer's, or a legacy one)."""
        m = dict(message)
        kind = m.get("kind")
        if kind is None:
            # legacy messages carry no kind: infer it the way the old displays did
            if "error" in m:
                kind = "error"
            elif source != "pipeline":
                kind = "scorer"
            elif "score_raw" in m:
                kind = "score"
            else:
                kind = "tick"
        vol = m.get("volume_idx", m.get("volume_index"))
        reg = m.get("reg_ready")

        def _f(key):
            v = m.get(key)
            return None if v is None else float(v)

        return cls(
            kind=str(kind),
            volume_idx=None if vol is None else int(vol),
            timestamp=_f("timestamp"),
            watchdog_timestamp=_f("watchdog_timestamp"),
            analysis_timestamp=_f("analysis_timestamp"),
            score_raw=_f("score_raw"),
            score_z=_f("score_z"),
            score_original=_f("score_original"),
            reg_ready=None if reg is None else bool(reg),
            stage=m.get("event_type"),
            source=source,
            meta={k: v for k, v in m.items() if k not in cls._KNOWN and k != "volume_index"},
        )

    def to_message(self) -> dict:
        """The dict form (what a legacy consumer expects); absent fields are omitted."""
        out: dict[str, Any] = {"kind": self.kind}
        for key in ("volume_idx", "timestamp", "watchdog_timestamp", "analysis_timestamp",
                    "score_raw", "score_z", "score_original", "reg_ready"):
            v = getattr(self, key)
            if v is not None:
                out[key] = v
        if self.stage is not None:
            out["event_type"] = self.stage
        out.update(self.meta)
        return out


@dataclass
class StageSchedule:
    """Trial-stage labels per volume, written to session_metadata["stages"].

    Volume v (1-based) gets: pre_label while v <= skip_first_trs; baseline_label
    for the next baseline_trs volumes; then the cycle repeats n_cycles times
    (None = forever); post_label afterwards. Labels land in scores.csv
    (event_type) and in ScoreEvent.stage.
    """

    skip_first_trs: int = 0
    baseline_trs: int = 0
    cycle: list[tuple[str, int]] = field(default_factory=list)   # ordered (stage name, n_trs)
    n_cycles: Optional[int] = None
    pre_label: str = "background"
    baseline_label: str = "background"
    post_label: str = "post_task"

    def to_metadata(self) -> dict:
        return {
            "version": STAGES_SCHEMA_VERSION,
            "skip_first_trs": int(self.skip_first_trs),
            "baseline_trs": int(self.baseline_trs),
            "cycle": [{"name": str(n), "trs": int(t)} for n, t in self.cycle],
            "n_cycles": None if self.n_cycles is None else int(self.n_cycles),
            "pre_label": self.pre_label,
            "baseline_label": self.baseline_label,
            "post_label": self.post_label,
        }

    def label_for_volume(self, volume_idx: int) -> Optional[str]:
        """Reference implementation of the labelling (rt_pipeline.ScoreEventTracker mirrors it)."""
        return label_for_volume(self.to_metadata(), volume_idx)


def label_for_volume(stages: dict, volume_idx: int) -> Optional[str]:
    """Label for a 1-based volume index from a stages dict (schema above)."""
    skip = int(stages.get("skip_first_trs", 0) or 0)
    baseline = int(stages.get("baseline_trs", 0) or 0)
    if volume_idx <= skip:
        return stages.get("pre_label", "background")
    exp_tr = volume_idx - skip
    if exp_tr <= baseline:
        return stages.get("baseline_label", "background")
    cycle: list[str] = []
    for item in stages.get("cycle", []) or []:
        cycle.extend([str(item["name"])] * max(0, int(item.get("trs", 0) or 0)))
    if not cycle:
        return None
    tr_after = exp_tr - baseline
    n_cycles = stages.get("n_cycles")
    cycle_idx = (tr_after - 1) // len(cycle)
    if n_cycles is not None and cycle_idx >= int(n_cycles):
        return stages.get("post_label", "post_task")
    return cycle[(tr_after - 1) % len(cycle)]


@dataclass
class PhysioConfig:
    """BIOPAC physio regressors (the ten `biopac_*` settings + the listener process)."""

    enable: bool = False
    host: str = "0.0.0.0"
    port: int = 15000
    timeout: float = 0.3
    phys_reg: str = "RICOR8"            # RICOR8 | RVT5 | RVT+RICOR13
    handshake: bool = True
    start_online_only: bool = False
    mode: str = "tcp"                   # tcp | file
    file: Optional[Path] = None
    poll_interval: float = 0.05
    listener: bool = False              # run the TCP receiver in its own process, writing the CSV

    def settings(self) -> dict:
        return {
            "enable_biopac_physio": self.enable,
            "biopac_host": self.host,
            "biopac_port": self.port,
            "biopac_timeout": self.timeout,
            "biopac_phys_reg": self.phys_reg,
            "biopac_handshake": self.handshake,
            "biopac_start_online_only": self.start_online_only,
            "biopac_mode": self.mode,
            "biopac_file": Path(self.file) if self.file else None,
            "biopac_poll_interval": self.poll_interval,
        }


@dataclass
class ProcessSpec:
    """An extra process the session starts and stops (e.g. a scorer).

    ``target(**kwargs)`` runs in a spawned child; the session adds the score
    queue under ``pass_queue`` and a stop Event under ``pass_stop_event`` (set
    either to None to omit). It is started after the pipeline, and stopped
    before it: stop event -> join(stop_timeout_s) -> terminate.
    """

    name: str
    target: Callable
    kwargs: dict = field(default_factory=dict)
    pass_queue: Optional[str] = "score_queue"
    pass_stop_event: Optional[str] = "stop_event"
    stop_timeout_s: float = 5.0


@dataclass
class SessionConfig:
    """Everything a launcher tells the session (the old argparse values)."""

    sub: str
    day: str
    run: str
    base_data: Path
    incoming_root: Path
    settings_file: Optional[Path] = None             # JSON for rt_global_settings.load_regressor_settings
    settings_overrides: dict = field(default_factory=dict)   # applied to the pipeline's settings after the file
    decoder_template: Optional[Path] = None
    decoder_roi_txt: Optional[Path] = None
    reference_score_run: Optional[str] = None       # --rs: run whose scores.csv gives z-score stats
    enable_scoring: bool = True
    enable_original_score: bool = False
    ap_block: Optional[int] = None
    pa_block: Optional[int] = None
    physio: Optional[PhysioConfig] = None            # None = BIOPAC off (as with no --biopac-* flags)
    qc_viewer: Optional[bool] = None                 # override enable_volume_streamer; None = settings file
    stages: Optional[StageSchedule] = None
    metadata: dict = field(default_factory=dict)     # launcher-owned keys merged into session_metadata.json
    extra_processes: list[ProcessSpec] = field(default_factory=list)
    queue_size: int = 200
    stop_timeout_s: float = 10.0                     # cooperative stop before terminate()


# --------------------------------------------------------------------------- process targets
def _pipeline_main(rt_cfg, settings: dict, score_queue, stop_event) -> None:
    """Pipeline child: apply the parent's settings, then run (cooperative stop)."""
    from rt_pipeline import REGRESSOR_SETTINGS, run_rt_pipeline

    REGRESSOR_SETTINGS.update(settings)
    run_rt_pipeline(rt_cfg, score_queue, stop_event=stop_event)


def _physio_main(config, stop_event) -> None:
    """BIOPAC listener child (the old `_run_biopac_listener`)."""
    from biopac_rt.biopac_receiver import BiopacRetroTSReceiver

    receiver = BiopacRetroTSReceiver(config)
    receiver.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        receiver.stop()


# --------------------------------------------------------------------------- helpers shared with launchers
def merge_session_metadata(run_dir: Path, payload: dict) -> None:
    """Shallow-merge `payload` into run_dir/session_metadata.json (the launchers' `_merge_session_metadata`)."""
    metadata_path = Path(run_dir) / "session_metadata.json"
    data: dict = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(payload)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _stop_process(proc, stop_event, timeout: float, queue=None, name: str = "") -> Optional[int]:
    """Cooperative stop, then terminate, then kill. Drains `queue` while waiting so a
    child blocked on a full queue pipe can exit. Returns the exit code (None if never started)."""
    if proc is None:
        return None
    if stop_event is not None:
        stop_event.set()
    deadline = time.time() + max(0.0, timeout)
    while proc.is_alive() and time.time() < deadline:
        _drain(queue)
        proc.join(timeout=0.05)
    if proc.is_alive():
        log.warning("[session] %s did not stop within %.1fs; terminating", name or proc.name, timeout)
        proc.terminate()
        deadline = time.time() + 5.0
        while proc.is_alive() and time.time() < deadline:
            _drain(queue)
            proc.join(timeout=0.05)
    if proc.is_alive():
        log.error("[session] %s ignored SIGTERM; killing", name or proc.name)
        proc.kill()
        proc.join(timeout=2.0)
    return proc.exitcode


def _drain(queue) -> list:
    out = []
    if queue is None:
        return out
    try:
        while True:
            out.append(queue.get_nowait())
    except Empty:
        pass
    except (OSError, ValueError):   # queue closed
        pass
    return out


# --------------------------------------------------------------------------- the session
class Session:
    """Owns the child processes and the score queue. Create with `start_session`
    (real pipeline) or directly with a custom `pipeline_target` (tests)."""

    def __init__(
        self,
        run_dir: Path,
        pipeline_target: Callable,
        pipeline_args: tuple = (),
        *,
        physio_target: Optional[Callable] = None,
        physio_args: tuple = (),
        extra_processes: Optional[list[ProcessSpec]] = None,
        queue_size: int = 200,
        stop_timeout_s: float = 10.0,
        rt_cfg=None,
        settings: Optional[dict] = None,
    ):
        self.run_dir = Path(run_dir)
        self.rt_cfg = rt_cfg
        self.settings = settings or {}
        self._ctx = mp.get_context("spawn")
        self.queue = self._ctx.Queue(maxsize=int(queue_size))
        self._pipeline_target = pipeline_target
        self._pipeline_args = tuple(pipeline_args)
        self._physio_target = physio_target
        self._physio_args = tuple(physio_args)
        self._extras = list(extra_processes or [])
        self._stop_timeout_s = float(stop_timeout_s)

        self.pipeline_stop = self._ctx.Event()
        self.pipeline_process = None
        self.physio_stop = self._ctx.Event() if physio_target is not None else None
        self.physio_process = None
        self.extra_stops: dict[str, Any] = {}
        self.extra_processes: dict[str, Any] = {}
        self.exit_codes: dict[str, Optional[int]] = {}
        self._started = False
        self._stopped = False
        self._end_emitted = False
        self._pending: list[ScoreEvent] = []
        self._source_by_pid: dict[int, str] = {}

    # ---- lifecycle
    def start(self) -> "Session":
        if self._started:
            return self
        self._started = True
        atexit.register(self.stop)
        try:
            if self._physio_target is not None:
                self.physio_process = self._ctx.Process(
                    target=self._physio_target, args=(*self._physio_args, self.physio_stop), name="physio")
                self.physio_process.start()
            self.pipeline_process = self._ctx.Process(
                target=self._pipeline_target, args=(*self._pipeline_args, self.queue, self.pipeline_stop), name="pipeline")
            self.pipeline_process.start()
            for spec in self._extras:
                stop = self._ctx.Event()
                kwargs = dict(spec.kwargs)
                if spec.pass_queue:
                    kwargs[spec.pass_queue] = self.queue
                if spec.pass_stop_event:
                    kwargs[spec.pass_stop_event] = stop
                proc = self._ctx.Process(target=spec.target, kwargs=kwargs, name=spec.name)
                proc.start()
                self.extra_stops[spec.name] = stop
                self.extra_processes[spec.name] = proc
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        """Stop everything, in order: extras -> pipeline -> physio. Idempotent."""
        if self._stopped or not self._started:
            return
        self._stopped = True
        for spec in reversed(self._extras):
            proc = self.extra_processes.get(spec.name)
            self.exit_codes[spec.name] = _stop_process(proc, self.extra_stops.get(spec.name), spec.stop_timeout_s,
                                                       queue=self.queue, name=spec.name)
        self.exit_codes["pipeline"] = _stop_process(self.pipeline_process, self.pipeline_stop, self._stop_timeout_s,
                                                    queue=self.queue, name="pipeline")
        self.exit_codes["physio"] = _stop_process(self.physio_process, self.physio_stop, 5.0, name="physio")

    def __enter__(self) -> "Session":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def pipeline_alive(self) -> bool:
        return self.pipeline_process is not None and self.pipeline_process.is_alive()

    # ---- consuming events
    def _event_from(self, message) -> ScoreEvent:
        source = "pipeline"
        if isinstance(message, dict):
            src = message.get("source")
            if isinstance(src, str):
                source = src
            elif "kind" not in message and self._extras:
                # the pipeline's own messages always carry "kind"; a kind-less message comes from
                # a legacy extra process (e.g. rs_pca_runtime's scorer) -> attribute it to the first extra
                source = self._extras[0].name
            return ScoreEvent.from_message(message, source=source)
        return ScoreEvent(kind="scorer", source=source, meta={"raw": message})

    def _end_events(self) -> list[ScoreEvent]:
        """Once the pipeline has exited: the trailing end/error event (emitted once)."""
        if self._end_emitted or self.pipeline_process is None or self.pipeline_process.is_alive():
            return []
        self._end_emitted = True
        code = self.pipeline_process.exitcode
        if code not in (0, None):
            return [ScoreEvent(kind="error", source="session", meta={"exit_code": code,
                                                                     "error": f"pipeline exited with code {code}"})]
        return [ScoreEvent(kind="end", source="session", meta={"exit_code": code})]

    def poll(self) -> list[ScoreEvent]:
        """Non-blocking: every event currently queued (what the old displays' drain loops did)."""
        events = [self._event_from(m) for m in _drain(self.queue)]
        events.extend(self._end_events())
        return events

    def scores(self, timeout: float = 0.02, raise_on_error: bool = True) -> Iterator[ScoreEvent]:
        """Yield events as they arrive; waits up to `timeout` s per attempt (so a display
        loop can draw between events). Ends after the "end" event; raises SessionError
        after an "error" event from the session (pipeline exit code != 0) if `raise_on_error`."""
        while True:
            try:
                message = self.queue.get(timeout=timeout)
            except Empty:
                message = None
            if message is not None:
                yield self._event_from(message)
                continue
            for ev in self._end_events():
                yield ev
                if ev.kind == "error" and raise_on_error:
                    raise SessionError(ev.meta.get("error", "pipeline failed"))
                return
            if self._end_emitted:
                return


# --------------------------------------------------------------------------- the real thing
def build_rt_config(cfg: SessionConfig):
    """The pipeline's RTSessionConfig for this session (creates the run dir, as before)."""
    from rt_pipeline import RTSessionConfig

    return RTSessionConfig(
        subject=str(cfg.sub),
        day=str(cfg.day),
        run=str(cfg.run),
        incoming_root=Path(cfg.incoming_root),
        base_data=Path(cfg.base_data),
        decoder_template=Path(cfg.decoder_template) if cfg.decoder_template else None,
        decoder_roi_txt=Path(cfg.decoder_roi_txt) if cfg.decoder_roi_txt else None,
        reference_score_run=cfg.reference_score_run,
        enable_scoring=bool(cfg.enable_scoring),
        enable_original_score=bool(cfg.enable_original_score),
        ap_block=cfg.ap_block,
        pa_block=cfg.pa_block,
    )


def prepare_session(cfg: SessionConfig) -> dict:
    """Everything `start_session` does before starting processes, so a launcher can
    inspect the result (run_dir, settings, rt_cfg) and write its own files first.
    Order matches the old launchers: settings file -> RTSessionConfig -> reference
    stats -> settings payload (+ overrides, physio, qc) -> metadata (+ stages)."""
    from rt_global_settings import load_regressor_settings
    from rt_pipeline import REGRESSOR_SETTINGS, load_reference_score_stats

    if cfg.settings_file:
        REGRESSOR_SETTINGS.update(vars(load_regressor_settings(cfg.settings_file)))
    rt_cfg = build_rt_config(cfg)
    rt_cfg.reference_score_stats = load_reference_score_stats(rt_cfg, rt_cfg.reference_score_run)

    settings = vars(REGRESSOR_SETTINGS).copy()
    settings.update(dict(cfg.settings_overrides))
    physio = cfg.physio or PhysioConfig()
    settings.update(physio.settings())
    if cfg.qc_viewer is not None:
        settings["enable_volume_streamer"] = bool(cfg.qc_viewer)

    run_dir = rt_cfg.rt_work_dir
    physio_config = None
    if physio.listener:
        if not physio.enable:
            raise ValueError("physio.listener requires physio.enable (--biopac-listener requires --biopac-enable)")
        if physio.mode != "file":
            raise ValueError("physio.listener requires physio.mode='file' (--biopac-mode=file)")
        from biopac_rt.biopac_receiver import BiopacReceiverConfig

        biopac_output = settings["biopac_file"] or (run_dir / "biopac_regressors_rx.csv")
        settings["biopac_file"] = biopac_output
        physio_config = BiopacReceiverConfig(
            host=physio.host, port=physio.port, timeout=physio.timeout,
            expected_regressors=PHYS_REG_COUNTS.get(physio.phys_reg, 8),
            handshake_tr=REGRESSOR_SETTINGS.TR if physio.handshake else None,
            subject=str(cfg.sub), day=str(cfg.day), run=str(cfg.run), output_path=biopac_output,
        )

    payload = dict(cfg.metadata)
    if cfg.stages is not None:
        payload["stages"] = cfg.stages.to_metadata()
    if payload:
        merge_session_metadata(run_dir, payload)

    return {"rt_cfg": rt_cfg, "settings": settings, "run_dir": run_dir, "physio_config": physio_config,
            "tr": float(REGRESSOR_SETTINGS.TR), "reference_score_stats": rt_cfg.reference_score_stats}


def start_session(cfg: SessionConfig, prepared: Optional[dict] = None) -> Session:
    """Prepare (unless `prepared` is given from `prepare_session`) and start the processes."""
    p = prepared or prepare_session(cfg)
    session = Session(
        run_dir=p["run_dir"],
        pipeline_target=_pipeline_main,
        pipeline_args=(p["rt_cfg"], p["settings"]),
        physio_target=_physio_main if p["physio_config"] is not None else None,
        physio_args=(p["physio_config"],) if p["physio_config"] is not None else (),
        extra_processes=cfg.extra_processes,
        queue_size=cfg.queue_size,
        stop_timeout_s=cfg.stop_timeout_s,
        rt_cfg=p["rt_cfg"],
        settings=p["settings"],
    )
    session.tr = p["tr"]
    session.reference_score_stats = p["reference_score_stats"]
    return session.start()
