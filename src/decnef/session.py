"""Session API: start the real-time machinery, consume ScoreEvents, stop cleanly.

    from decnef.config import load_settings
    from decnef.session import SessionConfig, start_session

    settings = load_settings("decnef.yaml")
    with start_session(SessionConfig(settings, sub="00086", day="3", run="4")) as session:
        for event in session.scores(timeout=0.02):     # display runs here, in the main process
            if event.kind == "score" and event.reg_ready: ...
            if done: break
    # leaving the block stops every child, in order, on Ctrl-C / exception / pipeline error too

Children (all spawned): the pipeline (`decnef.pipeline.run_pipeline`), the
physio listener when `physio.listener` is on, and any `ProcessSpec` extras
(e.g. a scorer plugin), started after the pipeline and stopped before it.
Messages on the queue are dicts with `kind` and `source`; `ScoreEvent`
is built on the consumer side (docs/session_api.md).
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterator, Optional

from .config import PHYS_REG_COUNTS, Settings
from .events import ScoreEvent
from .layout import RunLayout

log = logging.getLogger(__name__)


class SessionError(RuntimeError):
    """The pipeline (or an extra process) exited with an error."""


@dataclass
class ProcessSpec:
    """An extra process the session starts after the pipeline and stops before it.
    `target(**kwargs)` runs in a spawned child; the session adds the score queue
    under `pass_queue` and a stop Event under `pass_stop_event` (None = omit)."""

    name: str
    target: Callable
    kwargs: dict = field(default_factory=dict)
    pass_queue: Optional[str] = "score_queue"
    pass_stop_event: Optional[str] = "stop_event"
    stop_timeout_s: float = 5.0


@dataclass
class SessionConfig:
    settings: Settings
    sub: str
    day: str
    run: str
    incoming_root: Optional[Path] = None          # overrides settings.paths.incoming_root
    ap_block: Optional[int] = None
    pa_block: Optional[int] = None
    metadata: dict = field(default_factory=dict)  # launcher-owned keys merged into session_metadata.json
    extra_processes: list = field(default_factory=list)
    queue_size: int = 200
    stop_timeout_s: float = 10.0

    @property
    def layout(self) -> RunLayout:
        incoming = self.incoming_root or self.settings.paths.incoming_root
        if incoming is None:
            raise ValueError("incoming_root is required (config paths.incoming_root or SessionConfig.incoming_root)")
        return RunLayout(self.settings.paths.base_data, str(self.sub), str(self.day), str(self.run), Path(incoming),
                         self.ap_block, self.pa_block)


# ----------------------------------------------------------------------------- process targets
def _pipeline_main(settings_dict: dict, layout: RunLayout, score_queue, stop_event) -> None:
    from .config import from_dict
    from .pipeline import run_pipeline

    _configure_child_logging()
    run_pipeline(from_dict(settings_dict), layout, score_queue, stop_event)


def _physio_main(config, stop_event) -> None:
    from .physio import Receiver

    _configure_child_logging()
    receiver = Receiver(config)
    receiver.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        receiver.stop()


def _configure_child_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


# ----------------------------------------------------------------------------- helpers
def _drain(queue) -> list:
    out = []
    if queue is None:
        return out
    try:
        while True:
            out.append(queue.get_nowait())
    except Empty:
        pass
    except (OSError, ValueError):
        pass
    return out


def _stop_process(proc, stop_event, timeout: float, queue=None, name: str = "") -> Optional[int]:
    """Cooperative stop -> terminate -> kill; drains `queue` meanwhile so a child blocked on a full pipe can exit."""
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


# ----------------------------------------------------------------------------- the session
class Session:
    """Owns the child processes and the score queue. `start_session` builds one for
    the real pipeline; tests build one directly with a fake `pipeline_target`."""

    def __init__(self, run_dir: Path, pipeline_target: Callable, pipeline_args: tuple = (), *,
                 physio_target: Optional[Callable] = None, physio_args: tuple = (),
                 extra_processes: Optional[list] = None, queue_size: int = 200, stop_timeout_s: float = 10.0,
                 layout: Optional[RunLayout] = None, settings: Optional[Settings] = None,
                 feeder=None, simulation=None):
        self.run_dir = Path(run_dir)
        self.layout = layout
        self.settings = settings
        self.feeder = feeder                 # simulate.ScannerFeeder (started after the pipeline, stopped first)
        self.simulation = simulation         # simulate.SimPaths when simulating, else None
        self._ctx = mp.get_context("spawn")
        self.queue = self._ctx.Queue(maxsize=int(queue_size))
        self._pipeline_target, self._pipeline_args = pipeline_target, tuple(pipeline_args)
        self._physio_target, self._physio_args = physio_target, tuple(physio_args)
        self._extras = list(extra_processes or [])
        self._stop_timeout_s = float(stop_timeout_s)
        self.pipeline_stop = self._ctx.Event()
        self.pipeline_process = None
        self.physio_stop = self._ctx.Event() if physio_target is not None else None
        self.physio_process = None
        self.extra_stops: dict[str, Any] = {}
        self.extra_processes: dict[str, Any] = {}
        self.exit_codes: dict[str, Optional[int]] = {}
        self._started = self._stopped = self._end_emitted = False

    # ---- lifecycle
    def start(self) -> "Session":
        if self._started:
            return self
        self._started = True
        atexit.register(self.stop)
        try:
            if self._physio_target is not None:
                self.physio_process = self._ctx.Process(target=self._physio_target, args=(*self._physio_args, self.physio_stop), name="physio")
                self.physio_process.start()
            self.pipeline_process = self._ctx.Process(target=self._pipeline_target, args=(*self._pipeline_args, self.queue, self.pipeline_stop),
                                                      name="pipeline")
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
            if self.feeder is not None:
                self.feeder.start()
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        """Feeder -> extras -> pipeline -> physio. Idempotent."""
        if self._stopped or not self._started:
            return
        self._stopped = True
        if self.feeder is not None:
            self.feeder.stop()
        for spec in reversed(self._extras):
            self.exit_codes[spec.name] = _stop_process(self.extra_processes.get(spec.name), self.extra_stops.get(spec.name),
                                                       spec.stop_timeout_s, queue=self.queue, name=spec.name)
        self.exit_codes["pipeline"] = _stop_process(self.pipeline_process, self.pipeline_stop, self._stop_timeout_s, queue=self.queue, name="pipeline")
        self.exit_codes["physio"] = _stop_process(self.physio_process, self.physio_stop, 5.0, name="physio")

    def __enter__(self) -> "Session":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def pipeline_alive(self) -> bool:
        return self.pipeline_process is not None and self.pipeline_process.is_alive()

    # ---- consuming events
    def _end_events(self) -> list:
        if self._end_emitted or self.pipeline_process is None or self.pipeline_process.is_alive():
            return []
        self._end_emitted = True
        code = self.pipeline_process.exitcode
        if code not in (0, None):
            return [ScoreEvent(kind="error", source="session", meta={"exit_code": code, "error": f"pipeline exited with code {code}"})]
        return [ScoreEvent(kind="end", source="session", meta={"exit_code": code})]

    def poll(self) -> list:
        """Non-blocking: every event queued now (plus the end/error event once the pipeline has exited)."""
        events = [ScoreEvent.from_message(m) for m in _drain(self.queue)]
        events.extend(self._end_events())
        return events

    def scores(self, timeout: float = 0.02, raise_on_error: bool = True) -> Iterator[ScoreEvent]:
        """Yield events as they arrive (waiting up to `timeout` s each time, so a display can
        draw in between). Ends after the 'end' event; raises SessionError after a session
        'error' event (pipeline exit code != 0) when `raise_on_error`."""
        while True:
            try:
                message = self.queue.get(timeout=timeout)
            except Empty:
                message = None
            if message is not None:
                yield ScoreEvent.from_message(message)
                continue
            for ev in self._end_events():
                yield ev
                if ev.kind == "error" and raise_on_error:
                    raise SessionError(ev.meta.get("error", "pipeline failed"))
                return
            if self._end_emitted:
                return


# ----------------------------------------------------------------------------- the real thing
def start_session(cfg: SessionConfig, simulate=None) -> Session:
    """Write the launcher's metadata, start physio listener / pipeline / extras.

    `simulate=SimulationConfig(...)` (decnef.simulate) runs the same session against a
    simulation instead of the scanner: level "scores" swaps the pipeline for a fake
    process that emits ScoreEvents at the TR; level "scanner" feeds synthetic (or
    recorded) DICOMs through the real pipeline. Everything else -- metadata, physio,
    extras, shutdown -- is identical, so a launcher's real and dry-run code differ
    only by this argument. Simulation outputs live under `simulate.out_dir`."""
    from . import outputs

    settings = cfg.settings
    layout = cfg.layout if simulate is None else None      # a simulation brings its own paths; no incoming_root needed
    pipeline_target, pipeline_extra, feeder, sim_paths = _pipeline_main, (), None, None
    if simulate is not None:
        from . import simulate as S

        sim_paths = S.resolve_sim_paths(settings, cfg.run, simulate, cfg.ap_block, cfg.pa_block, sub=cfg.sub, day=cfg.day)
        real_settings = settings
        settings = S.simulation_settings(settings, sim_paths, simulate)
        layout = sim_paths.layout
        S.reset_run_dir(sim_paths)
        if simulate.level == "scanner":
            if simulate.from_dicoms is not None:
                S.require_scanner_tools(settings)
                S.stage_subject_copy(real_settings, sim_paths)      # the real tree is read once, never written
                items = S.recorded_items(simulate.from_dicoms, simulate.source_run, int(cfg.run), simulate.n_volumes or None)
            else:
                phantom = S.make_phantom(settings, simulate, sim_paths)
                items = S.phantom_items(phantom, settings, simulate, int(cfg.run))
            feeder = S.ScannerFeeder(items, sim_paths.incoming, S.arrival_tr(settings, simulate))
        else:
            pipeline_target, pipeline_extra = S.scores_pipeline_main, (simulate.to_dict(),)
    layout.create_run_dirs()
    if cfg.metadata:
        outputs.write_metadata(layout.metadata_json, cfg.metadata)
    if simulate is not None:   # provenance: this run was simulated/replayed, and how
        outputs.write_metadata(layout.metadata_json, {"simulation": {"level": simulate.level, **simulate.to_dict()}})
    physio_target = physio_args = None
    ph = settings.physio
    if ph.listener:
        from .physio import ReceiverConfig

        output = Path(ph.file) if ph.file else layout.biopac_regressors_csv
        settings = settings.copy()
        settings.physio.file = output
        physio_target = _physio_main
        physio_args = (ReceiverConfig(host=ph.host, port=ph.port, timeout=ph.timeout, expected_regressors=PHYS_REG_COUNTS[ph.phys_reg],
                                      handshake_tr=settings.pipeline.TR if ph.handshake else None, subject=layout.sub,
                                      day=layout.day, run=layout.run, output_path=output),)
    session = Session(run_dir=layout.run_dir, pipeline_target=pipeline_target, pipeline_args=(settings.to_dict(), layout, *pipeline_extra),
                      physio_target=physio_target, physio_args=physio_args or (), extra_processes=cfg.extra_processes,
                      queue_size=cfg.queue_size, stop_timeout_s=cfg.stop_timeout_s, layout=layout, settings=settings,
                      feeder=feeder, simulation=sim_paths)
    return session.start()
