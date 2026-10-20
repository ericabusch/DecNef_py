"""decnef: real-time fMRI decoded neurofeedback.

    decnef.config    Settings (one documented config file; no hidden defaults for paths/decoders)
    decnef.layout    RunLayout: every path of a subject/day/run
    decnef.session   start_session / Session / ScoreEvent: run the machinery, consume scores
    decnef.pipeline  the per-volume processing (unwarp -> MC -> QC -> regression -> transform -> smooth -> score)
"""
__version__ = "2.0.0a0"
