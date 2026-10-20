# Migrated lab settings

The lab's `*_settings.json` files translated into the `decnef` config layout
with `decnef config migrate` (`docs/config.md`). They are study configuration,
not part of the package; they live here until the study-specific material
moves to its own repository.

Values that were implicit before are now explicit in each file:

- `paths.base_data: /SSD2/DecNef_py/data` (was the `--base-data` default)
- `decoder.template: /SSD2/DecNef_py/decoders/rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii` (was the silent fallback)
- `paths.incoming_root` is not set: pass `--incoming` on the command line (it changes per session)

`minjoo.yaml`: the old file's `volume_stream_window_x/y` had no effect (the
old loader ignored unknown keys) and were dropped.

Where the files differ from each other (everything else is identical):

| key | rt | jane | kostya | minjoo | sara |
|---|---|---|---|---|---|
| `physio.enable` | true | false | false | false | false |
| `pipeline.TR` | 1.0 | 1.4 | 1.0 | 1.0 | 1.0 |
| `pipeline.analysis_space` | mni | t1 | t1 | t1 | t1 |
| `pipeline.engine.max_workers` | 6 | 1 | 2 | 1 | 2 |
| `pipeline.truncate_t1_to_epi_fov` | false | false | true | false | true |
| `qc.volume_viewer` | false | false | false | true | false |
| `qc.window_title` | RT volume QC | RT volume QC | RT volume QC | BrainStream QC | RT volume QC |

Usage:

```bash
decnef run --config configs/jane.yaml --sub 00086 --day 3 --run 4 --incoming /path/to/dicoms
python examples/nf_events_launcher.py --config configs/jane.yaml --sub 00086 --day 3 --run 4 --incoming /path/to/dicoms
```

Note `rt.yaml` is the only one with `physio.enable: true`; the launchers used
to override that from their `--biopac-enable` flag (default off), so a run
with `rt.yaml` now needs `--set physio.enable=false` unless BIOPAC is really
connected.
