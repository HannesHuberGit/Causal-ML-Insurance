# Notes for Claude Code

- Run every script from the repo root; paths are relative to it (e.g. `final/layers`, `out/mc_conv.json`).
- Colours: ink #14213D = truth (dashed); orange #EB6834 = LightGBM; blue #2A78D6 = Double ML (text #1C5CAB); aqua #18A070 (text #0F7F5A) = lapse model ℓ; violet #4A3AA7 = price model m.
- Font: IBM Plex Sans (fonts/); formulas: STIX mathtext via final/formulas_final.py (fixed 56 px height, common baseline).
- Charts: final/charts_final.py draws each chart once on a fixed axes rect and saves one transparent PNG per build layer (`<chart>_L<k>.png`), stacked in the slide at the same position with `data-build-in="fade k"`.
- Slides: deck/project/slides/*.html (1920×1080, inline styles, absolutely positioned build elements). Speaker notes in `<aside>`.
- Images in slides are `/_blob/<id>` assets of the live Claude Slides artifact; render/blobmap.json maps ids to local files for render/render.py.
- The user values correctness: every number on a slide or in the notes must come from the simulation; state rates as typical orders, with measured values in the notes.
