# Submission checklist — PCOMPBIOL-D-26-00465 (revision)

## Files in this folder
- `manuscript_revised.tex` — revised manuscript (LaTeX source; upload as "LaTeX Source File").
- `manuscript_revised_draftfigures.pdf` — compiled preview (placeholder figure boxes; compile on Overleaf with real figures for the final "Manuscript" PDF).
- `manuscript_original_backup.tex` — as-submitted version, for generating tracked changes.
- `Response_to_Reviewers.docx` / `.md` — point-by-point response (upload as "Response to Reviewers").
- `CHANGES.docx` / `.md` — itemized change log.
- `ref.bib`, `Figures/` (two new supplementary figures).

Repo additions: `README.md`, `requirements.txt`, `examples/` (two MWEs),
`revision_experiments/` (exp 2a/2b/2c with READMEs).

## Done (addresses every reviewer + editor point)
- [x] Major 1a reproducibility: README + 2 minimal working examples + documented experiment scripts.
- [x] Major 1b: fitting algorithm box + "Fitting and training procedure" methods subsection.
- [x] Major 1c: f_damp promoted to main text with basis-function parameterization + Fig ref.
- [x] Minor 1a-1e: methods content moved to Supplementary Results; supplement split into
      Results vs Figures/Tables; damping analyses grouped; Statistical testing section added.
- [x] Other 2a: new Scenario-3 simulation (coupling + variability) + supplementary figure.
- [x] Other 2b: gain stratification on REAL data (session 757216464) + synthetic control + Discussion paragraph + figure.
- [x] Other 2c: quantitative S6/S7 comparison on REAL data (S6 r=0.96, S7 r=0.998) + tool + figure.
- [x] Other 2d: Discussion generalizability paragraph.
- [x] Writing 3.1/3.2/3.3: intro equations, population sentence, BIC definition.
- [x] Editor 2,3,8,9: LaTeX source, 179-word Author Summary, Financial Disclosure, Competing Interests.
- [x] Editor 4,6: Ethics statement added; supplement separated into Supporting Information.
- [x] Manuscript compiles cleanly (42 pp, 0 undefined refs, 0 errors).

## Still needs you (author-only, before clicking submit)
- [ ] Editor 1: enter CRediT contributions for all 4 authors in Editorial Manager.
- [ ] Editor 4: confirm/insert the Allen Institute IACUC protocol number.
- [ ] Editor 5: export main figures to TIFF/EPS via the PLOS NAAS tool.
- [ ] Editor 7: confirm Figure 1B clip-art provenance (or swap for open-source icons).
- [ ] Editor 10: confirm final GitHub release/commit or data DOI in the Data Availability statement.
- [ ] Generate the tracked-changes file:
      `latexdiff manuscript_original_backup.tex manuscript_revised.tex > manuscript_trackchanges.tex`
      then compile (Overleaf "track changes" does this in one click).

## Note on the local data cache
The Allen session NWB `D:\ecephys_cache_dir\session_757216464\session_757216464.nwb` was
re-fetched during this work (allensdk's integrity check triggered a re-download); it has been
fully restored to its original size (3,039,748,856 bytes) and verified as a valid file. A small,
backward-compatible change was made to `DataLoader.py` (an optional `ALLEN_MANIFEST_PATH`
environment variable) so the cache path can be set without editing the host-name logic.
