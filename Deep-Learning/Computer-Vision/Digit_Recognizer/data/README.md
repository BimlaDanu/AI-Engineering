# data/

Both competition CSVs are committed **gzipped** — 8.9 MB and 5.6 MB, against
73 MB and 49 MB. pandas reads them as they are, so a clone has the data with
nothing downloaded and nothing unpacked, which is what makes the notebook run
on Colab and Kaggle straight from the repository.

The uncompressed copies are not committed; they are only faster copies of the
same bytes, and `make data-gz` rebuilds the gzips from them after a download.

## Getting `train.csv`

Either let the Makefile fetch it:

```bash
make data        # downloads only if neither train.csv nor train.csv.gz is here
make data-check  # confirm what is here is usable
make data-gz     # rebuild train.csv.gz after a fresh download
```

`make data` is a no-op on any clone, because the gzip is committed, so it never
needs credentials there.

Downloading for real needs the competition rules accepted on the website and the
Kaggle CLI authenticated. Older releases read `KAGGLE_USERNAME`/`KAGGLE_KEY` or
`~/.kaggle/kaggle.json`; 2.2.x reads neither. It wants `kaggle auth login`, or a
token from <https://www.kaggle.com/settings/api> in `KAGGLE_API_TOKEN` or
`~/.kaggle/access_token`. Use `make data-force` to re-fetch a file that is
already present.

or download `train.csv` by hand from
<https://www.kaggle.com/competitions/digit-recognizer/data> and drop it in
`data/raw/`. Nothing else in the project changes between the two routes.

On Kaggle Notebooks the file is already mounted, read-only, at
`/kaggle/input/digit-recognizer/train.csv`; pass it with `--csv` instead of
copying it.

```bash
make data-check    # report what is in data/raw and whether it is usable
```

## Layout

| path | holds | committed |
|---|---|---|
| `raw/train.csv.gz` | 42,000 labelled rows — everything the project trains and scores on | yes |
| `raw/test.csv.gz` | 28,000 unlabelled rows — the leaderboard set | yes |
| `raw/train.csv`, `raw/test.csv` | the same rows, as Kaggle publishes them | no |
| `processed/` | intermediate files, if a step ever writes one | no |

`test.csv` carries no labels, so accuracy cannot be computed from it and nothing
in `src/` reads it. The project's own test split — the 20% held out of
`train.csv` in `src/data.py` — is what section 6 of the notebook scores. The
competition file is here for a Kaggle submission, which nothing here produces.

Both directories keep a `.gitkeep` so the structure survives a fresh clone.
`load_raw_train` prefers `train.csv` when it is there and falls back to the
gzip, so the two routes differ only in speed.

## What the pipeline does with it

`src/data.py` reads `raw/train.csv`, removes rows that cannot be learned from
(missing values, out-of-range pixels, duplicate images, images whose copies
carry conflicting labels, and constant frames), and only then splits the result
60/20/20 with a fixed seed, so no duplicate can straddle the split.
