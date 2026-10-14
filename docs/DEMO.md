# Live demo: GloVe in action

A runbook for showing GloVe working end to end: train word vectors on a small slice of Wikipedia, then ask the model live questions ("what's similar to *dog*?", "*man* is to *king* as *woman* is to ?"). It pairs with the "GloVe for Teens" slide deck, but works for any audience.

Every command and output below was run on 2026-10-14 on an Apple M5 MacBook (10 cores, macOS, Python 3.9). Your numbers will differ slightly because training starts from a random seed.

## At a glance

| | |
|---|---|
| Time | ~4 min for the first run here, mostly the 31 MB download; training itself ~45 s |
| Disk | ~2.1 GB while it runs (two 926 MB co-occurrence files), ~185 MB after cleanup |
| Needs | `make`, a C compiler, Python 3 with NumPy, internet for the first run |

## 1. Setup from a fresh clone (once)

```sh
# C compiler + make (macOS). Linux: install gcc and make from your package manager.
xcode-select --install

# Get the code
git clone https://github.com/stanfordnlp/GloVe
cd GloVe

# NumPy for the evaluation and query scripts, in a virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# Check: should print a NumPy version
python -c "import numpy; print(numpy.__version__)"
```

Keep the venv activated for the rest of the demo. Your prompt shows `(.venv)` while it's active. In a new terminal, run `source .venv/bin/activate` again from the repo root. `demo.sh` calls `python` if it exists and falls back to `python3`, so an activated venv makes sure it picks up the one with NumPy. `.venv/` is git-ignored.

If you'd rather not activate it, call the venv's Python directly, for example `.venv/bin/python eval/python/distance.py`.

## 2. Train the model: `./demo.sh`

```sh
./demo.sh
```

The one script runs the whole pipeline. Each stage maps onto a step in the deck:

| Stage | Program | What it does | Output | Time here (— = not timed separately) |
|---|---|---|---|---|
| Build | `make` | Compiles the four C tools into `build/` | `build/*` | — |
| Get data | `curl` / `wget` | Downloads **text8**: the first 100M characters of Wikipedia, lowercased, ~17M words | `text8` (95 MB) | download-bound |
| Count words | `build/vocab_count` | Keeps words seen at least 5 times | `vocab.txt` (71,290 words) | ~1.5 s |
| Count neighbors | `build/cooccur` | Slides a 15-word window over the text and tallies who appears near whom (closer = more weight) | `cooccurrence.bin` (926 MB) | — |
| Shuffle | `build/shuffle` | Randomizes the pair order so training isn't biased by order | `cooccurrence.shuf.bin` (926 MB) | — |
| Train | `build/glove` | 15 passes, learning 50 numbers per word | `vectors.txt`, `vectors.bin` | ~45 s |
| Test | `eval/python/evaluate.py` | Runs ~19,500 analogy puzzles | accuracy printout | — |

**What to point at while it runs:** the `cost` going down on each pass. That's the model getting better at predicting the co-occurrence counts ("practice until the numbers fit"):

```
iter: 001, cost: 0.071318
iter: 005, cost: 0.041403
iter: 010, cost: 0.037567
iter: 015, cost: 0.036205
```

**The score at the end** (this run):

```
capital-common-countries.txt:  ACCURACY TOP1: 63.24% (320/506)
gram6-nationality-adjective.txt: ACCURACY TOP1: 59.24% (901/1521)
family.txt:                    ACCURACY TOP1: 37.62% (158/420)
...
Semantic accuracy:  26.08%  (1934/7416)
Syntactic accuracy: 20.24%  (2107/10411)
Total accuracy:     22.67%  (4041/17827)
```

"TOP1" means the model's *single* best guess had to be exactly right, out of 71,000 possible words. Countries → capitals works well. Grammar puzzles (*good → better*) are harder with this little data.

## 3. Ask the model questions (the fun part)

The repo ships two interactive tools. Both load `vocab.txt` + `vectors.txt` from the current directory in under a second and then wait for input. Type `EXIT` to quit.

### Nearest neighbors: "what words are like this one?"

```sh
python eval/python/distance.py
# Enter word or sentence (EXIT to break): dog
```

It prints the 100 closest words by cosine similarity (1.0 = same direction). To show only the top few and not flood the screen, pipe in the question:

```sh
printf 'dog\nEXIT\n' | python eval/python/distance.py | grep -E '^ +[a-z]+\s+0\.' | head -6
```

Real results from this run:

| Ask | Top neighbors (similarity) |
|---|---|
| `france` | spain 0.84, italy 0.83, germany 0.80, netherlands 0.79, belgium 0.77 |
| `computer` | computers 0.81, hardware 0.77, systems 0.77, technology 0.77 |
| `dog` | cat 0.73, horse 0.72, bonzo 0.71, dogs 0.71, sled 0.68 |
| `pizza` | fashions 0.61, volleyball 0.61, fisherman 0.59 … (nonsense) |

Entering several words (`king queen`) sums their vectors and finds words near the blend.

### Word math: analogies

```sh
python eval/python/word_analogy.py
# Enter three words (EXIT to break): man king woman
```

The input order is **a b c**, meaning "*a* is to *b* as *c* is to ?". The tool computes `b − a + c` and lists the closest words.

| Ask | Top guesses (this run) |
|---|---|
| `man king woman` | son, elizabeth, **queen** (#3), prince, wife |
| `france paris italy` | venice, leipzig, **rome** (#3), bologna, vienna |
| `good better bad` | quickly, significantly, enough, too, **worse** (#5) |

### Turn the misses into the lesson

The misses are worth showing on purpose:

- **`pizza` gives junk** because it appears only **89 times** in text8, versus 958 for `dog` and 1,940 for `queen` (check with `grep -n '^pizza ' vocab.txt`). With so little "company" to learn from, the vectors stay vague. *More text means better meaning.*
- **The answer is close but not first** (queen #3, rome #3): the model learned the right *direction*, but 17M words is a tiny training set.
- **Then compare with the real thing.** Stanford's pre-trained vectors were trained on 6 billion to 840 billion tokens, hundreds to tens of thousands of times more text. With those, `king − man + woman` lands on `queen` reliably (see step 5).

## 4. Running it again

On later runs `demo.sh` skips the download because `text8` exists, so it only rebuilds and retrains. To only replay the queries, skip training entirely and just run the two Python tools, since `vocab.txt` and `vectors.txt` are already there.

## 5. Optional: use pre-trained vectors

*Not run as part of this write-up.* It's an 822 MB download.

```sh
curl -fL -O https://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip glove.6B.50d.txt            # 400K words × 50 numbers
cut -d' ' -f1 glove.6B.50d.txt > vocab.6B.txt  # the tools need a word list in the same order
python eval/python/word_analogy.py --vocab_file vocab.6B.txt --vectors_file glove.6B.50d.txt
```

More vector sets (2024 Dolma / Wikipedia+Gigaword, Common Crawl, Twitter) are listed in the [README](../README.md).

## 6. Cleanup

The demo outputs are git-ignored, but they take up space:

```sh
rm -f cooccurrence.bin cooccurrence.shuf.bin   # 1.85 GB, only needed for training
rm -f text8 vocab.txt vectors.txt vectors.bin  # everything else
make clean                                     # removes build/
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `unzip: End-of-central-directory signature not found` and a ~800-byte `text8.zip` | The download saved an HTTP redirect page, not the zip. Fixed in `demo.sh` (it now uses `https://` and `curl -fL`). Delete `text8.zip` and rerun. |
| `python: command not found` | The venv isn't active. macOS ships only `python3`. Run `source .venv/bin/activate` from the repo root, or create the venv first (step 1). |
| `ModuleNotFoundError: No module named 'numpy'` at the evaluation step | Activate the venv from step 1 (`source .venv/bin/activate`). Training already finished, so you can just run `python eval/python/evaluate.py`. |
| `Word: xyz  Out of dictionary!` | text8 is all lowercase, has no punctuation, and spells out digits (`1999` → `one nine nine nine`). Words seen fewer than 5 times are dropped. |
| `Only 2 words were entered..` | `word_analogy.py` needs exactly three words: `a b c`. |
| Hundreds of lines of output | The tools always print the top 100. Pipe through `head` as shown in step 3. |
