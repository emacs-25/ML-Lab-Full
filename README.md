# ICS1512: Machine Learning Algorithms Laboratory

Nine experiments. Each one lives in its own folder (`Ex1` ... `Ex9`) and is delivered in three equivalent forms:

| File | What it is |
|---|---|
| `experiment.org` | Org-mode report with runnable Python code blocks (Org-Babel), results and inferences. **Source of truth for the PDF.** |
| `experiment.ipynb` | The same code as a Jupyter notebook. |
| `experiment.pdf` | The report exported from the ORG file (via LaTeX). |

Every folder also contains `experiment.tex` (the exported LaTeX), a question PDF (`Experiment_<N>Q.pdf`), the figures, and the dataset it needs.

## Experiments

| Folder | Topic | Data (relative to the folder) |
|---|---|---|
| `Ex1` | NumPy, Pandas, SciPy, scikit-learn, Matplotlib | `data/spambase.data`, `data/loan_train.csv`, scikit-learn built-ins (iris, digits, diabetes) |
| `Ex2` | Spam/ham classification: Naive Bayes and KNN | `data/spambase.data` |
| `Ex3` | Regression: linear and regularized models | `data/loan_train.csv` |
| `Ex4` | Logistic regression and SVM | `data/spambase.data` |
| `Ex5` | Decision tree and random forest | `data/wdbc.data` |
| `Ex6` | Bagging, boosting, stacking (**Experiment 7**) | scikit-learn built-in (breast cancer) |
| `Ex7` | PCA and model evaluation (**Experiment 6**) | scikit-learn built-in (breast cancer) |
| `Ex8` | K-Means, DBSCAN, hierarchical clustering (UCI HAR) | `UCI HAR Dataset/` |
| `Ex9` | Perceptron vs MLP with hyperparameter tuning | `data/english.csv`, `data/Img/` |

> Folders `Ex6` and `Ex7` are swapped relative to the experiment numbers printed in their reports. The folder names were kept as they are.

## 1. Requirements

### System tools

| Tool | Needed for | Notes |
|---|---|---|
| Python 3.10+ | everything | tested with 3.12 |
| Emacs 27+ with Org 9.x | running the ORG files | tested with Emacs 29.3 / Org 9.6 |
| TeX Live (pdfLaTeX) | ORG to PDF | needs `ulem`, `wrapfig`, `capt-of`, `longtable`, `booktabs`, `fancyhdr`, `enumitem`, `listings`, `float`, `array`, `fvextra`, `xurl`, `geometry`, `hyperref`, `graphicx` |
| Jupyter | running the notebooks | `nbconvert` and `ipykernel` |

Install examples:

```bash
# Debian / Ubuntu
sudo apt-get update
sudo apt-get install -y emacs-nox texlive-latex-recommended texlive-latex-extra texlive-plain-generic

# macOS (Homebrew)
brew install --cask emacs
brew install --cask mactex-no-gui        # or: brew install basictex, then add packages with tlmgr
brew install libomp                      # needed by xgboost on macOS

# Windows: install Emacs (gnu.org) and MiKTeX (miktex.org, auto-installs missing packages)
```

### Python packages: `requirements.txt`

Create a file named `requirements.txt` in the repository root with exactly this content:

```text
# requirements.txt
numpy>=1.26
pandas>=2.1
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.8
seaborn>=0.13
pillow>=10.0
xgboost>=2.0
jupyter
nbconvert>=7.0
ipykernel>=6.0
```

The experiments were executed and verified with: Python 3.12.3, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8, seaborn 0.13.2, pillow 12.1.1, xgboost 3.4.1, nbconvert 7.17.1, ipykernel 7.4.0. The lower bounds above are looser and have **not** been tested. If something behaves differently, install the versions listed here.

### Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name ml-lab    # optional: named Jupyter kernel
```

> **Start Emacs from the activated environment** (run `emacs` in the same terminal), otherwise Org-Babel will use the system Python and will not find the packages. See "Using a virtual environment inside Emacs" below if you start Emacs from a launcher.

## 2. Always run from inside the experiment folder

All paths in the code are relative (`data/...`, `UCI HAR Dataset/`, `nbassets/`, `figures/`). Change into the folder first:

```bash
cd Ex9
```

Running from the repository root will fail with "file not found".

## 3. Option A: Jupyter notebook

Interactive:

```bash
cd Ex9
jupyter lab experiment.ipynb          # or: jupyter notebook experiment.ipynb
```
Run **Kernel, Restart & Run All**.

Headless (executes and saves the outputs back into the file):

```bash
cd Ex9
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 experiment.ipynb
```

Use `--output executed.ipynb` instead of `--inplace` if you want to keep the original untouched.

## 4. Option B: Org-Babel, interactively in Emacs

### One-time Emacs setup

Add this to `~/.emacs.d/init.el` (or `~/.emacs`):

```elisp
(require 'org)

;; Languages that code blocks may use
(org-babel-do-load-languages
 'org-babel-load-languages
 '((python . t) (shell . t) (emacs-lisp . t)))

;; Do not ask "Evaluate this code block?" every time
(setq org-confirm-babel-evaluate nil)

;; Python interpreter used by Org-Babel sessions (use "python" on Windows)
(setq python-shell-interpreter "python3"
      org-babel-python-command "python3")

;; LaTeX export: code blocks via listings, three pdflatex passes
(setq org-latex-listings t)
(add-to-list 'org-latex-packages-alist '("" "listings"))
(setq org-latex-pdf-process
      '("pdflatex -interaction=nonstopmode -halt-on-error %f"
        "pdflatex -interaction=nonstopmode -halt-on-error %f"
        "pdflatex -interaction=nonstopmode -halt-on-error %f"))
```

### Run an experiment

1. `cd Ex9 && emacs experiment.org`
2. Run code:
   - **Whole file:** `C-c C-v b` (`org-babel-execute-buffer`). Runs every block in order.
   - **One block:** put the cursor inside it and press `C-c C-c`.
   - Blocks share one Python session per file (`:session exp9`, `exp1`, ... `har`), so run them **top to bottom**; later blocks use variables from earlier ones.
3. Show figures inside Emacs: `C-c C-x C-v` (toggle inline images).
4. Export:
   - LaTeX only: `C-c C-e l l`, which writes `experiment.tex`.
   - **PDF:** `C-c C-e l p`, which writes `experiment.pdf`.
5. Reset a stuck session: kill the session buffer (for example `M-x kill-buffer RET *exp9* RET`) and run again.

### How the code blocks are written

- `#+PROPERTY: header-args:python :session expN :exports both` at the top sets the session and exports both code and results.
- Figure blocks use `:results file` (or `:results value file`) and **return the image path as the last line**, for example `"figures/roc.png"`. Org then inserts `[[file:figures/roc.png]]` as the result. Do not add `:file` to such blocks; it makes Org overwrite the PNG with text.
- Results appear under `#+RESULTS:` after each block.

### Using a virtual environment inside Emacs

Pick one:

- Start Emacs from the activated shell (simplest), or
- set the interpreter explicitly in `init.el`:

  ```elisp
  (setq python-shell-interpreter "/full/path/to/.venv/bin/python"
        org-babel-python-command "/full/path/to/.venv/bin/python")
  ```

## 5. Option C: Org-Babel from the command line (batch)

Save this as `run_org.el` in the repository root:

```elisp
;; run_org.el: execute every block in an ORG file, save the results, export LaTeX.
;; Usage:  emacs --batch -l ../run_org.el experiment.org     (run inside ExN/)
(require 'org)
(setq org-confirm-babel-evaluate nil
      org-latex-listings t
      python-shell-interpreter "python3"
      org-babel-python-command "python3")
(org-babel-do-load-languages
 'org-babel-load-languages '((python . t) (shell . t) (emacs-lisp . t)))
(add-to-list 'org-latex-packages-alist '("" "listings"))
(let ((f (car command-line-args-left)))
  (find-file f)
  (org-babel-execute-buffer)   ; run all blocks, insert #+RESULTS
  (save-buffer)                ; write results back into the .org file
  (org-latex-export-to-latex)) ; produce experiment.tex
```

Run one experiment end to end:

```bash
cd Ex9
emacs --batch -l ../run_org.el experiment.org        # execute + save results + experiment.tex
for i in 1 2 3; do pdflatex -interaction=nonstopmode experiment.tex; done   # PDF (3 passes: TOC and labels)
```

Export only (use the results already stored in the ORG, do not re-run code):

```bash
emacs --batch --eval '(progn (require (quote org)) (setq org-export-babel-evaluate nil org-latex-listings t) (add-to-list (quote org-latex-packages-alist) (quote ("" "listings"))) (find-file "experiment.org") (org-latex-export-to-latex))'
for i in 1 2 3; do pdflatex -interaction=nonstopmode experiment.tex; done
```

## 6. Option D: run everything

Save as `run_all.sh` in the repository root (`chmod +x run_all.sh`):

```bash
#!/usr/bin/env bash
# Usage: ./run_all.sh            run all nine experiments
#        ./run_all.sh Ex2 Ex9    run only the listed ones
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
EXPS=("$@"); [ ${#EXPS[@]} -eq 0 ] && EXPS=(Ex1 Ex2 Ex3 Ex4 Ex5 Ex6 Ex7 Ex8 Ex9)

for ex in "${EXPS[@]}"; do
  echo "=== $ex ==="
  cd "$ROOT/$ex" || continue
  emacs --batch -l "$ROOT/run_org.el" experiment.org   > run_org.log 2>&1 || echo "  ORG run failed, see $ex/run_org.log"
  for i in 1 2 3; do pdflatex -interaction=nonstopmode experiment.tex > pdflatex.log 2>&1; done
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 experiment.ipynb > nbconvert.log 2>&1 || echo "  notebook failed, see $ex/nbconvert.log"
  echo "  pdf warnings: $(grep -cE 'Warning|^! |Overfull|Underfull' pdflatex.log)"
done
```

Approximate run times (ORG, single CPU core): Ex7 about 4 minutes, Ex8 about 6, Ex9 about 10, Ex6 about 12. Ex1-5 vary; allow up to about 15 minutes. Ex9 reads 3,410 images, and Ex6 and Ex9 run grid searches.

## 7. Checking that a run worked

```bash
cd Ex9
grep -c '#+RESULTS' experiment.org           # results blocks inserted
grep -c 'Traceback' experiment.org           # should be 0
grep -E 'Warning|^! |Overfull|Underfull' pdflatex.log   # should print nothing
pdfinfo experiment.pdf | grep Pages
```

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `LaTeX Error: File 'ulem.sty' not found` | Missing TeX packages. Install `texlive-plain-generic` and `texlive-latex-extra` (or use MiKTeX, which installs on demand). |
| `ModuleNotFoundError` in an ORG block | Emacs is using a different Python. Start Emacs from the activated venv or set `python-shell-interpreter` (section 4). |
| Block shows `Evaluate this code block?` prompts | Set `org-confirm-babel-evaluate` to `nil`. |
| `FileNotFoundError: data/...` | You are not inside the experiment folder. `cd ExN` first. |
| PDF build: `libpng error: Not a PNG file` | A figure block used `:file` together with `:results file`, so Org overwrote the PNG. Remove `:file ...` and return the path as the last line instead (section 4). |
| Figure result is an empty `[[file:]]` | Block inherited `:results output`. Use `:results value file`. |
| Variables undefined in a later block | Blocks share a session. Run them in order, or run the whole buffer with `C-c C-v b`. |
| Stale or wrong state after editing code | Kill the session buffer (`*exp9*`, ...) and re-run all blocks. |
| `xgboost` import error on macOS | `brew install libomp`. |
| Overfull/Underfull box warnings after editing text | Long unbreakable lines or narrow table cells. Use the `L{width}` column type (ragged-right) defined in each ORG header. |

## 9. Notes

- Running an ORG file or notebook **overwrites** its stored results and regenerates the figure PNGs. Commit or copy the folder first if you want to keep the current outputs.
- `experiment.executed.org`, where present, is an older duplicate of `experiment.org` and is not needed.
- Experiments 6 and 7 use the built-in scikit-learn breast-cancer data, so no download is required. All other datasets are included in their folders.
