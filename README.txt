PP1 — Naive Bayes for Text Categorization (Sentiment)

Implements the assignment requirements:
- Multinomial Naive Bayes for sentiment with ML (m=0) and MAP/Dirichlet smoothing (m>0)
- Prediction in log-space (to avoid underflow)
- Skip test tokens not present in the training vocabulary (as required)
- 10-fold stratified cross-validation
- Learning curves: training fractions 0.1..0.9 for m=0 and m=1
- Smoothing sweep: m = 0,0.1,...,0.9 and 1,2,...,10 (20 values)
- Optional extra credit preprocessing experiments (only when --extra_credit is used)

Input format:
- Each dataset file contains one example per line:
    <sentence>\t<label>
  where <label> is 0 (negative) or 1 (positive).

Tokenizer policy:
- Main assignment tokenizer is verbatim whitespace tokenization only:
    sentence.strip().split()
  (No lowercasing, punctuation removal, stemming, etc.)
- IMPORTANT scoring rule: tokens not in the training vocabulary V (never seen in training) are skipped.
  For m=0, a token that is in V but has zero count in a class has probability 0 in that class, i.e., log(0) = -inf.
  This is expected behavior and is exactly why smoothing improves performance.

Dependencies:
- Python 3.x
- numpy and matplotlib (no sklearn, pandas, etc.)

Directory structure:
- Run code from the pp1 directory.
- Data files should be in a subdirectory (default: pldata/):
    pldata/amazon_cells_labelled.txt
    pldata/imdb_labelled.txt
    pldata/yelp_labelled.txt

How to run (main experiments):
  cd pp1
  python3 main.py --data_dir pldata --out_dir out --seed 0

Main outputs:
  out/amazon_learning_curves.png
  out/imdb_learning_curves.png
  out/yelp_learning_curves.png
  out/amazon_smoothing_sweep.png
  out/imdb_smoothing_sweep.png
  out/yelp_smoothing_sweep.png

Also saved (numeric arrays used for plots):
  out/amazon_learning_curves.npz
  out/amazon_smoothing_sweep.npz
  out/imdb_learning_curves.npz
  out/imdb_smoothing_sweep.npz
  out/yelp_learning_curves.npz
  out/yelp_smoothing_sweep.npz

Extra credit:
  python3 main.py --data_dir pldata --out_dir out --seed 0 --extra_credit

Extra credit outputs:
  out/extra_credit/extra_*.png
  out/extra_credit/*_extra_credit.tsv

Notes:
- All results are reproducible given the seed (default --seed 0).