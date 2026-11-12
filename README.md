# Credit Card Fraud Classification

An educational classification project using a class-weighted logistic regression model trained on the ULB Credit Card Fraud Detection dataset. It includes a browser dashboard for scoring transactions from the dataset feature schema.

## Train

Place the extracted CSV at `data/creditcard.csv`, then run `python -m pip install -r requirements.txt` and `python ml/train.py`. Training creates `model.json` with a stratified holdout evaluation and coefficients used by the web dashboard.

## Preview

Run `python -m http.server 8000` from this folder and open http://localhost:8000. Import this repository in Vercel as a static site; no build command is needed.

The 151 MB dataset is excluded from Git. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and review its terms before redistribution. The model is for learning and is not suitable for real payment decisions.
