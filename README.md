# Hotel review sentiment classification

Multiclass model that predicts a hotel review's star score (1–5) from its title and body. The problem comes from real industry work on guest reviews: each review has a translated title, a translated text, and a numeric score.

## Problem

Hotel reviews are short, noisy, and unevenly distributed across scores. This project classifies the sentiment of those reviews by predicting the score from the written title and text. The raw export stays out of the repository.

## Approach

The pipeline in `main.ipynb` is a classical bag-of-words classifier.

### Preprocessing

Reviews are loaded from `reviews.json`. `translation_title` and `translation_text` are concatenated into `combined_text`:

- When both fields are missing, the result is an empty string.
- When one field is missing or blank, the other field is kept.
- Otherwise the title and the text are joined with a single space.

Rows whose `combined_text` is empty are dropped. Scores outside {1, 2, 3, 4, 5} are mapped with `numpy.ceil`. Leading and trailing whitespace is stripped from `combined_text`.

The rows are split with `train_test_split` (`test_size=0.2`, `random_state=42`). The label is `score`.

### Features

`CountVectorizer` (scikit-learn defaults) is fit on the training texts and then applied to the test texts. Each review becomes a sparse vector of token counts.

### Model

`LogisticRegression(max_iter=1500)` is fit on the vectorized training set. The notebook sets no other hyperparameters. The saved run used the `lbfgs` solver and stopped when it reached the iteration limit (`ConvergenceWarning`).

## Results

Saved evaluation in `main.ipynb`:

| Split | Accuracy |
| --- | ---: |
| Training | 91.68% |
| Testing | 87.97% |

Test classification report (74,084 reviews):

| Score | Precision | Recall | F1 | Support |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.85 | 0.84 | 0.84 | 1,462 |
| 2 | 0.82 | 0.81 | 0.81 | 2,176 |
| 3 | 0.84 | 0.81 | 0.82 | 6,605 |
| 4 | 0.84 | 0.88 | 0.86 | 27,541 |
| 5 | 0.92 | 0.90 | 0.91 | 36,300 |

Scores 4 and 5 make up most of the test set. Macro-averaged F1 is 0.85. Weighted-average F1 is 0.88.

## How to run

Keep the review export on your machine. `.gitignore` lists `reviews.csv` and `reviews.json`, so neither file is part of this repository.

`main.ipynb` loads `reviews.json` from the project root. Expected columns:

| Column | Role |
| --- | --- |
| `translation_title` | Review title. May be null or blank. |
| `translation_text` | Review body. May be null or blank. |
| `score` | Numeric rating. Values outside 1–5 are ceiled into that range. |

```bash
pip install -r requirements.txt
```

Then run `main.ipynb` from top to bottom. The notebook imports numpy, pandas, and scikit-learn. `requirements.txt` also pins tensorflow.

## Next steps

- Add a positivity or negativity score from BERT, an LLM, or a similar model, and use it as a feature.
- Engineer more features from the text.
- Plot tokens against classes and look for clusters or other relationships.
- Use mutual information to see which tokens help and which do not.
- Review the available plot types and keep the ones that are useful here.
- Try clustering or PCA as a source of new features.
- Apply PCA on its own.
- Try target encoding.
- Plot training loss against validation loss across training.
- Try other deep-learning architectures, including dropout and batch normalization.
- Study activation functions and how they apply to this problem.
- Augment the data with paraphrases that keep the same meaning, so the model depends less on exact wording.
- Try convolutional layers to pick up local sentiment patterns.
- Try hybrid models: boosted hybrids and stacked hybrids.
