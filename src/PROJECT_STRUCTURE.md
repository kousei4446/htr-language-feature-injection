# Project Structure

## Directory Layout
```text
src/
  train.py               # thin entry point
  config.yaml            # runtime settings
  app/
    __init__.py
    main.py              # mode dispatcher (train / infer)
    config/
      __init__.py
      settings.py        # config loader and defaults
    data/
      __init__.py
      dataset.py         # dataset and preprocessing
    models/
      __init__.py
      trocr_ctc.py       # model definition
    training/
      __init__.py
      train_loop.py      # training flow and evaluation
    inference/
      __init__.py
      predictor.py       # inference flow
    utils/
      __init__.py
      text.py            # decode, CER, tokenizer helpers
      llm.py             # LAIL CLM loss helper
```

## Code Flow
1. `python train.py`
2. `app.main.main()`
3. `args.mode == "train"` -> `app.training.run_train(...)`
4. `args.mode == "infer"` -> `app.inference.run_infer(...)`
