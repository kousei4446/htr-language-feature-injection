# 実行ガイド

この README は、この研究を実際に動かすための手順を順番に説明します。  
まずはこの通りに進めれば学習を開始できます。


## 1. `src` に移動

```
cd src
```

## 2. 仮想環境を作る

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```


## 3. 依存ライブラリを入れる

```powershell
pip install -r requirements.txt
```

## 4. PyTorch を入れる

GPU（CUDA 12.8）を使う場合:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 5. データを置く

`config.yaml` の `data_root` に指定したパスに、次の構成で配置します。

```text
data_root/
  train/
    images/
      *.jpg
    labels.txt
  val/
    images/
      *.jpg
    labels.txt
  test/        # 任意
    images/
    labels.txt
```

`labels.txt` の形式:

- 1行 = 1サンプル
- 形式 = `画像ID 文字列`
- 例 = `c04-110-00 some text here`

`画像ID` は拡張子なしでOKです（`.jpg` を自動で探します）。

## 6. `config.yaml` を確認

`train.py` は起動時に `src/config.yaml` を読みます。  
主に以下を確認してください。

- `mode`: `train` か `infer`
- `data_root`: データセットの親ディレクトリ
- `out_dir`: 学習結果の出力先（実行ごとの run フォルダがこの下に作られます）
- `run_name`: run フォルダ名の先頭文字列
- `save_every`: `0` 以外なら、その間隔で `epoch_XXXX.pt` を保存
- `epochs`, `batch_size`, `lr`: 学習パラメータ
- `use_lail`: `true` で LLM を使う、`false` で使わない
- `llm_name`: 使用する LLM モデルID

## 7. 学習を実行

```powershell
python train.py
```

`mode: train` のとき、`out_dir/<run_name>_<timestamp>/` が作成され、
最良モデルは `out_dir/<run_name>_<timestamp>/checkpoints/best.pt` に保存されます。  
互換パスとして `out_dir/best.pt` も更新されます。

## 8. 推論を実行（1画像 / フォルダ）

`config.yaml` を以下のように変更します。

```yaml
mode: infer
ckpt: saved_models/best.pt
# infer_image: path/to/your_image.jpg
infer_dir: path/to/your_images_dir
```
`infer_image` または `infer_dir` のどちらかを指定できます（`infer_image` を指定した場合は単画像推論を優先）。

その後、同じコマンドで実行:

```powershell
python train.py
```

`infer_dir` 指定時は進捗バーを表示し、各画像の予測は標準出力へ逐次表示しません。  
推論結果は JSONL で保存されます（1行目: 集計、2行目以降: 各画像）。  
保存先は `ckpt` で指定したチェックポイントの **1つ上のフォルダ** です。

## 9. LLM を使う場合（`use_lail: true`）

Hugging Face にログインしておきます。

```powershell
huggingface-cli login
huggingface-cli whoami
```

注意:

- 利用するモデル（例: Meta Llama 系）は、Hugging Face 側で利用許可が必要な場合があります。
- 未許可だとダウンロード時に `401` / `403` が出ます。

## 10. 出力物

- 最新 run の参照: `out_dir/latest_run.txt`
- run 設定: `out_dir/<run_name>_<timestamp>/config.resolved.yaml`
- 文字語彙: `out_dir/<run_name>_<timestamp>/vocab.json`
- チェックポイント:
  - `out_dir/<run_name>_<timestamp>/checkpoints/best.pt`
  - `out_dir/<run_name>_<timestamp>/checkpoints/last.pt`
  - `out_dir/<run_name>_<timestamp>/checkpoints/epoch_XXXX.pt`（`save_every > 0` のとき）
- 互換パス（既存運用向け）:
  - `out_dir/best.pt`
  - `out_dir/vocab.json`
- TensorBoard ログ: `out_dir/<run_name>_<timestamp>/tensorboard`

TensorBoard:

```powershell
tensorboard --logdir saved_models
```

## 11. PROJECT_STRUCTURE
### Directory Layout
```text
src/
  train.py               # エントリポイント
  config.yaml            # 実行時設定
  app/
    __init__.py
    main.py              # モード分岐 (train / infer)
    config/
      __init__.py
      settings.py        # 設定読み込みとデフォルト値
    data/
      __init__.py
      dataset.py         # データセットと前処理
    models/
      __init__.py
      trocr_ctc.py       # モデル定義
    training/
      __init__.py
      train_loop.py      # 学習フローと評価
    inference/
      __init__.py
      predictor.py       # 推論フロー
    utils/
      __init__.py
      text.py            # デコード、CER、トークナイザ補助
      llm.py             # LAIL CLM損失の補助
```

### Code Flow
1. `python train.py`
2. `app.main.main()`
3. `args.mode == "train"` -> `app.training.run_train(...)`
4. `args.mode == "infer"` -> `app.inference.run_infer(...)`
