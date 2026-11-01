# llm-jp-logkv

`llm-jp/llm-jp-4.1-8b-thinking` の学習済み Llama を出発点に、Attention を
LogKV に置換し、蒸留・追加学習による適応を検証するプロジェクトです。

現在は環境と参照実装の準備段階です。LogKV コア、元の回帰テスト、
保存先設定、モデル取得コマンドを用意しています。GQA/RoPE を扱う独立APIの最小 Attention アダプターを追加しました。
Transformers の生成APIへの組み込み、モデル全体の置換、蒸留トレーナーは未実装です。

## 環境

- Python 3.12 / uv
- PyTorch 2.11.0 (CUDA 12.8)、Transformers 5.4.0
- Accelerate / PEFT / datasets / Hugging Face Hub
- 依存関係の確定版は `uv.lock` を参照してください。

```bash
cd /home/myxy/llm-jp-logkv
# 新しいチェックアウトでは .env.example を .env にコピーし、パスを編集する。
# このマシンの初期セットアップでは .env は作成済み。
./scripts/uv.sh sync --locked
./scripts/uv.sh run --locked llm-jp-logkv prepare-storage
./scripts/uv.sh run --locked llm-jp-logkv doctor
```

`scripts/uv.sh` はプロジェクト直下へ移動し、ローカルの `.env` を読み込んで
`uv` を実行する薄いラッパーです。`uv sync` にも同じ保存先を適用します。
`.env` はシェルからも読み込むので、信頼できるローカル設定として扱ってください。
Python の CLI も HF ライブラリの import より前に `.env` を読み込みます。
Python 側も `.env` に明記された値を優先します。ログインシェルに別プロジェクト用の
`HF_HOME` があっても、このプロジェクトのキャッシュ先へ揃えます。

通常の `uv run` を使う場合も、環境の選択前に `.env` を渡してください。

```bash
uv run --env-file .env --locked llm-jp-logkv doctor
```

uv 自体は既存の環境変数を優先するため、別の `UV_PROJECT_ENVIRONMENT` や
`UV_CACHE_DIR` を export している場合は `scripts/uv.sh` を使ってください。

`.venv` はこのマシンでは `${UV_PROJECT_ENVIRONMENT}` へのシンボリックリンクです。
VS Code の Python interpreter は `.venv/bin/python` を指定できます。
別マシンでは `.env` を編集して同期後、必要ならその仮想環境への `.venv` リンクを作成します。

## 大きなファイルの保存先

既定の `DATA_DIR` は `/mnt/raid0/llm-jp-logkv` です。

| 環境変数 | 用途・既定の相対パス |
|---|---|
| `MODEL_DIR` | モデルの固定 revision ごとの取得先: `models/` |
| `DATASET_DIR` | データ・トークン化結果: `datasets/` |
| `CHECKPOINT_DIR` | 新規学習チェックポイント: `checkpoints/` |
| `EXPERIMENT_DIR` | 蒸留用中間状態・実験出力: `experiments/` |
| `LOG_DIR` | 実行ログ: `logs/` |
| `HF_HOME` | HF 設定・キャッシュ: `cache/huggingface/` |
| `HF_HUB_CACHE` | Hub キャッシュ: `${HF_HOME}/hub` |
| `HF_DATASETS_CACHE` | datasets キャッシュ: `${HF_HOME}/datasets` |
| `TORCH_HOME` | torch キャッシュ: `cache/torch/` |
| `UV_CACHE_DIR` | パッケージキャッシュ: `cache/uv/` |
| `UV_PROJECT_ENVIRONMENT` | Python 仮想環境: `venvs/py312/` |

`LOGKV_SOURCE_DIR=/home/myxy/RecursiveCompressor` は移植元を示す参照用設定です。
実行時にこのディレクトリへ依存する import や `PYTHONPATH` 追加は行いません。

`.env`、仮想環境、重み、データセットは Git 管理外です。
`DATA_DIR` 未設定や相対パスはエラーにして、ソースツリーへの暗黙の書き込みを防ぎます。
実験結果は RAID に保存し、小さい条件表・評価要約だけ `doc/` に記録してください。

## 元モデルの取得

`.env` の `BASE_MODEL_ID` と `BASE_MODEL_REVISION` を使います。
モデル revision は `7c19bc90e49b717381e4f474ee95964c05217a28` に固定しています。

```bash
# config・tokenizer・chat template・モデルカードのみ。重みは取得しない。
./scripts/uv.sh run --locked llm-jp-logkv download-model

# 実モデルでの実験を開始するときに BF16 の全重みも取得する。
./scripts/uv.sh run --locked llm-jp-logkv download-model --weights
```

保存先は `${MODEL_DIR}/llm-jp--llm-jp-4.1-8b-thinking/${BASE_MODEL_REVISION}/`。
HF の download metadata もこの保存先内に作られます。取得モードと revision は
同じディレクトリの `download_manifest.json` に記録します。
このコマンドは重みをロードしたり学習を開始したりしません。

llm-jp の tokenizer / chat template を維持します。元プロジェクトの
`[INST]...[/INST]` 形式やトークン化済みデータは、そのまま流用できません。
特に語彙数は 196,608 のため、元プロジェクトの uint16 トークン保存形式は使えません。
新しいトークンキャッシュを実装する際は uint32 等を使用してください。

## 検証

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  ./scripts/uv.sh run --locked pytest -q
```

移植元の独立した Attention 参照、forward/step 一致、階層の被覆、
grouped forward の勾配検証を含みます。さらに小さいランダム初期化 Llama により
GQA・生成キャッシュ・LoRA の勾配を CPU で確認します。8B モデルの品質評価ではありません。

## 配置と次の作業

- `src/llm_jp_logkv/logkv/attention.py`: LogKV コアの無改変コピー
- `UPSTREAM.json`: 移植元 commit・ファイルの SHA-256・テストの import 調整記録
- `src/llm_jp_logkv/settings.py`: `.env` と保存先
- `src/llm_jp_logkv/cli.py`: 保存先作成・環境確認・モデル取得
- `tests/logkv/`: 移植元由来の回帰テスト
- `src/llm_jp_logkv/llama_attention.py`: 投影流用・RoPE後のdense/LogKV独立API
- [アダプター仕様と検証](doc/llama-adapter-20261101.md)
- [引き継ぎと実装計画](doc/handoff-20261101.md)
- [初期環境の検証結果（174 tests passed）](doc/setup-validation-20261101.md)

コードは MIT（移植元の LICENSE を保持）。元モデルのライセンスはモデルカードを参照してください。
