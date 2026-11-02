# llm-jp-logkv

`llm-jp/llm-jp-4.1-8b-thinking` の学習済み Llama を出発点に、Attention を
LogKV に置換し、蒸留・追加学習による適応を検証するプロジェクトです。

元Llamaの全32層でQ/K/V/O、RoPE、FFN、RMSNormを流用し、Attentionのみ
LogKVに変更する独立推論APIとCLI、および固定減衰LogKVのLoRA蒸留を実装しています。
TransformersのCache/model.generateへの組み込みは未対応で、独自のgreedy生成を使用します。
初期評価では元モデルが4問とも正答した一方、未蒸留LogKVの両条件は反復出力となりました。
[実重みでの初期品質・性能結果](doc/initial-quality-20261102.md)を参照してください。
蒸留の設定・データ・再現手順は[蒸留実験記録](doc/distillation-20261102.md)にまとめています。
全6 GPU・8時間の蒸留を完了し、同条件の手書き4問は未蒸留0/4から2/4に改善しました。
元モデルの4/4には届かず、推論品質の保持は確認できていません。

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
同じディレクトリの日時・ID付き `download_manifest-*.json` に追記します。
初期取得時の `download_manifest.json` も保持します。
このコマンドは重みをロードしたり学習を開始したりしません。

llm-jp の tokenizer / chat template を維持します。元プロジェクトの
`[INST]...[/INST]` 形式やトークン化済みデータは、そのまま流用できません。
特に語彙数は 196,608 のため、元プロジェクトの uint16 トークン保存形式は使えません。
新しいトークンキャッシュを実装する際は uint32 等を使用してください。

## 蒸留なしの初期推論

```bash
# 固定revisionの全重みが必要（初回のみ）
./scripts/uv.sh run --locked llm-jp-logkv download-model --weights

# 1 GPU、全32層LogKV、C=4、固定レベル減衰、BF16
CUDA_VISIBLE_DEVICES=0 ./scripts/uv.sh run --locked llm-jp-logkv infer \
  --prompt '日本の首都はどこですか。短く答えてください。' --max-new-tokens 512

# 元モデル / LogKV固定減衰 / LogKV減衰なしの小規模比較
CUDA_VISIBLE_DEVICES=0 ./scripts/uv.sh run --locked llm-jp-logkv evaluate-initial
```

`infer --mode original` は元モデルのHF生成、`--no-level-bias` はLogKVの減衰なし。
元tokenizerとHarmony chat templateを使い、reasoning_effort=low、greedy、EOS=[2,13]。
`--reasoning-effort`、`--max-new-tokens`、`--prefill-chunk-size` を変更可能です。
元templateを通じてanalysis/finalのチャネルを生成し、生の特殊トークン付き出力も保存します。
初期生成でanalysisのみのまま上限に達した場合、最終回答が完了したとは扱いません。

重みをローカルからロードし、全層を単一deviceへ配置します。paddingなし・1プロンプト、
連続した位置での推論に対応します。量子化・学習・FFN/RMSNormの変更はありません。
圧縮状態は32 Qヘッドで保持し、8 KVヘッド共有とは異なります。

出力は `${EXPERIMENT_DIR}/inference-*` または `initial-quality-*` の新規ディレクトリへ保存。
manifestにソースcommit、model revision、依存関係、設定、入力suiteを記録します。
固定文章のNLL/PPLと生成例は小さな手書きsmoke検証で、標準ベンチマークではありません。
性能値は生成全体の時間と割当ピークの参考値です。速度比較に特化した測定ではありません。

## 検証

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  ./scripts/uv.sh run --locked pytest -q
```

移植元の独立した Attention 参照、forward/step 一致、階層の被覆、
grouped forward の勾配検証を含みます。さらに小さいランダム初期化 Llama により
GQA・生成キャッシュ・LoRA の勾配を CPU で確認します。8B モデルの品質評価ではありません。

## 固定減衰LogKVの蒸留

`scripts/run_distillation.py` はtorchrunによるDDP訓練を管理し、終了コードを保存します。
教師の全重みと生徒の非Attention部分を固定し、全32層のQ/K/V/Oに追加したLoRAを
全語彙KL + next-token CEで学習します。BF16、層checkpoint、語彙損失の分割を用い、
一定stepごとに生成全文・token IDs・検証損失・再開可能なcheckpointを記録します。

ユーザー承認の全6 GPU・8時間試行は3,126 stepsで正常終了しました。詳細な結果・起動・再開手順と
データrevisionは[蒸留実験記録](doc/distillation-20261102.md)を参照してください。
蒸留後は `infer --adapter CHECKPOINT_DIR` と `evaluate-initial --adapter CHECKPOINT_DIR`
で評価できます。通常の `infer` は引き続き未蒸留モデルを使用します。

手動で新しく開始する例（2 GPU、global batch=24、時間制限なしで1 epoch）。run名は毎回変えてください。

```bash
cd /home/myxy/llm-jp-logkv
distill_run="manual-$(date +%Y%m%d-%H%M%S)-$$"
distill_root=/mnt/raid0/llm-jp-logkv
./scripts/uv.sh run --locked python scripts/run_distillation.py \
  --gpus 0,1 --log-dir "$distill_root/logs/$distill_run" --evaluate-final -- \
  --data "$distill_root/datasets/distill-cache-20261101T155341Z-4326ce39" \
  --run-dir "$distill_root/experiments/$distill_run" \
  --batch-size 1 --grad-accum 12 \
  --sample-every 200 --save-every 200 --validate-every 200
```

`--max-hours` の省略時は時間無制限、`--num-epochs` の既定値は1です。
複数epochは `--num-epochs 3`、任意の時間上限は `--max-hours 8` のように指定します。
epoch数・時間・step数を複数指定した場合は、最初に達した上限で保存して終了します。
epoch末尾の勾配蓄積が指定回数に満たなくても、残りのミニバッチを更新してから終了します。
学習率は既定ではepochの進捗に応じて減衰し、経過時間には依存しません。

6 GPUにする場合は `--gpus 0,1,2,3,4,5 --grad-accum 4`。
前回のようにepoch数を制限せず8時間で区切る場合は、さらに `--num-epochs 0 --max-hours 8` を指定します。
ログは `logs/$distill_run/training.log` に出ます。切断後も動かす場合はtmux等から起動します。

別ターミナルから以下を書き込むと、現在の処理が終わったstep境界で全rankが同期し、
重み・optimizer・RNG・データ位置を保存して正常終了します。

```bash
cd /home/myxy/llm-jp-logkv
printf 'save_and_exit\n' > control.cmd
```

`control.cmd` は受理時に削除され、中断時の終了後評価・生成は省きます。
複数runでは起動時の `--` より後に `--control-file /絶対パス/control.cmd` を指定し、
runごとに別のファイルを使ってください。
再開は同じGPU台数・訓練設定と新しいrun/log名で `--resume CHECKPOINT_DIR` を追加します。
`--num-epochs` は再開前を含めた通算epoch数です。再開時にepoch数・時間上限を変更できます。
旧版の8時間上限付きcheckpointも、`--max-hours` を省けば時間上限を外して残りを再開できます。
1 epoch以上を完了済みなら、追加学習には `--num-epochs 2` 等の大きい通算上限を指定してください。
`--max-hours` を明示した場合は、元の開始時刻から停止中の時間も含めて計算します。
保存先・再開条件の詳細は[停止・再開手順](doc/distillation-20261102.md#実行と再開)を参照してください。

## 配置と次の作業

- `src/llm_jp_logkv/logkv/attention.py`: LogKV コアの無改変コピー
- `UPSTREAM.json`: 移植元 commit・ファイルの SHA-256・テストの import 調整記録
- `src/llm_jp_logkv/settings.py`: `.env` と保存先
- `src/llm_jp_logkv/cli.py`: 保存先作成・環境確認・モデル取得
- `tests/logkv/`: 移植元由来の回帰テスト
- `src/llm_jp_logkv/llama_attention.py`: 投影流用・RoPE後のdense/LogKV独立API
- `src/llm_jp_logkv/distillation.py`: 教師から独立したLoRAと全語彙KL/CE
- `src/llm_jp_logkv/train.py`: epoch制御・任意の時間制限付きDDP訓練、定期記録、checkpoint再開
- `src/llm_jp_logkv/training_data.py`: 完結会話の変換、本文hash分割、uint32キャッシュ
- [固定減衰LogKVの蒸留](doc/distillation-20261102.md)
- [実重みの初期評価（蒸留なし）](doc/initial-quality-20261102.md)
- [アダプター仕様と検証](doc/llama-adapter-20261101.md)
- [引き継ぎと実装計画](doc/handoff-20261101.md)
- [初期環境の検証結果（174 tests passed）](doc/setup-validation-20261101.md)

コードは MIT（移植元の LICENSE を保持）。元モデルのライセンスはモデルカードを参照してください。
