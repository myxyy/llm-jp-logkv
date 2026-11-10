# llm-jp-logkv

`llm-jp/llm-jp-4.1-8b-thinking` の学習済み Llama を出発点に、Attention を
LogKV に置換し、蒸留・追加学習による適応を検証するプロジェクトです。

既定では元Llamaの全32層でQ/K/V/O、RoPE、FFN、RMSNormを流用し、Attentionのみ
LogKVに変更する独立推論APIとCLI、および固定減衰LogKVのLoRA蒸留を実装しています。
蒸留時には `--no-rope` で生徒のRoPEを無効化し、`--conv-kernel-size 4` で
各層のAttention前に元LogKVと同じCausalConvを追加できます。
Attentionの学習方式は `--attention-training lora`（既定）または `full` を選択できます。
`full` はQ/K/V/Oの全重みを直接更新し、FFNなどの共有部分と教師は凍結します。
TransformersのCache/model.generateへの組み込みは未対応で、独自の生成処理を使用します。
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

## チェックポイントを使うチャットサーバ

```bash
# run内の保存完了済み最大stepを起動時に選択し、1 GPUでロードする
CUDA_VISIBLE_DEVICES=0 ./scripts/uv.sh run --locked python scripts/chat_server.py \
  --run-name distill-20261102 --checkpoint latest

# 特定のcheckpointを指定する場合
CUDA_VISIBLE_DEVICES=0 ./scripts/uv.sh run --locked python scripts/chat_server.py \
  --checkpoint /mnt/raid0/llm-jp-logkv/checkpoints/distill-20261102/step-0002599 \
  --port 8000
```

ブラウザで `http://127.0.0.1:8000` を開く。`--run-name` 指定時の
`--checkpoint` の既定値は `latest`。step名の指定も可能で、起動後の自動切替は行わない。
固定revisionの元モデル重みとcheckpointの学習済みAttentionを読み込み、model revisionとadapterの
hashを検査する。LoRA rank・alpha・LogKV chunk size・RoPE・CausalConvはcheckpointの
設定を使用し、有効な場合は畳み込みの学習済み重みと正規化重みも読み込む。
会話中はLogKV状態に加えて畳み込みの直近kernel size−1 tokensの履歴も保持する。

- 思考過程（analysis）と最終回答（final）を分けてストリーム表示。思考と生の生成文字列は
  折り畳み可能で、表示にはモデルが実際に生成した内容を使用する。
- 画面でreasoning effort、temperature、top-p、生成上限を指定可能。
  既定はlow、temperature 0（greedy）、top-p 1、最大2,048 tokens。
  `--max-new-tokens` は毎ターンの初期値と上限を設定し、思考部分もこの上限に含む。
  Reasoning effortは先頭のシステム文に含まれるため会話開始時に固定し、
  「新しい会話」で変更する。temperature・top-p・生成上限はターンごとに変更できる。
- ブラウザのタブごとにLogKV状態をサーバで保持し、2ターン目以降は新しい質問だけを
  追加する。会話履歴の読み直しは行わず、過去のanalysisもfinalも圧縮状態に保持する。
  同時生成は1件で、他の要求にはbusyを返す。
- 入力トークン数の上限はなく、`--max-input-tokens` は廃止した。
  元モデルの `max_position_embeddings` による拒否も行わない。
  入力は `--prefill-chunk-size`（既定128）ずつ処理する。
- 「停止」はprefillのチャンク間・生成のトークン間で反映する。生成中の停止・上限では
  途中までの思考と応答を保持し、ターンを閉じて次の質問を受け付ける。入力の読込途中で
  停止した場合は、そのターン全体を破棄する。切断・推論エラー時は未確定の状態を破棄する。
- 無操作の状態は既定30分で解放する。`--session-ttl 3600` なら1時間保持する。
  生成中のセッションは期限切れにしない。「新しい会話」・ページ再読込で状態をリセットし、
  タブを閉じたときにも解放を要求する。期限切れやサーバ再起動後は新しい会話が必要。
- 表示の `93+16 tokens` は今回追加した入力+生成トークン数。「状態累計」は
  過去の思考とターン終端の補完を含む処理済みトークン数で、保存する生の履歴長ではない。
  tok/sはprefillを除いた生成区間の値、秒数はprefillを含む。

追加依存なしのローカルサーバ（認証なし）で、既定hostは `127.0.0.1`。
SSH先で起動した場合はVS Codeのポート転送、または
`ssh -L 8000:127.0.0.1:8000 HOST` でアクセスできる。
`--device cuda:0`、`--host`、`--port` も指定可能。Ctrl+Cでサーバを終了する。
起動条件は `${LOG_DIR}/chat-*.json` に保存し、会話本文はサーバのファイルへ保存しない。
本体プロジェクトの作業ツリーやサーバ用Python環境には依存しない。

状態を保持した長時間対話の試行・早期反復・元モデルとの比較は
[2026-11-05の対話観察記録](doc/long-dialogue-20261105.md)を参照。
rank 128の完了モデルでも短い設定確認から対話を試したが、訂正への不追従・物語からの逸脱と
算数誤答により4ターン・952生成tokensで早期終了した。大量反復は観測していない。
[2026-11-08の対話観察記録](doc/long-dialogue-20261108.md)を参照。長期生成の評価には未到達。
記憶課題を外した雑談では、rank128を52ターン・16,773生成tokensまで継続した。
以前の単語の大量連続は再現しなかったが、約1.5万生成tokens台で空のコードブロックなどの
持続的な反復が出て、話題変更後にも再発したため停止。
rank32の同じ質問列との観察と制約は [2026-11-09の雑談記録](doc/casual-dialogue-20261109.md)を参照。

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
教師の全重みと生徒のembedding・FFN・既存RMSNorm・LM headを固定し、既定では全32層の
Q/K/V/Oに追加したLoRAを全語彙KL + next-token CEで学習します。
`--attention-training full` ではQ/K/V/O本体を教師から独立した重みとして学習します。
CausalConvを有効にした場合は、その追加重みも
学習します。BF16、層checkpoint、語彙損失の分割を用い、
一定stepごとに生成全文・token IDs・検証損失・再開可能なcheckpointを記録します。

ユーザー承認の全6 GPU・8時間試行は3,126 stepsで正常終了しました。詳細な結果・起動・再開手順と
データrevisionは[蒸留実験記録](doc/distillation-20261102.md)を参照してください。
蒸留後は `infer --adapter CHECKPOINT_DIR` と `evaluate-initial --adapter CHECKPOINT_DIR`
で評価できます。通常の `infer` は引き続き未蒸留モデルを使用します。

手動で新しく開始する例（2 GPU、global batch=24、時間制限なしで1 epoch）。
新規学習には新しいrun名を付け、再開時は同じ名前を使います。
以下は3サブセットのlow/medium/highから各サブセット合計10万件を候補抽出した拡充版です。
キャッシュは訓練227,698件・検証2,243件で、既定の系列長1,024では訓練161,240件・検証1,618件を使います。
フィルタ条件・作成コマンドは[データの説明](doc/distillation-20261102.md#データ)を参照してください。
旧runの再開には、そのrunで使用した旧キャッシュを指定します。

```bash
cd /home/myxy/llm-jp-logkv
distill_run=my-distillation
distill_root=/mnt/raid0/llm-jp-logkv
./scripts/uv.sh run --locked python scripts/run_distillation.py \
  --run-name "$distill_run" --gpus 0,1 --evaluate-final \
  --data "$distill_root/datasets/distill-cache-all-reasoning-20261103-100k" \
  --batch-size 1 --grad-accum 12 \
  --sample-every 200 --save-every 200 --max-checkpoints 2 --validate-every 200
```

同じrunの最新checkpointから再開する場合は、元のコマンドに **`--resume latest`** を追加します。

```bash
./scripts/uv.sh run --locked python scripts/run_distillation.py \
  --run-name "$distill_run" --resume latest --gpus 0,1 --evaluate-final \
  --data "$distill_root/datasets/distill-cache-all-reasoning-20261103-100k" \
  --batch-size 1 --grad-accum 12
```

RoPEなし・CausalConvありで新規蒸留する場合は、別のrun名で次のように起動します。

```bash
./scripts/uv.sh run --locked python scripts/run_distillation.py \
  --run-name distill-no-rope-conv4 --gpus 0,1 --evaluate-final \
  --data /mnt/raid0/llm-jp-logkv/datasets/distill-cache-all-reasoning-20261103-100k \
  --batch-size 1 --grad-accum 12 --no-rope --conv-kernel-size 4
```

`--no-rope` は生徒のQ/K回転を無効化します。教師のRoPEは維持します。
`--conv-kernel-size` は0で無効（既定）、1以上で有効です。4は元LogKVの訓練CLIの既定値です。
追加ブロックは元LogKVと同じ初期化・配置で、元LlamaのFFNやRMSNormを置き換えません。
再開時も両オプションを同じ値で指定してください。構成の異なるcheckpointからの再開は拒否します。
チャットや `infer --adapter`、終了後の評価はcheckpointの構成を自動復元します。
オプションのない旧checkpointも引き続き使用できます。詳細と検証結果は
[構成の切り替え](doc/distillation-20261102.md#ropeとcausalconvの切り替え2026-11-08)を参照してください。

AttentionをLoRAではなく直接学習する場合は、新しいrunで **`--attention-training full`** を
追加します。`--no-rope --conv-kernel-size 4` と併用できます。FFNは引き続き凍結し、
`--rank` / `--alpha` は使いません。再開時も `--attention-training full` を指定し、
推論・チャットではcheckpointから学習方式を自動復元します。LoRAとfullの間での再開は拒否します。

従来の `run_distillation.py` のDDPでは、各GPUにBF16教師約16 GiBとFP32 Attentionの
重み・勾配・Adam状態20 GiBが必要で、中間値・通信領域を除いても約36 GiBになります。
**RTX 3090（24 GiB）には収まらず、GPU台数を増やしても1台あたりの使用量は減りません。**
[fullの条件・保存容量・検証](doc/distillation-20261102.md#attentionの全パラメータ学習2026-11-08)を参照してください。

RTX 3090 × 6では、専用の `run_distillation_fsdp.py` で教師1台・生徒5台に分けられます。
`--gpus` は**生徒だけ**を指定し、`--teacher-gpu` と重複させません。既定は教師0、生徒1〜5。
Attentionと有効なCausalConvの重み・勾配・Adam状態をFSDP2で分割し、FFNなどは凍結します。
このスクリプトはfull専用です。1 epoch・時間無制限・最新2 checkpoint保持が既定です。

```bash
./scripts/uv.sh run --locked python scripts/run_distillation_fsdp.py \
  --run-name distill-full-fsdp-20261109 \
  --teacher-gpu 0 --gpus 1,2,3,4,5 \
  --data /mnt/raid0/llm-jp-logkv/datasets/distill-cache-all-reasoning-20261103-100k \
  --no-rope --conv-kernel-size 4 --skip-samples
```

上の例は訓練中の文章生成を省略します。FSDPの逐次生成には各tokenで重みを集める通信があり、
実8Bでは初期サンプル6問だけでも数十分かかります。`--skip-samples` を外すと生成を有効化し、
`--skip-initial-samples` なら初回だけ省略します。損失の検証はどちらでも実行します。
検証・生成の進捗は `training.log` と `experiments/RUN/latest/progress.jsonl` に出力します。

再開は同じコマンドに `--resume latest` を追加します。`control.cmd` の `save_and_exit` にも対応。
FSDP再開は同じ生徒台数・訓練設定が必要で、DDP/LoRA checkpointからのresumeは受け付けません。
保存した `adapter.pt` は通常のチャット・推論で使用できます。
初回は `--max-steps 2 --skip-samples --skip-final-validation --val-examples-per-rank 1` を付けた
別runでVRAMを確認できます。詳細は [専用教師＋FSDP2訓練](doc/distillation-fsdp-20261109.md)。

各ブロックの教師・生徒出力を近づける補助損失は `--hidden-mse-weight 0.1` で追加できます。
通常のLoRA/full蒸留と専用教師＋FSDPの双方に対応し、FFNは凍結を維持します。
既定は `0`（無効）。全32層のFFN後・残差加算後、末尾RMSNorm前の出力を比較し、
有効token・隠れ次元・層で平均した生のMSEをKL＋CEに加えます。
`0.1` は指定例であり、最適係数ではありません。再開時は同じ係数が必要です。
FSDPでは教師状態の転送とVRAM使用量が増えます。
[損失・ログ・起動例](doc/distillation-20261102.md#ブロック出力のmse蒸留2026-11-10)を参照してください。

`--run-dir`・`--log-dir` は廃止しました。保存先は `.env` の設定とrun名から自動決定します。
ログは `logs/RUN/training.log` に追記し、checkpointは `checkpoints/RUN/step-NNNNNNN/` に保存します。
起動ごとの設定・終了記録・生成例は `experiments/RUN/sessions/ID/`、監督プロセスの記録は
`logs/RUN/sessions/ID/` に保持します。両方の `latest/` から今回の記録を参照できます。
旧版のrunも同名で `--resume latest` を指定して再開できます。

`--max-hours` の省略時は時間無制限、`--num-epochs` の既定値は1です。
複数epochは `--num-epochs 3`、任意の時間上限は `--max-hours 8` のように指定します。
epoch数・時間・step数を複数指定した場合は、最初に達した上限で保存して終了します。
epoch末尾の勾配蓄積が指定回数に満たなくても、残りのミニバッチを更新してから終了します。
学習率は既定ではepochの進捗に応じて減衰し、経過時間には依存しません。

チェックポイントはrunごとに最新2個を保持します。本体LogKVと同じ `--max-checkpoints N`
（1以上の整数、既定値2）で変更できます。オプション区切りの `--` は不要です。
新しい保存の完了後に古いstepを削除し、通常終了・`save_and_exit`による保存も保持数に含めます。
再開時に保持数を変更できます。整理対象は現在のrunのみで、再開元の別runには適用しません。

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
複数runでは起動時に `--control-file /絶対パス/control.cmd` を指定し、
runごとに別のファイルを使ってください。
再開時もGPU台数・データ・訓練設定は元と同じ値を指定します。
`--resume latest` は保存完了済みの最大stepを選びます。該当checkpointがなければエラーになります。
同じrunの二重起動や、`--resume` を付けずに既存runを開始する操作もエラーになります。
特定のcheckpointは `--resume step-0000200` やパスで指定できます。
同じrunを過去のstepへ巻き戻す場合は、新しいrun名と元checkpointのパスを指定してください。
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
