# 専用教師GPUとFSDP2によるAttention全パラメータ学習

`scripts/run_distillation_fsdp.py` は教師専用1 GPUと生徒用GPU群を使う訓練スクリプト。
生徒の全32層Q/K/V/Oと、指定時のCausalConvを学習し、FFN・embedding・RMSNorm・LM headは凍結する。
生徒GPU間で学習する重み・勾配・Adam状態をFSDP2により分割する。
凍結部分の重みは各生徒GPUに保持する。教師の重み・Attention・RoPEは変更しない。

## 起動と再開

```bash
./scripts/uv.sh run --locked python scripts/run_distillation_fsdp.py \
  --run-name distill-full-fsdp-20261109 \
  --teacher-gpu 0 --gpus 1,2,3,4,5 \
  --data /mnt/raid0/llm-jp-logkv/datasets/distill-cache-all-reasoning-20261103-100k \
  --no-rope --conv-kernel-size 4 --skip-samples
```

- `--teacher-gpu` は教師専用GPU、`--gpus` は生徒GPUのみ。重複を拒否する。
- full専用で `--attention-training full` は省略可能。`--rank` / `--alpha` は無視する。
- RoPEなし・conv4は上の例での指定。省略時の構成は従来どおりRoPEあり・convなし。
- 既定: 1 epoch、時間無制限、系列長1,024、各生徒batch1、勾配蓄積4、最新2 checkpoint保持。
- 語彙方向のメモリ節約のため `--loss-chunk-tokens` は64を既定とする。
- 上の例では `--skip-samples` で初回・定期・終了時の文章生成を省略する。
  loss検証は省略しない。文章生成を行う場合はこの指定を外す。
  `--skip-initial-samples` は初回だけ省略し、定期・終了時の生成は残す。
- 再開は同じコマンドに `--resume latest` を追加する。GPU番号の入れ替えは可能だが、
  生徒台数と訓練・データ設定は同じにする。DDPやLoRAのcheckpointからのresumeは拒否する。
- `--num-epochs` / `--max-steps` は再開前を含む総数。`--max-hours` は任意指定。
- `echo save_and_exit > control.cmd` または `just save-and-exit` でstep境界に保存・停止。
  `--control-file PATH` も指定できる。停止後の全検証・生成・外部評価は省略する。

初回のVRAM確認には別run名を使い、例えば次を追加する。

```bash
--max-steps 2 --skip-samples --skip-final-validation --val-examples-per-rank 1
```

初回Adam状態の確保だけでなく、その状態を保持した次stepも測る。
本番runには確認用のstep制限を付けない。`--evaluate-final` を付けると、訓練プロセス終了後に
従来の評価CLIを生徒GPUの先頭1台で実行する。

## 計算と通信

教師は各生徒の入力を受け取り、最終RMSNorm後のBF16 hidden stateを返す。
生徒側の同じ凍結LM headから全語彙の教師分布を計算するため、目的関数は従来の
既定の目的関数は `KL(teacher || student) × T² + ce_weight × NTP CE` のまま。
`--hidden-mse-weight` を指定した場合だけ、後述のブロック出力MSEを追加する。
有効な全非padding targetsを使い、CE係数の既定は0.1。
1,024 tokens・batch1の場合、転送hiddenは8 MiBで、BF16全語彙logitsの384 MiBより小さい。

教師は生徒のbatchを順番に処理し、同時に5倍のactivationを持たない。
最初の実装は同期式で、教師の先行計算・非同期キューは使わない。
PyTorchのeager NCCLではP2P操作の直列化について警告が出ることがあるが、この方式では
教師への転送順を明示している。実機はNVLinkなしのPCIe接続であり、通信による速度の制約は残る。

生徒はCPU上の一時モデルから初期化し、使用しない元Attentionと教師全体への参照を外す。
CPUモデルの構築・GPU転送は生徒ごとに順番に行う。FFNは凍結しても入力への勾配を通す。
FSDP2のFP32 master/moments、BF16計算、FP32勾配集約を使い、毎microbatchでreduce-scatterする。
勾配蓄積中に全パラメータの勾配を各GPUへ保持する `no_sync` は使わない。
lossを全生徒の有効token総数で正規化し、clip normは全shardにわたる値で計算する。

検証は全生徒のforward回数・shapeを合わせ、端数のdummy rowsは損失のmaskで除外する。
データを重複して採点しない。初回・定期検証は `--val-examples-per-rank` に基づく限定集合、
正常終了時は全held-outを検証する。生成サンプルは全生徒が同じ入力を処理し、
rank0が選んだtokenとEOSに揃える。教師はこの間待機する。

### 生成中に学習ログが出ない場合

生成を有効にすると、最初のoptimizer更新より前に6問×最大256 tokensの生成を行う。
FSDPで分割したAttention/convの重みを各token・各層で集め直すため、
訓練のまとめた系列forwardよりも通信の負担が大きい。
実8Bの本番run `distill-full-fsdp-20261109` では、22:53:37 JSTに初期検証が完了し、
23:19 JSTでも3問目の242 token目を生成中だった。その後4問目への進行を確認した。
この時点では学習step 0であり、初期生成に時間を使っていた。
GPU使用率だけでは計算の進行とNCCL待機を判別できないため、workerのスタックと
生成位置の変化を確認した。先の実8B VRAM確認は `--skip-samples` 付きであり、
この生成時間は確認できていなかった。

進捗表示修正後は、`training.log` と `experiments/RUN/latest/progress.jsonl` に
検証の開始・終了、各問のprefill完了・最初のtoken・16 tokensごと・生成完了を記録する。
`samples/PHASE-step-NNNNNNN.json` も1問完了ごとに原子的に更新し、
全問完了前は `complete: false`、全問完了後は `complete: true` とする。
実行中の旧プロセスにはコード変更は反映されない。
現状の `save_and_exit` はstep境界で処理するため、生成の途中では即座に停止しない。
診断記録は `/mnt/raid0/llm-jp-logkv/logs/fsdp-startup-debug-20261109-01/`。

## ブロック出力MSEの追加（2026-11-10）

```bash
./scripts/uv.sh run --locked python scripts/run_distillation_fsdp.py \
  --run-name distill-full-fsdp-mse-20261110 \
  --teacher-gpu 0 --gpus 1,2,3,4,5 \
  --data /mnt/raid0/llm-jp-logkv/datasets/distill-cache-all-reasoning-20261103-100k \
  --no-rope --conv-kernel-size 4 --hidden-mse-weight 0.1 --skip-samples
```

全32層のFFN後・残差加算後、末尾RMSNorm前の出力を1対1で比較する。
係数0が既定。`0.1` は指定例で、最適係数や品質向上を確認した値ではない。
詳しい平均の定義と層別ログは [MSE損失の説明](distillation-20261102.md#ブロック出力のmse蒸留2026-11-10)を参照。
再開は同じ係数で `--resume latest`。係数なしの旧checkpointは0扱いで、途中の係数変更は拒否する。

有効時、教師は各生徒batchの全ブロック出力を1つのBF16 bufferへコピーし、
最終正規化後hiddenに続けて送る。1生徒分を送った後に解放して次のbatchを処理する。
系列長1,024・batch1・隠れ次元4,096の場合、追加bufferは生徒1台あたり256 MiB。
教師から生徒5台への追加転送はmicrobatchあたり1.25 GiB、勾配蓄積4では1更新あたり5 GiB。
一時演算領域やcheckpoint再計算もあるため、これだけをVRAMピークの増分とは見なさない。
0指定時はbufferの確保・ブロック状態の追加転送を行わない。

## 保存

既存と同じ `--run-name` のパス、排他ロック、起動ごとのsession、追記ログを使う。
checkpointは一時ディレクトリへの保存とhash確定後にrenameし、成功後だけ古いものを削除する。

- `adapter.pt`: 通常の推論・チャットで読める全Attention/conv重み。1パラメータずつ集め、
  rank0のCPUだけに全体を保持する。畳み込みなしでは約5 GiB。
- `optimizer-rank-NNNNN.pt`: 生徒ごとのCPU化したoptimizer shard。5台では各約2 GiB。
- `training_state.pt`: step、epoch、データカーソル、全生徒のCPU/CUDA RNG、backend、訓練設定。
- `config.json` / `complete.json`: 構成、全重み・state・shardのhashなど。

合計は1個約15 GiB、既定の保持2個で約30 GiB。保存中はさらに1個分の空きが必要。
`training_state.pt` の形式はDDP版と異なるが、推論が読む `adapter.pt` / `config.json` は共通。

## 検証結果（2026-11-09）

CPU 367 tests passed。ローカル教師と転送hiddenによる損失・勾配の一致、凍結、
RoPE/convの各構成、CLI、端数を含む検証データの一度だけの採点などを検査した。

小型BF16モデルを実際の教師1＋生徒5 GPUで3 steps更新し、検証・生成サンプル・最新2個保持を確認。
`save_and_exit` 後に `--resume latest` した場合、連続実行と全学習重み・5 optimizer shardsが完全一致。
教師と生徒の凍結重みも不変。通常の単一GPU推論へのcheckpoint読み込みも通過した。
同じデータと各rankのloss倍率を単一GPUで順次計算しFP32で平均した参照との更新差は
最大 `4.66e-10`。単一GPU側で先にloss倍率を変えるとBF16丸めの差が出るため、
倍率・集約の位置を揃えて比較している。
RoPEあり・convなしでも4 stepsで1 epochを完了し、最後の勾配蓄積が1 microbatchになる場合を確認。
全held-out 7例の有効81 tokensを重複なしで採点した。

実8B・RTX 3090 × 6・系列長1,024・batch1・RoPEなし・conv4で、短いVRAM確認を行った。
モデル初期値は固定revisionの元モデル。`--loss-chunk-tokens 64`、sampleと終了時全検証は省略。

| run | 勾配蓄積 | 更新数 | 生徒の学習step中割当ピーク | 教師の割当ピーク | 更新時間 |
|---|---:|---:|---:|---:|---:|
| `fsdp-full-preflight-20261109-01` | 1 | 1 | 17.56 GiB | 16.12 GiB | 10.71秒 |
| `fsdp-full-preflight-20261109-02` | 4 | 2 | 17.56 / 18.37 GiB | 16.12 GiB | 42.29 / 42.18秒 |

PyTorchのallocated計測値であり、CUDA context・NCCLの外部確保やallocatorのreservedは含まない。
両runとも初回検証・有限なloss/勾配・optimizer更新・約15 GiBのcheckpoint保存を通過し、
`reason=max_steps` で正常終了。2回目の更新は既存のAdam状態を持ったまま実行できた。
保存後の全checkpointファイルのhash、学習重みのFP32・有限性・パラメータ数も再検査した。
本訓練は開始しておらず、この短い確認から長時間の速度や品質改善は結論しない。
実8Bの生成サンプルと全held-out評価はこの確認では行っていない。
起動とログは `logs/RUN/`、条件・metrics・終了記録は `experiments/RUN/`、重みは
`checkpoints/RUN/` に保存。全GPUは確認後に解放した。

小型検査のスクリプト・ログは `/mnt/raid0/llm-jp-logkv/logs/remote-fsdp-20261109-01/`。
`validation.json` に集計、`artifact-hashes.json` に検証成果物のhashを保存している。
これは訓練機構の確認であり、フルパラメータ蒸留後の品質改善を示すものではない。

実装参照: [PyTorch 2.11 FSDP2](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html)。
