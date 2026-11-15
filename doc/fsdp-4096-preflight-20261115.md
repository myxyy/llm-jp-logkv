# 系列長4,096のFSDP実機検証（2026-11-15）

教師1台＋生徒5台のRTX 3090で、系列長4,096・Attention全パラメータ学習・
RoPEなし・conv4・隠れ状態MSE係数0.1を検証した。
初期検証、2回のoptimizer更新、checkpoint保存がOOMなしで完了した。
損失と勾配は有限。2回目の更新ではAdam状態を保持したまま実行できた。
長期訓練や生成品質の検証ではなく、訓練コード・既存の本番キャッシュは変更していない。

## 条件とデータ

- 実行コード: `095bd2215851f1c754990e5dd2830db52f745414`、起動時working tree clean。
- 元モデル: `llm-jp/llm-jp-4.1-8b-thinking`、revision
  `7c19bc90e49b717381e4f474ee95964c05217a28`。元モデルから初期化し、既存の蒸留checkpointからの再開ではない。
- 元SFT: `llm-jp/llm-jp-4.1-thinking-sft-data`、revision
  `cb4210190af6a0000fd91cb5bb8361ad6d6ade01`。
- full Attention＋conv、FFNなどは凍結。batch1、勾配蓄積4、loss chunk64、CE係数0.1。
- 物理GPU 0が教師、1〜5が生徒。開始前に全GPUの計算プロセスがないことを確認。
- run: `fsdp-4096-preflight-20261115-01`。

旧キャッシュは最大2,049 tokensなので、そのpaddingを伸ばすだけでは有効tokenの長さを
確認できない。先行調査の9,000件の標本から、全長4,097以下の2往復会話を長い順に選び、
内容hashで重複を除いた45件の専用キャッシュを作成した。
最初の40件をtrain、残り5件をvalidとした。品質評価用の代表標本ではない。

- 全体: 4,009〜4,096 tokens。train: 4,030〜4,096、valid: 4,009〜4,025。
- 全例にuser 2個、analysis 2個、final 2個。切り詰めや無関係な会話の連結はない。
- 元テンプレートの過去assistant分岐にanalysisを残す処理を加えた検証用コピーを使用。
  元モデル配下のテンプレートは変更していない。日付などは既存の変換関数に合わせた。
- `uint32`、train/validは重複なし。教師・生徒に同じ入力を渡す。
  実装のshapeは4,097 IDs（入力4,096＋next-token target用1個）、右padding。
  損失は非padding位置だけを採点する。
- 選択された45件はMagpieのmedium 2/high 15、数学・コードのlow 17/medium 11。
  split構成を調整した本番データセットの代用ではない。

キャッシュはRAIDの `datasets/multiturn-4096-preflight-20261115-01/`。
元parquetのhashを確認し、選択位置・ID・全長・group hash・使用テンプレートを保存した。
マルチターンを取り込む本番用前処理の実装は別途必要。

## 実行コマンド

```bash
./scripts/uv.sh run --locked python scripts/run_distillation_fsdp.py \
  --run-name fsdp-4096-preflight-20261115-01 \
  --teacher-gpu 0 --gpus 1,2,3,4,5 \
  --data /mnt/raid0/llm-jp-logkv/datasets/multiturn-4096-preflight-20261115-01 \
  --sequence-length 4096 --batch-size 1 --grad-accum 4 \
  --no-rope --conv-kernel-size 4 --hidden-mse-weight 0.1 \
  --loss-chunk-tokens 64 --max-steps 2 --num-epochs 0 \
  --skip-samples --skip-final-validation --val-examples-per-rank 1 \
  --control-file /mnt/raid0/llm-jp-logkv/logs/fsdp-4096-preflight-20261115-01/control.cmd
```

上は実行済みコマンド。再検証は新しいrun名と専用control-fileを使う。
2更新で止めるための`--num-epochs 0 --max-steps 2`であり、本番の既定は1 epochのまま。
外側の監視スクリプトにも30分の上限を設けたが、約270秒で正常終了し発動しなかった。

## 結果

| 更新 | 生徒のPyTorch割当ピーク（全生徒の最大） | 更新時間 | 有効target tokens | loss | 勾配ノルム（clip前） |
|---|---:|---:|---:|---:|---:|
| 1 | 19.85 GiB | 91.77秒 | 81,237 | 20.7517 | 155.3105 |
| 2 | 21.86 GiB | 92.68秒 | 81,207 | 20.7054 | 159.4335 |

教師のPyTorch割当ピークは17.46 GiB。初期検証5件は約12.25秒、loss=20.7843。
全32層のMSE/RMSを含め、記録された数値はすべて有限。

`nvidia-smi`を2秒間隔で測ったデバイス全体の使用量の最大値は、教師18,567 MiB、
生徒最大23,943 MiB（約23.38 GiB）。生徒GPUの表示上の空きは最小633 MiBだった。
こちらはallocatorの予約領域・CUDA/NCCLなども含む。PyTorchのallocated値と意味が異なり、
サンプリング間の瞬間ピークも保証しない。予約領域には再利用できる部分があるため、
633 MiBをそのまま追加tensorの確保上限とは見なさない。

`reason=max_steps`、step 2、終了コード0。`adapter.pt`・`training_state.pt`・
全5 optimizer shardsのSHA-256を再計算して`complete.json`と一致することを確認した。
終了後、全GPUの計算プロセスがなくなったことも確認した。
本番訓練・生成サンプル・終了時の全held-out検証・checkpointからの再開は実施していない。

## 判断と記録

この条件の4,096-token学習は短い実機確認を通過した。まずは4,096を候補に
マルチターンの本番キャッシュを構成できる。ただしVRAMの余裕は小さく、GPU同居や
batch増加、生成・検証・長期実行の全条件まで保証するものではない。
全32層MSEを有効にしたまま通過しており、今回メモリ節約のための訓練コード修正は不要だった。

RAID `/mnt/raid0/llm-jp-logkv/` 以下:

- `logs/fsdp-4096-preflight-20261115-01/`: `prepare_probe.py`、`run_probe.py`、
  `command.json`、`training.log`、`gpu-telemetry.jsonl`、終了記録、`collect_results.py`、`result.json`。
- `experiments/fsdp-4096-preflight-20261115-01/`: 実行manifest、runtime、metrics、検証値、終了記録。
- `checkpoints/fsdp-4096-preflight-20261115-01/step-0000002/`: 検証用checkpoint。
- `logs/multiturn-lengths-20261115-01/`: 元となる長さ調査の標本・スクリプト・集計。

データ準備スクリプトの初回実行は空のArrow選択に失敗した。空batchをスキップする修正後に
再実行して成功し、失敗ログも保持した。GPU検証は1回のみ。
変更は実験ドキュメントのみのためCPUテストスイートは再実行せず、実機確認と成果物検証を行った。
