# 初期環境の確認結果（2026-11-01）

## 完了範囲

- `/home/myxy/llm-jp-logkv` を独立した Git リポジトリ（main）として初期化。
- uv による Python 3.12 環境・依存関係導入と `uv.lock` の作成。
- `.env`、RAID 上の保存先、`.venv` シンボリックリンクを作成。
- 移植元 `02abb4f803ca85db5a01a17acc3394e8b44f8bb2` の LogKV コアと回帰テストを取り込み。
- 元モデルの設定・tokenizer・chat template を固定 revision から取得。

Llama Attention の置換と蒸留は未実装。8B の重み取得、GPU 学習、実モデルの品質・速度測定は未実施。

## 確認した環境

| 項目 | バージョン |
|---|---|
| Python | 3.12.3 |
| uv | 0.10.9 |
| torch | 2.11.0+cu128 |
| transformers | 5.4.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 4.8.5 |
| huggingface-hub | 1.33.0 |

`uv pip check` は導入済み81パッケージに互換性エラーなし。
CUDA runtime は12.8、`torch.cuda.is_available()` は true。
この確認では GPU の計算性能・数値精度を測っていない。

## 検証結果

CPU・オフラインで **174 passed**（最終実行5.79秒）。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./scripts/uv.sh run --locked pytest -q
```

- 移植元の独立参照と LogKV の forward/step/predict、階層構造、各オプションの検証。
- grouped Attention の統計・勾配と streaming 経路の一致。
- `.env` の展開、保存先の明示性、相対パス・未固定 revision の拒否。
- 小型ランダム Llama の GQA、prefill/decode と一括 forward の一致、LoRA 初期出力と勾配。
- コピー元・コピー先ファイルの SHA-256 と `UPSTREAM.json` の一致。

実モデルのメタデータをオフラインで読み込み、`LlamaConfig`、Q32/KV8、
vocab/tokenizer長196608、chat template からのプロンプト生成を確認。
テスト用の日本語質問は89トークン。token ID は config の語彙範囲内。
`apply_chat_template` で ID リストが必要な箇所は `return_dict=False` を明示する。

## 保存先の確認と修正

`scripts/uv.sh` と `uv run --env-file .env` の両方で、RAID 上の Python 環境を使用。
最終 doctor の全保存先は `/mnt/raid0/llm-jp-logkv` 配下で、存在・書き込み可を確認した。
設定上の `HF_HOME` は `/mnt/raid0/llm-jp-logkv/cache/huggingface`。

初回確認でログインシェルの既存 `HF_HOME` が `.env` に優先する差を検出。
Python 側も `.env` 優先へ修正し、継承した HF 設定との衝突を回帰テストに含めた。
uv 自体の環境選択にも `.env` を優先するには `scripts/uv.sh` を使う。

元プロジェクトの作業ツリーは変更なし。`.env`、`.venv`、モデル・データが
新リポジトリの Git 管理対象外であることも確認済み。

## ローカルの記録

すべて `/mnt/raid0/llm-jp-logkv/logs/` 配下:

- `setup-20261101-pytest-final.xml`: 最終テスト結果
- `setup-20261101-doctor-final.json`: 最終環境・保存先
- `setup-20261101-model-metadata.json`: 元モデルの設定・tokenizer 検証と取得ファイルのハッシュ
- `setup-20261101-pytest.xml`、`setup-20261101-doctor.json`: 修正前の初回確認結果を保存

モデル配置:
`/mnt/raid0/llm-jp-logkv/models/llm-jp--llm-jp-4.1-8b-thinking/7c19bc90e49b717381e4f474ee95964c05217a28/`

取得済みメタデータは約13 MiB、`.safetensors` 重みファイルは0個。
