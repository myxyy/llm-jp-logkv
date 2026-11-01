# 最小Llama Attentionアダプター（2026-11-01）

## 決定した仕様

`LlamaAttentionAdapter(source, mode="dense" | "logkv", chunk_size=4,
no_level_bias=False)` は既存LlamaAttentionのQ/K/V/Oを同じModuleとして流用する。
入力は `[B,T,D]` と元LlamaRotaryEmbeddingが返すcos/sin、任意の明示的state。
戻り値は `(output, state)`。state.offsetから始まる絶対位置のcos/sinを呼び出し側で渡す。
圧縮済みキーへRoPEを再適用しない。追加conv/gate/normはない。

- dense: 因果マスク、selfを含む。キャッシュは元のKVヘッド数。
  pinned Llama eagerに合わせsoftmaxはFP32に変換する。
- logkv: RoPE適用後にGQA KVをQヘッド数に展開し、無改変参照コアの
  projected tensor入口 `_attend` を呼ぶ。末尾Qでpoolingし、self slotあり。
  デフォルトはレベル固定減衰、比較用にno_level_biasを指定可能。
  キャッシュはQヘッド数で、対象8Bでは32ヘッド。8ヘッド共有ではない。
  coreの未使用投影ModuleはIdentityに置き換え、追加パラメータを持たせない。
- 明示的stateは入力呼び出しでは変更しない。学習時にはグラフを保持する。
  推論時は呼び出し側でtorch.no_gradを使う。

対応範囲はpaddingなし、非空入力、dropout=0、連続した系列位置。
任意のattention mask、padding、packed sequence、beam reorder、HF Cache、
model.generateには未対応。既存self_attnのdrop-in置換ではない。
coreのprivate入口への依存は回帰テストで監視する。
元のembedding、FFN、RMSNorm、tokenizer、chat templateには変更を加えていない。

## 検証

依存関係はuv.lock（Torch 2.11.0、Transformers 5.4.0）。全てoffline CPU、
seed=17のランダムAttention（hidden=32、Q heads=4、KV heads=2、head_dim=8、
batch=2、length=19）。実モデル重み・データは取得していない。
モデル参照revisionはUPSTREAMではなく、対象モデルの
`7c19bc90e49b717381e4f474ee95964c05217a28`。この検証ではロードしていない。
参照コアの由来はUPSTREAM.json、意味変更なし。
開始時のrepository baselineは `246fdd9`。

- 元Llama eagerとdenseの出力、入力勾配、分割入力とKVキャッシュを比較。
  FP64/FP32はatol=rtol=1e-6、BF16はatol=rtol=2e-2。
- LogKVはC=2/4、固定減衰/減衰なしで一括と分割の出力・入力勾配を比較。
  FP64、atol=rtol=1e-12。未完成状態のスロット数が各レベルC未満も確認。
- 別実装の逐次FP64 oracleが、読み出し後にチャンクを繰り上げる方法で
  pooling/レベル減衰/因果性を検証。atol=rtol=1e-12。
- 既存の独立参照・streaming・grouped勾配テストも維持。

実行コマンド:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ./scripts/uv.sh run --locked pytest -q
```

結果: **182 passed in 5.77s**。

品質指標・GPU速度・メモリ性能は未計測。推論品質の維持は主張しない。

## 次の区切り

Transformersのモデル全体でdenseモードの等価性を確認するための接続層と
Cacheインターフェースを設計する。padding/位置/生成時のstate管理を明確にし、
その後少数層置換・実重み取得・蒸留の時間と必要ストレージを見積もる。
