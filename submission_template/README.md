# 提出物テンプレート

## ディレクトリ構成

```
submission.zip
├── policy_server.py     # ← MyPolicy クラスを編集する（必須）
├── requirements.txt     # ← 追加依存があれば記載（必須）
└── model_weights/       # ← チェックポイント等を配置（任意）
```

## 手順

1. `policy_server.py` の `MyPolicy` クラスにモデルのロードと推論を実装する
2. モデル重みを `model_weights/` に配置する
3. 追加ライブラリがあれば `requirements.txt` に追記する
4. zip にまとめて提出する:
   ```bash
   zip -r submission.zip policy_server.py requirements.txt model_weights/
   ```

## ローカルテスト

```bash
# サーバー起動
pip install -r requirements.txt
python policy_server.py

# 別ターミナルで評価（ランダムポリシーの動作確認）
python -m pipeline --server-url http://localhost:8000 --track track1 --benchmark libero_spatial --n-episodes 2 --max-steps 10
```

## 提出前セルフチェック（推奨）

提出→採点は 1 回 15〜20 分かかります。フル採点を待たずにローカルで問題を潰せる
よう、バリデータを同梱しています。**提出前に必ず実行してください。**

```bash
# zip を丸ごと検査（静的チェック + サーバーを起動して I/O スモークテストまで）
python validate_submission.py submission.zip

# 展開済みディレクトリでも可
python validate_submission.py .

# サーバーを起動せず静的チェックのみ（高速）
python validate_submission.py submission.zip --static

# 依存を入れてから動的チェック（クリーンな環境での再現確認）
python validate_submission.py submission.zip --install
```

検査内容: zip 健全性 / Zip Slip・zip bomb / サイズ上限（20GB）/ 必須ファイル /
`policy_server.py` の構文・エンドポイント / `requirements.txt` の構文・外部ソース
禁止（`git+`・`--index-url` 等は不可）/ サーバー起動 → `/health`→`/reset`→`/act`
で action が float32 shape (7,)・NaN/Inf なし・レイテンシまで確認します。

`PASS` かつ ERROR 0 件で提出可能です（WARN は推奨事項）。採点環境でも unzip 直後に
同じ検査を通します。

## 注意事項

- `policy_server.py` のサーバー部分（FastAPI エンドポイント、シリアライゼーション）は変更しないでください
- `requirements.txt` に `git+https://…` や `--index-url` 等の外部ソース指定は使えません（採点環境は外部通信を遮断します）
- `get_action()` は 10 秒以内に応答してください（タイムアウトで失敗扱い）
- `reset()` はエピソードごとに呼ばれます。内部状態（action chunking のキャッシュ等）をクリアしてください
