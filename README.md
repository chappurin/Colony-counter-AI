README.md
# Lab File Organizer (DX-Min v1)

研究用ファイル（画像・データ・文書等）をルールに基づいて自動仕分け・リネーム・重複検知する簡易ツール。

## 使い方
1. `rules.json` を編集して仕分けルールを定義
2. ドライラン: `python3 organize.py --src <整理したいフォルダ> --dry-run`
3. 実行: `python3 organize.py --src <整理したいフォルダ>`
4. ログ: `./_organizer_logs/last_run.log` / `./_organizer_logs/manifest.json`（UNDO用）

## 主な機能
- 拡張子＆簡易内容判定でカテゴリ分け（image/data/docs/misc）
- 日付抽出（EXIF/更新日/ファイル名）で `YYYY/MM/` に自動振り分け
- 重複検知（内容ハッシュ）
- 衝突しないリネーム（`_v2` 等）
- 乾式実行（--dry-run）とUNDOサポート（manifest参照）

## 例