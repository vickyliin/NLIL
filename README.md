# NLIL
News Legal Information Labeling

a.k.a Natural Language in Law

## Data

- news.json
	- https://drive.google.com/file/d/0B5Uu9BOINP_lWDlrUUpmUDB2T00/view
	- 新聞原始檔案，不分欄位
	
- implicit.jsonl
	- 沒出現 `{法, 律, 條例, 通則}` 的新聞
	- 分成 `title` `time` `from` `content` 四個欄位

- explicit.jsonl
	- 有出現 `{法, 律, 條例, 通則}` 至少一個的新聞
	- 分成 `title` `time` `from` `content` 四個欄位
	
- explicit_ckip.jsonl
	- `explicit.jsonl` 加上 ckip 斷詞結果（`pos_title` `pos_content`）

- explicit_jseg.jsonl
	- `explicit.jsonl` 加上 jseg 斷詞結果（`pos_title` `pos_content`）
	
- laws.csv
	- 法律大全

- selected_frames.json
	- https://drive.google.com/file/d/0B5Uu9BOINP_lei1CM0NVeVhsWWc/view
	- `explicit_ckip.jsonl` 裡 `content` 提到法律大全裡的法律的新聞，紀錄在欄位 `law`
	
- occurrence.csv
	- `selected_frames.json` 裡各個法律的出現次數統計
