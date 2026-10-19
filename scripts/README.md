# Hugging Face 模型下載工具

`download.py` 會從 Hugging Face Hub 下載指定模型 revision 的檔案，並放入明確指定的本機目錄。預設下載該 revision 的完整快照，包含權重、設定、Tokenizer 與模型程式碼等檔案；下載中斷後再次執行會沿用 Hub 快取及目標目錄中的既有檔案。

## 安裝依賴

在專案根目錄執行：

```bash
python -m pip install "huggingface_hub==1.33.0"
```

此版本與 Mage-VL Transformers 環境目前鎖定的 `huggingface_hub` 版本一致。

## 下載完整模型

每次執行都必須明確提供模型 `repo_id`、revision 與下載目錄。建議 revision 使用 commit SHA，確保後續可重現：

```bash
python scripts/download.py \
  --repo-id microsoft/Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --local-dir checkpoints/microsoft-Mage-VL-d88b153
```

`--local-dir` 是模型檔案的實際目標目錄。專案根目錄的 `.gitignore` 會忽略 `checkpoints/` 下的模型權重與快取，同時保留 `checkpoints/val.md` 及 `checkpoints/val.xlsx`。

## 私有模型與存取權杖

公開模型不需要權杖。下載私有或受限模型時，先將 Hugging Face 權杖放入環境變數，不要直接寫進命令列或文件：

```bash
export HF_TOKEN=hf_your_token
python scripts/download.py \
  --repo-id organization/private-model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --local-dir checkpoints/private-model
```

若使用的環境變數名稱不是 `HF_TOKEN`，以 `--token-env NAME` 指定名稱。

## 選擇檔案與下載參數

可重複提供 `--allow-pattern`，讓 Hub 只下載符合模式的檔案；也可用 `--ignore-pattern` 排除檔案。若沒有指定模式，則下載完整快照。部分下載可能缺少模型執行所需的設定或程式碼，使用前請確認模型格式與推論服務的需求。

```bash
python scripts/download.py \
  --repo-id microsoft/Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --local-dir checkpoints/mage-vl-config-only \
  --allow-pattern '*.json' \
  --allow-pattern '*.py' \
  --ignore-pattern '*safetensors*'
```

`--max-workers N` 可調整並行下載數，預設為 `8`。需要重新抓取本機已有檔案時，加入 `--force-download`。執行 `python scripts/download.py --help` 可查看完整參數。
