.PHONY: install app test collect train backtest lint clean help

help:  ## ヘルプ表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## 依存パッケージのインストール
	pip install -r requirements.txt
	pip install pytest

app:  ## Streamlitアプリを起動
	streamlit run src/app/streamlit_app.py

test:  ## テスト実行
	python -m pytest tests/ -v

collect:  ## 過去データ収集（直近6ヶ月）
	python -m src.scraper.history_collector --months 6

train:  ## LightGBMモデルの学習
	python -m src.model.trainer

backtest:  ## バックテスト実行
	python -m src.model.backtester --budget 3000 --strategy balance

lint:  ## 構文チェック
	python -m py_compile src/scraper/client.py
	python -m py_compile src/scraper/race_data.py
	python -m py_compile src/scraper/history_collector.py
	python -m py_compile src/features/pipeline.py
	python -m py_compile src/model/predictor.py
	python -m py_compile src/model/trainer.py
	python -m py_compile src/model/backtester.py
	python -m py_compile src/betting/optimizer.py
	python -m py_compile src/app/streamlit_app.py
	@echo "All files OK"

clean:  ## キャッシュ削除
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
