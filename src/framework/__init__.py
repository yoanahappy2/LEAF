"""
agent_framework — 可遷移的瀕危語言 Multi-Agent 保護框架

框架代碼（agent_framework/）與語言配置（paiwan_config/）完全分離。
換語言只需替換配置，不改任何框架代碼。

架構：
    Orchestrator（LLM #1）— 總管，理解用戶意圖，分派任務
        ├── Knowledge Agent（LLM #2）— 知識檢索、翻譯、語料
        ├── Teaching Agent（LLM #3）— 教學策略、學習路徑
        └── Quality Agent（LLM #4）— 品質控制、自我評估

作者: 地陪
日期: 2026-05-12
"""
