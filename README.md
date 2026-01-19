# mlsharp-service

基于 Apple ml-sharp 的单图高斯喷溅建模与渲染服务（FastAPI）。

## 依赖
- Python 3.13
- CUDA 渲染仅在 Linux/Windows + NVIDIA GPU 可用

## 模型权重
服务不会自动下载权重。请手动放置：
- `models/sharp_2572gikvuh.pt`

> 目录内含 `models/.keep` 占位文件。

## 配置
复制并修改 `.env.example`：
- `API_KEY`：接口认证用（Bearer）
- `MODEL_PATH`：模型文件路径
- `DATA_DIR`：数据目录
- `DB_PATH`：SQLite 路径

## Docker 启动
CPU（macOS/无 CUDA）：
```
docker compose -f docker-compose.cpu.yml up --build
```

CUDA（Linux/Windows NVIDIA）：
```
docker compose -f docker-compose.cuda.yml up --build
```

服务端口：`11011`

## API 简述
- `POST /v1/predict`：上传图片（单张），返回 `task_id` + `file_id`
- `POST /v1/render`：基于已有 `file_id` 渲染视频（CUDA 才可用）
- `GET /v1/tasks/{task_id}`：查询任务
- `GET /v1/files/{file_id}`：文件信息
- `GET /v1/files/{file_id}/original|gaussians|render|render-depth`

## 说明
- 渲染只支持已有推理结果（通过 `file_id` 关联）。
- CPU/MPS 环境下渲染接口会返回错误（需 CUDA）。
