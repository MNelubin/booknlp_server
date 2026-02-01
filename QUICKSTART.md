# 🚀 Quick Start - BookNLP GPU Service

## Шаг 1: Настройка окружения

```bash
# 1. Склонируйте или перейдите в директорию
cd /opt/gstory/booknlp-gateway

# 2. Создайте .env файл
cp .env.example .env

# 3. Отредактируйте путь к моделям (важно!)
nano .env
```

**Важно в .env:**
```bash
# Укажите путь где хранить модели на хосте
MODELS_PATH=/opt/booknlp-models  # Или любой другой путь

# Размер модели
BOOKNLP_MODEL=big  # или small
```

## Шаг 2: Создайте директории

```bash
# Создайте папку для моделей
mkdir -p ${MODELS_PATH}  # Используйте значение из .env

# Создайте папку для данных
mkdir -p data temp
```

## Шаг 3: Запустите сервис

```bash
# Соберите и запустите
docker compose up -d

# Следите за логами
docker compose logs -f

# Дождитесь сообщения: "Application startup complete"
```

## Шаг 4: Проверьте работу

```bash
# Проверка здоровья сервиса
curl http://localhost:8888/health

# Должно вернуть что-то вроде:
# {
#   "status": "healthy",
#   "cuda_available": true,
#   "gpu_count": 1,
#   "gpu_name": "NVIDIA GeForce RTX 3090"
# }
```

## Шаг 5: Тестовый запрос

```bash
curl -X POST http://localhost:8888/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Frodo Baggins was a hobbit who lived in the Shire. He had a friend named Samwise Gamgee.",
    "book_id": "test"
  }'
```

## 📡 Использование из gstory (LXC)

В gstory создайте клиент для связи с GPU сервисом:

```python
# В LXC: src/gstory/extractors/remote_booknlp.py
import requests

class RemoteBookNLPClient:
    def __init__(self, host_url: str = "http://YOUR_HOST_IP:8888"):
        self.base_url = host_url

    def extract(self, text: str, book_id: str) -> dict:
        response = requests.post(
            f"{self.base_url}/extract",
            json={"text": text, "book_id": book_id},
            timeout=3600  # 1 hour timeout
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
```

## 🔧 Управление сервисом

```bash
# Остановить
docker-compose down

# Перезапустить
docker-compose restart

# Посмотреть логи
docker-compose logs -f

# Обновить код
docker-compose up -d --build

# Зайти в контейнер (для отладки)
docker-compose exec booknlp-gpu bash
```

## ⚡ Производительность

- **Big model на RTX 3090**: ~2 мин / 100K tokens
- **Small model на CPU**: ~15 мин / 100K tokens
- **Ускорение**: ~7.5x

## 📂 Структура файлов после обработки

```
/opt/gstory/booknlp-gateway/
├── data/
│   └── booknlp_test/
│       ├── test.tokens
│       ├── test.entities
│       ├── test.quotes
│       └── ...
├── models/  (смонтирована с хоста)
└── temp/
```

## 🐛 Troubleshooting

### GPU не видна
```bash
# Проверьте NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Модели не загружаются
```bash
# Проверьте права на папку моделей
ls -la ${MODELS_PATH}

# Убедитесь что папка смонтирована
docker-compose exec booknlp-gpu ls -la /models
```

### Проверьте логи
```bash
docker-compose logs -f --tail=100
```
