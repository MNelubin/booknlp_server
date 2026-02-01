# BookNLP GPU Service

**BookNLP GPU microservice** - FastAPI сервер для обработки текстов через BookNLP на видеокарте.

## 🎯 Назначение

Отдельный сервис для запуска на машине с GPU. Принимает тексты по HTTP API, обрабатывает через BookNLP big model на видеокарте, возвращает результаты.

## 📋 Структура

```
booknlp-gateway/
├── docker-compose.yml      # Docker конфигурация
├── Dockerfile              # Образ контейнера
├── booknlp_server.py       # FastAPI сервер
├── .env.example            # Пример конфигурации
├── .gitignore
└── README.md
```

## 🚀 Быстрый старт

### 1. Настройка

```bash
# Скопируйте .env.example в .env
cp .env.example .env

# Отредактируйте пути к моделям и данные
nano .env
```

### 2. Запуск

```bash
docker-compose up -d
```

### 3. Проверка

```bash
# Health check
curl http://localhost:8888/health

# Должен вернуть информацию о GPU
```

## ⚙️ Конфигурация через .env

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MODELS_PATH` | Путь к моделям на хосте | `./models` |
| `DATA_PATH` | Путь для данных | `./data` |
| `BOOKNLP_MODEL` | Размер модели (small/big) | `big` |
| `GPU_DEVICE` | Номер GPU | `0` |
| `API_PORT` | Порт API | `8888` |

## 📡 API Эндпоинты

### `GET /` - Информация о сервисе
```json
{"service": "BookNLP GPU Service", "model": "big", "status": "ready", "gpu": true}
```

### `GET /health` - Проверка здоровья и GPU
```json
{"status": "healthy", "cuda_available": true, "gpu_count": 1, "gpu_name": "NVIDIA ..."}
```

### `POST /extract` - Извлечение сущностей
```json
{
  "text": "Frodo Baggins lived in the Shire...",
  "book_id": "lotr",
  "pipeline": "entity,quote,supersense,event,coref"
}
```

## 🔧 Управление

```bash
# Логи
docker-compose logs -f

# Перезапуск
docker-compose restart

# Остановка
docker-compose down

# Обновление
docker-compose up -d --build
```

## 📂 Монтирование моделей

Модели хранятся на хосте и монтируются в контейнер:

```yaml
volumes:
  - ${MODELS_PATH:-./models}:/models
```

Это позволяет:
- Управлять моделями через файловую систему хоста
- Делать бэкапы
- Обновлять модели без пересборки контейнера

## 🔗 Интеграция с gstory

Из gstory (в LXC) использовать клиент:

```python
import requests

class RemoteBookNLPClient:
    def __init__(self, base_url: str = "http://HOST_IP:8888"):
        self.base_url = base_url

    def extract(self, text: str, book_id: str):
        response = requests.post(
            f"{self.base_url}/extract",
            json={"text": text, "book_id": book_id}
        )
        return response.json()
```
