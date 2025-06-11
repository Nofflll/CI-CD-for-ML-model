# CI/CD for ML model

Этот проект демонстрирует полный цикл MLOps для модели машинного обучения, от разработки и версионирования до непрерывной интеграции и развертывания. Цель — предсказание качества красного вина на основе его химических характеристик.

## 🚀 Возможности

- **Версионирование:** Код (Git), Данные и Модели (DVC).
- **Автоматизация:** CI/CD пайплайн с использованием GitHub Actions.
- **Обучение:** Автоматизированное обучение и оценка модели.
- **Развертывание:** Модель обслуживается через REST API с помощью FastAPI и контейнеризирована с Docker.
- **Непрерывное развертывание:** Docker-образ автоматически собирается и публикуется в Docker Hub при каждом изменении в ветке `main`.

## 🛠️ Стек технологий

- **Python 3.9**
- **Машинное обучение:** Scikit-learn, Pandas, NumPy
- **Версионирование:** Git, DVC
- **API:** FastAPI
- **Контейнеризация:** Docker, Gunicorn
- **CI/CD:** GitHub Actions
- **Отслеживание экспериментов:** MLflow (используется локально)

## 📁 Структура проекта
```
.
├── .github/workflows/ci.yaml   # Описание CI/CD пайплайна
├── data/                       # Директория с данными (отслеживается DVC)
│   ├── raw/
│   └── processed/
├── models/                     # Директория с моделями (отслеживается DVC)
├── notebooks/                  # Jupyter-ноутбуки для исследований
├── src/                        # Исходный код
│   ├── get_data.py             # Скрипт для загрузки данных
│   ├── train.py                # Скрипт для обучения модели
│   └── app.py                  # FastAPI приложение
├── tests/                      # Тесты для проекта
├── .gitignore
├── Dockerfile                  # Docker-файл для сборки API
├── params.yaml                 # Параметры проекта
├── requirements.txt            # Зависимости Python
└── README.md
```

## ⚡ Как запустить

### 1. Необходимые утилиты

-   Git
-   Docker

### 2. Запуск готового Docker-контейнера

CI/CD пайплайн автоматически собирает и публикует готовый к использованию Docker-образ в Docker Hub.

1.  **Скачайте образ из Docker Hub:**
    ```bash
    docker pull noffll/ci-cd-for-ml-model:latest
    ```
    *(Замените `noffll` на ваш логин в Docker Hub, если он отличается)*

2.  **Запустите контейнер:**
    Эта команда запустит API-сервер на `localhost:8000`.
    ```bash
    docker run -p 8000:8000 noffll/ci-cd-for-ml-model:latest
    ```

3.  **Отправьте запрос для предсказания:**
    Откройте новый терминал и используйте `curl`, чтобы отправить запрос с характеристиками вина.
    ```bash
    curl -X 'POST' 'http://localhost:8000/predict' \
    -H 'Content-Type: application/json' \
    -d '{
      "fixed_acidity": 7.4,
      "volatile_acidity": 0.7,
      "citric_acid": 0.0,
      "residual_sugar": 1.9,
      "chlorides": 0.076,
      "free_sulfur_dioxide": 11.0,
      "total_sulfur_dioxide": 34.0,
      "density": 0.9978,
      "pH": 3.51,
      "sulphates": 0.56,
      "alcohol": 9.4
    }'
    ```

### 3. Локальная разработка (Опционально)

Если вы хотите запустить проект из исходного кода:

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/Nofflll/CI-CD-for-ML-model.git
    cd CI-CD-for-ML-model
    ```

2.  **Установите зависимости:**
    Рекомендуется использовать виртуальное окружение.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Восстановите данные и модели:**
    Вам понадобится установленный DVC (`pip install dvc`).
    ```bash
    dvc pull
    ```
    *(Примечание: это требует настройки удаленного хранилища DVC. Для этого проекта вы можете пересоздать артефакты, запустив скрипты)*

4.  **Запустите пайплайн обучения:**
    ```bash
    python3 src/get_data.py --output_folder data/raw
    python3 src/train.py --config params.yaml
    ```

5.  **Запустите API локально:**
    ```bash
    python3 src/app.py
    ```
