# Используем официальный образ Python 3.10
FROM python:3.10-slim

# Установка системных зависимостей для сборки пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Удобно: сначала обновим инструмент для установки Python-пакетов
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Указываем рабочую директорию внутри контейнера
WORKDIR /home/amismailov/project

# Копируем зависимости и ставим их с кешированием слоёв Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для Jupyter Lab
EXPOSE 8888

CMD ["bash"]
