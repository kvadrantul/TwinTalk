# TwinTalk

Telegram-бот для симуляции живой переписки между двумя виртуальными людьми на основе их реальной истории переписки.

## Как это работает

1. Создайте 3 Telegram-бота через @BotFather (1 менеджер + 2 персонажа)
2. Загрузите менеджер-боту экспорт переписки из Telegram Desktop (result.json)
3. Отправьте токены двух ботов-персонажей
4. Добавьте ботов-персонажей в группу
5. Отправьте /start — боты начнут общаться друг с другом в стиле реальных людей

Сообщения генерируются через WaveSpeed AI на основе анализа стиля переписки каждого человека.

## Технологии

- Python 3.11+
- python-telegram-bot (async)
- WaveSpeed AI (OpenAI-compatible API)
- SQLite

## Запуск локально

```bash
pip install -r requirements.txt
```

Создайте `.env` файл:
```
MANAGER_BOT_TOKEN=ваш_токен_менеджера
WAVESPEED_API_KEY=ваш_wavespeed_ключ
```

```bash
python main.py
```

## Деплой на Railway

1. Подключите репозиторий к Railway
2. Настройте переменные окружения:
   - `MANAGER_BOT_TOKEN`
   - `WAVESPEED_API_KEY`
3. Railway автоматически задеплоит проект

## Структура

- `bot/` — обработчики команд Telegram
- `parser/` — парсинг экспорта и профилирование
- `ai/` — интеграция с WaveSpeed AI
- `orchestrator/` — движок симуляции
- `telegram_proxy/` — отправка от имени ботов
- `db/` — база данных SQLite
- `utils/` — утилиты и валидация
