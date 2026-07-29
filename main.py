import asyncio
import logging
import os

from telegram.ext import Application, CommandHandler, MessageHandler, filters

import config
from db.schema import init_db
from bot.handlers import (
    start_handler, import_handler, pause_handler,
    resume_handler, speed_handler, stop_handler,
    document_handler, text_handler, regenerate_handler,
)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    os.makedirs("data", exist_ok=True)

    asyncio.run(init_db())

    application = Application.builder().token(config.MANAGER_BOT_TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("import", import_handler))
    application.add_handler(CommandHandler("pause", pause_handler))
    application.add_handler(CommandHandler("resume", resume_handler))
    application.add_handler(CommandHandler("speed", speed_handler))
    application.add_handler(CommandHandler("stop", stop_handler))
    application.add_handler(CommandHandler("regenerate", regenerate_handler))

    # Document handler — JSON file uploads in private chat
    application.add_handler(MessageHandler(
        filters.Document.ALL & filters.ChatType.PRIVATE,
        document_handler,
    ))

    # Text handler — token collection in private chat (excludes commands)
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
        text_handler,
    ))

    application.run_polling()


if __name__ == "__main__":
    main()
