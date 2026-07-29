import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error."""
    def __init__(self, message: str, user_message: str = None):
        self.message = message
        self.user_message = user_message or "Произошла ошибка. Попробуйте позже."
        super().__init__(message)


class ConfigError(AppError):
    """Configuration error (missing env vars, etc)."""
    pass


class ParseError(AppError):
    """Error parsing Telegram export file."""
    pass


class TokenValidationError(AppError):
    """Invalid bot token."""
    def __init__(self):
        super().__init__(
            "Invalid bot token",
            "Неверный токен бота. Убедитесь что токен корректный и попробуйте снова."
        )


class SessionError(AppError):
    """Error with session state."""
    pass


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global error handler for the Telegram bot.
    Logs the error and sends a user-friendly message.
    """
    logger.error("Exception while handling an update:", exc_info=context.error)
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуйте позже или отправьте /start."
        )


def handle_parse_error(e: Exception) -> str:
    """Convert parse errors to user-friendly Russian messages."""
    error_msg = str(e).lower()
    if "not valid json" in error_msg or "json" in error_msg:
        return "❌ Файл не является корректным JSON. Убедитесь что это результат экспорта из Telegram Desktop."
    if "participants" in error_msg or "2" in error_msg:
        return "❌ В переписке должно быть ровно 2 участника."
    if "50" in error_msg or "messages" in error_msg:
        return "❌ В переписке должно быть минимум 50 сообщений."
    return f"❌ Ошибка при обработке файла: {e}"
