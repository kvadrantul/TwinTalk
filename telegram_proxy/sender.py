import aiohttp
import asyncio
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramBotProxy:
    """
    Sends messages on behalf of a character bot using its token.
    Includes rate limiting (token bucket: max 20 messages per minute per bot).
    """

    def __init__(self, token: str):
        self.token = token
        self._rate_limiter = TokenBucket(rate=20, per=60)  # 20 msgs per 60 seconds
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_message(self, chat_id: int, text: str) -> dict:
        """
        Send a message to a chat as this character bot.

        POST https://api.telegram.org/bot{token}/sendMessage
        Body: {"chat_id": chat_id, "text": text}

        Waits for rate limiter before sending.
        Returns the Telegram API response dict.
        Raises TelegramAPIError on failure.
        """
        url = f"{TELEGRAM_API_BASE}/bot{self.token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}

        logger.debug("send_message → chat_id=%s, text_len=%d", chat_id, len(text))

        await self._rate_limiter.acquire()

        session = self._get_session()
        try:
            async with session.post(url, json=payload) as resp:
                status = resp.status
                body = await resp.json()

                if status == 401:
                    raise TelegramAPIError(401, body.get("description", "Unauthorized – invalid token"))
                if status == 403:
                    raise TelegramAPIError(403, body.get("description", "Forbidden – bot not in group"))
                if status == 429:
                    retry_after = body.get("parameters", {}).get("retry_after", 30)
                    logger.warning("Rate limited by Telegram, retry_after=%s", retry_after)
                    raise TelegramAPIError(429, f"Too Many Requests – retry after {retry_after}s")
                if status >= 400:
                    raise TelegramAPIError(status, body.get("description", "Unknown error"))

                if not body.get("ok", False):
                    desc = body.get("description", "ok=false with no description")
                    raise TelegramAPIError(status, desc)

                logger.debug("send_message ← ok, message_id=%s", body.get("result", {}).get("message_id"))
                return body

        except aiohttp.ClientError as exc:
            raise TelegramAPIError(0, f"Network error: {exc}") from exc

    async def get_me(self) -> dict:
        """
        Call getMe to validate the token and get bot info.

        GET https://api.telegram.org/bot{token}/getMe

        Returns: {"ok": true, "result": {"id": ..., "username": ..., "first_name": ...}}
        Raises TelegramAPIError if token is invalid (401).
        """
        url = f"{TELEGRAM_API_BASE}/bot{self.token}/getMe"

        logger.debug("get_me →")

        session = self._get_session()
        try:
            async with session.get(url) as resp:
                status = resp.status
                body = await resp.json()

                if status == 401:
                    raise TelegramAPIError(401, body.get("description", "Unauthorized – invalid token"))
                if status >= 400:
                    raise TelegramAPIError(status, body.get("description", "Unknown error"))

                if not body.get("ok", False):
                    desc = body.get("description", "ok=false with no description")
                    raise TelegramAPIError(status, desc)

                logger.debug("get_me ← ok, username=%s", body.get("result", {}).get("username"))
                return body

        except aiohttp.ClientError as exc:
            raise TelegramAPIError(0, f"Network error: {exc}") from exc

    async def get_chat_member(self, chat_id: int, user_id: int) -> dict:
        """
        Check if this bot is a member of a group chat.

        GET https://api.telegram.org/bot{token}/getChatMember
        Params: chat_id, user_id (the bot's own user id)

        Returns member info with "status" field (member, administrator, creator, left, kicked)
        Raises TelegramAPIError if bot is not in the group.
        """
        url = f"{TELEGRAM_API_BASE}/bot{self.token}/getChatMember"
        params = {"chat_id": chat_id, "user_id": user_id}

        logger.debug("get_chat_member → chat_id=%s, user_id=%s", chat_id, user_id)

        session = self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                status = resp.status
                body = await resp.json()

                if status == 401:
                    raise TelegramAPIError(401, body.get("description", "Unauthorized – invalid token"))
                if status == 403:
                    raise TelegramAPIError(403, body.get("description", "Forbidden – bot not in group"))
                if status >= 400:
                    raise TelegramAPIError(status, body.get("description", "Unknown error"))

                if not body.get("ok", False):
                    desc = body.get("description", "ok=false with no description")
                    raise TelegramAPIError(status, desc)

                logger.debug(
                    "get_chat_member ← ok, status=%s",
                    body.get("result", {}).get("status"),
                )
                return body

        except aiohttp.ClientError as exc:
            raise TelegramAPIError(0, f"Network error: {exc}") from exc

    async def close(self):
        """Close the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


class TokenBucket:
    """
    Simple token bucket rate limiter.
    rate: max operations per `per` seconds.
    """

    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a token is available, then consume it."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per))
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.per / self.rate)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class TelegramAPIError(Exception):
    """Error from Telegram Bot API."""

    def __init__(self, status_code: int, description: str):
        self.status_code = status_code
        self.description = description
        super().__init__(f"Telegram API error {status_code}: {description}")


async def validate_bot_token(token: str) -> dict:
    """
    Validate a bot token by calling getMe.
    Returns bot info dict on success.
    Raises TelegramAPIError on failure.

    This is a standalone function (creates temporary session).
    """
    proxy = TelegramBotProxy(token)
    try:
        info = await proxy.get_me()
        return info
    finally:
        await proxy.close()


async def check_bot_in_group(token: str, chat_id: int, bot_user_id: int) -> bool:
    """
    Check if a bot is a member of a group.
    Returns True if bot is member/admin/creator, False otherwise.
    """
    proxy = TelegramBotProxy(token)
    try:
        member = await proxy.get_chat_member(chat_id, bot_user_id)
        return member.get("result", {}).get("status") in ("member", "administrator", "creator")
    except TelegramAPIError:
        return False
    finally:
        await proxy.close()
