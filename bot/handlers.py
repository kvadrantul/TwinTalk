import uuid
import json
import logging
from telegram import Update
from telegram.ext import ContextTypes

from db import repository
from parser.export_parser import parse_telegram_export_bytes
from parser.profiler import build_character_profile, select_few_shot_examples, profile_to_dict
from telegram_proxy.sender import validate_bot_token, TelegramAPIError
from orchestrator.engine import ConversationOrchestrator
from bot.conversation import (
    STATE_AWAITING_FILE, STATE_AWAITING_TOKEN_A,
    STATE_AWAITING_TOKEN_B, STATE_READY,
    active_orchestrators,
)

logger = logging.getLogger(__name__)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start in private and group chats."""
    chat = update.effective_chat
    user = update.effective_user
    
    if chat.type == "private":
        await update.message.reply_text(
            "👋 Привет! Я менеджер чат-симуляции.\n\n"
            "Я создам живую переписку двух виртуальных людей на основе их реальной переписки из Telegram.\n\n"
            "Отправьте /import чтобы начать настройку."
        )
    else:
        # Group chat: find session and start orchestrator
        session_id = None
        
        # Try to find existing session for this user
        sessions = await repository.get_sessions_by_user(user.id)
        for s in sessions:
            if s.get("status") in ("idle", "ready", "stopped", "paused"):
                session_id = s["id"]
                break
        
        if not session_id:
            await update.message.reply_text(
                "❌ Нет настроенной симуляции.\n\n"
                "Сначала в личном чате со мной:\n"
                "1. /import → загрузите result.json\n"
                "2. Отправьте токены двух ботов\n"
                "3. Добавьте менеджера и ботов-персонажей в эту группу\n"
                "4. Отправьте /start здесь"
            )
            return
        
        # Update group_chat_id in session
        await repository.update_session_group(session_id, chat.id)
        
        # Initialize and start orchestrator
        try:
            orch = ConversationOrchestrator(session_id, chat.id)
            await orch.initialize()
            active_orchestrators[session_id] = orch
            
            # Store session_id in user_data for this group context
            context.user_data["session_id"] = session_id
            
            await orch.start()
            await update.message.reply_text("▶️ Симуляция запущена!")
        except Exception as e:
            logger.exception("Error starting orchestrator")
            await update.message.reply_text(f"❌ Ошибка запуска: {e}")


async def import_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /import — start the onboarding flow."""
    if update.effective_chat.type != "private":
        await update.message.reply_text("Используйте /import в личном чате со мной.")
        return
    
    user_id = update.effective_user.id
    await repository.set_user_state(user_id, STATE_AWAITING_FILE)
    await update.message.reply_text(
        "📎 Отправьте мне файл экспорта чата из Telegram (result.json).\n\n"
        "Как получить:\n"
        "1. Откройте Telegram Desktop\n"
        "2. Настройки → Продвинутые → Экспорт данных\n"
        "3. Выберите формат: JSON\n"
        "4. Отправьте мне полученный result.json"
    )


async def pause_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pause."""
    session_id = context.user_data.get("session_id")
    if session_id and session_id in active_orchestrators:
        await active_orchestrators[session_id].pause()
        await update.message.reply_text("⏸ Симуляция приостановлена.")
    else:
        await update.message.reply_text("Нет активной симуляции.")


async def resume_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /resume."""
    session_id = context.user_data.get("session_id")
    if session_id and session_id in active_orchestrators:
        await active_orchestrators[session_id].resume()
        await update.message.reply_text("▶️ Симуляция возобновлена.")
    else:
        await update.message.reply_text("Нет активной симуляции.")


async def speed_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /speed <multiplier>."""
    if not context.args:
        await update.message.reply_text("Использование: /speed <0.5|1|2|4>")
        return
    
    try:
        multiplier = float(context.args[0])
        if multiplier not in (0.5, 1.0, 2.0, 4.0):
            raise ValueError
    except ValueError:
        await update.message.reply_text("Допустимые значения: 0.5, 1, 2, 4")
        return
    
    session_id = context.user_data.get("session_id")
    if session_id and session_id in active_orchestrators:
        await active_orchestrators[session_id].set_speed(multiplier)
        await update.message.reply_text(f"⚡ Скорость: {multiplier}x")
    else:
        await update.message.reply_text("Нет активной симуляции.")


async def stop_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stop."""
    session_id = context.user_data.get("session_id")
    if session_id and session_id in active_orchestrators:
        await active_orchestrators[session_id].stop()
        del active_orchestrators[session_id]
        await repository.update_session_status(session_id, "stopped")
        await repository.clear_user_state(update.effective_user.id)
        context.user_data.clear()
        await update.message.reply_text("⏹ Симуляция остановлена.")
    else:
        await update.message.reply_text("Нет активной симуляции.")


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle JSON file upload during onboarding."""
    user_id = update.effective_user.id
    state = await repository.get_user_state(user_id)
    
    if not state or state["current_step"] != STATE_AWAITING_FILE:
        await update.message.reply_text("Сначала отправьте /import чтобы начать настройку.")
        return
    
    try:
        await update.message.reply_text("⏳ Загружаю и анализирую переписку...")
        
        # Download file
        tg_file = await context.bot.get_file(update.message.document.file_id)
        file_bytes = bytes(await tg_file.download_as_bytearray())
        
        # Parse
        parsed = parse_telegram_export_bytes(file_bytes)
        
        if len(parsed.participants) != 2:
            await update.message.reply_text(
                f"⚠️ В переписке найдено {len(parsed.participants)} участников. Нужно ровно 2."
            )
            return
        
        # Store parsed data in context
        context.user_data["parsed_chat"] = {
            "participants": parsed.participants,
            "messages": [
                {
                    "sender_name": m.sender_name,
                    "text": m.text,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in parsed.messages
            ],
        }
        
        await repository.set_user_state(user_id, STATE_AWAITING_TOKEN_A)
        
        await update.message.reply_text(
            f"✅ Переписка загружена! Найдены участники:\n"
            f"• {parsed.participants[0]}\n"
            f"• {parsed.participants[1]}\n\n"
            f"Теперь отправьте токен бота для *{parsed.participants[0]}*.\n"
            f"Создайте бота через @BotFather и отправьте его токен.",
            parse_mode="Markdown"
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ Ошибка формата файла: {e}")
    except Exception as e:
        logger.exception("Error processing file upload")
        await update.message.reply_text(f"❌ Ошибка при обработке файла: {e}")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages during onboarding (token collection)."""
    user_id = update.effective_user.id
    state = await repository.get_user_state(user_id)
    
    if not state:
        return
    
    current_step = state["current_step"]
    text = update.message.text.strip()
    
    if current_step == STATE_AWAITING_TOKEN_A:
        await _handle_token_a(update, context, text, user_id)
    elif current_step == STATE_AWAITING_TOKEN_B:
        await _handle_token_b(update, context, text, user_id)


async def _handle_token_a(update, context, token: str, user_id: int):
    """Validate and store token for character A."""
    try:
        bot_info = await validate_bot_token(token)
        bot_result = bot_info["result"]
        
        context.user_data["token_a"] = token
        context.user_data["bot_a_info"] = {
            "id": bot_result["id"],
            "username": bot_result["username"],
            "first_name": bot_result.get("first_name", ""),
        }
        
        parsed = context.user_data["parsed_chat"]
        name_b = parsed["participants"][1]
        
        await repository.set_user_state(user_id, STATE_AWAITING_TOKEN_B)
        
        await update.message.reply_text(
            f"✅ Токен принят!\n\n"
            f"Теперь отправьте токен бота для *{name_b}*.",
            parse_mode="Markdown"
        )
    except TelegramAPIError:
        await update.message.reply_text("❌ Неверный токен. Отправьте корректный токен бота.")
    except Exception as e:
        logger.exception("Error validating token A")
        await update.message.reply_text(f"❌ Ошибка валидации токена: {e}")


async def _handle_token_b(update, context, token: str, user_id: int):
    """Validate and store token for character B, then create session and characters."""
    try:
        bot_info = await validate_bot_token(token)
        bot_result = bot_info["result"]
        
        context.user_data["token_b"] = token
        context.user_data["bot_b_info"] = {
            "id": bot_result["id"],
            "username": bot_result["username"],
            "first_name": bot_result.get("first_name", ""),
        }
        
        # Create session and characters
        await update.message.reply_text("⏳ Создаю персонажей на основе переписки...")
        
        session_id = uuid.uuid4().hex
        await repository.create_session(session_id, user_id)
        
        parsed = context.user_data["parsed_chat"]
        name_a = parsed["participants"][0]
        name_b = parsed["participants"][1]
        
        # Reconstruct ParsedMessage-like objects for profiling
        from parser.export_parser import ParsedMessage
        from datetime import datetime
        
        all_messages = []
        for m in parsed["messages"]:
            all_messages.append(ParsedMessage(
                id=len(all_messages),
                sender_name=m["sender_name"],
                sender_id="",
                text=m["text"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
            ))
        
        msgs_a = [m for m in all_messages if m.sender_name == name_a]
        msgs_b = [m for m in all_messages if m.sender_name == name_b]
        
        # Build profiles
        profile_a = build_character_profile(name_a, msgs_a, msgs_b)
        profile_b = build_character_profile(name_b, msgs_b, msgs_a)
        
        examples_a = select_few_shot_examples(msgs_a, count=15)
        examples_b = select_few_shot_examples(msgs_b, count=15)
        
        # Create characters in DB
        char_a_id = uuid.uuid4().hex
        char_b_id = uuid.uuid4().hex
        
        await repository.create_character(
            char_a_id, session_id, name_a,
            context.user_data["token_a"],
            profile_to_dict(profile_a), examples_a,
        )
        await repository.create_character(
            char_b_id, session_id, name_b,
            context.user_data["token_b"],
            profile_to_dict(profile_b), examples_b,
        )
        
        await repository.update_session_characters(session_id, char_a_id, char_b_id)
        
        # Save original messages to DB for future memory re-extraction
        from db.repository import save_original_messages
        await save_original_messages(session_id, parsed["messages"])
        
        # Extract memories from the full chat history
        await update.message.reply_text("⏳ Анализирую переписку и создаю память персонажей...")
        
        from parser.memory_extractor import extract_memories
        from db.repository import update_character_memories
        
        # Reconstruct ParsedMessage objects for memory extraction
        all_parsed_messages = []
        for m in parsed["messages"]:
            all_parsed_messages.append(ParsedMessage(
                id=len(all_parsed_messages),
                sender_name=m["sender_name"],
                sender_id="",
                text=m["text"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
            ))
        
        msgs_for_a = [m for m in all_parsed_messages if m.sender_name == name_a]
        msgs_for_b = [m for m in all_parsed_messages if m.sender_name == name_b]
        
        try:
            memories_a = await extract_memories(name_a, name_b, msgs_for_a)
            await update_character_memories(char_a_id, memories_a)
            
            memories_b = await extract_memories(name_b, name_a, msgs_for_b)
            await update_character_memories(char_b_id, memories_b)
            
            logger.info("Memories extracted for both characters")
        except Exception as e:
            logger.warning("Memory extraction failed (continuing without memories): %s", e)
            # Continue without memories — not fatal
        
        # Style analysis (deep profile via claude-sonnet-5)
        await update.message.reply_text("⏳ Анализирую стиль общения...")
        
        from parser.style_analyzer import analyze_style
        from db.repository import update_character_style_analyzer
        
        try:
            style_a = await analyze_style(name_a, name_b, msgs_a, all_messages=all_messages)
            await update_character_style_analyzer(char_a_id, style_a)
            
            style_b = await analyze_style(name_b, name_a, msgs_b, all_messages=all_messages)
            await update_character_style_analyzer(char_b_id, style_b)
            
            logger.info("Style analysis complete for both characters")
        except Exception as e:
            logger.warning("Style analysis failed (continuing without): %s", e)
        
        # Store session_id for later commands
        context.user_data["session_id"] = session_id
        await repository.set_user_state(user_id, STATE_READY, {"session_id": session_id})
        
        bot_a_username = context.user_data["bot_a_info"]["username"]
        bot_b_username = context.user_data["bot_b_info"]["username"]
        manager_username = context.bot.username
        
        await update.message.reply_text(
            f"✅ Персонажи созданы!\n\n"
            f"• {name_a} → @{bot_a_username}\n"
            f"• {name_b} → @{bot_b_username}\n\n"
            f"Теперь:\n"
            f"1. Создайте группу в Telegram\n"
            f"2. Добавьте @{manager_username} (менеджер), @{bot_a_username} и @{bot_b_username} в группу\n"
            f"3. Отправьте /start в группе"
        )
    except TelegramAPIError:
        await update.message.reply_text("❌ Неверный токен. Отправьте корректный токен бота.")
    except Exception as e:
        logger.exception("Error creating session")
        await update.message.reply_text(f"❌ Ошибка при создании персонажей: {e}")


async def regenerate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /regenerate — regenerate the last message."""
    user = update.effective_user
    chat = update.effective_chat

    session_id = context.user_data.get("session_id")
    if not session_id:
        sessions = await repository.get_sessions_by_user(user.id)
        for s in sessions:
            if s.get("status") in ("running", "paused", "ready"):
                session_id = s["id"]
                break

    if not session_id:
        await update.message.reply_text("Нет активной симуляции.")
        return

    count = await repository.get_message_count(session_id)
    if count == 0:
        await update.message.reply_text("Нет сообщений для перегенерации.")
        return

    all_msgs = await repository.get_messages_by_session(session_id, limit=count, offset=0)
    last_msg = all_msgs[-1]

    characters = await repository.get_characters_by_session(session_id)
    speaker = None
    other = None
    for c in characters:
        if c["id"] == last_msg["character_id"]:
            speaker = c
            other = [x for x in characters if x["id"] != c["id"]][0]
            break

    if not speaker:
        await update.message.reply_text("Не удалось найти персонажа.")
        return

    conversation_history = [
        {"sender": m.get("sender_name", ""), "text": m["text"]}
        for m in all_msgs[:-1]
    ]

    from ai.client import WaveSpeedClient
    ai_client = WaveSpeedClient()

    try:
        new_text = await ai_client.generate_message(
            character_name=speaker["name"],
            other_name=other["name"],
            profile=speaker["profile_json"],
            conversation_history=conversation_history,
            few_shot_examples=speaker["few_shot_examples"],
            memories=speaker.get("memories_json"),
        )

        await repository._execute(
            "UPDATE chat_history SET text = ? WHERE id = ?",
            (new_text, last_msg["id"]),
        )

        from telegram_proxy.sender import TelegramBotProxy
        proxy = TelegramBotProxy(speaker["token"])
        try:
            await proxy.send_message(chat.id, new_text)
        finally:
            await proxy.close()

        await update.message.reply_text("🔄 Сообщение перегенерировано.")
    except Exception as e:
        logger.exception("Regenerate failed")
        await update.message.reply_text(f"❌ Ошибка перегенерации: {e}")


async def refresh_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /refresh — re-extract memories from saved chat history."""
    user = update.effective_user
    
    # Find active session
    session_id = context.user_data.get("session_id")
    if not session_id:
        sessions = await repository.get_sessions_by_user(user.id)
        for s in sessions:
            if s.get("status") in ("idle", "ready", "running", "paused"):
                session_id = s["id"]
                break
    
    if not session_id:
        await update.message.reply_text("Нет активной сессии.")
        return
    
    # Get original messages
    from db.repository import get_original_messages, get_characters_by_session, update_character_memories
    from parser.memory_extractor import extract_memories
    from parser.export_parser import ParsedMessage
    from datetime import datetime
    
    original_msgs = await get_original_messages(session_id)
    if not original_msgs:
        await update.message.reply_text(
            "❌ Оригинальная переписка не найдена. "
            "Пройдите /import заново чтобы загрузить переписку."
        )
        return
    
    await update.message.reply_text("⏳ Перечитываю переписку и обновляю память...")
    
    # Reconstruct ParsedMessage objects
    all_messages = []
    for m in original_msgs:
        all_messages.append(ParsedMessage(
            id=len(all_messages),
            sender_name=m["sender_name"],
            sender_id="",
            text=m["text"],
            timestamp=datetime.fromisoformat(m["timestamp"]),
        ))
    
    # Get characters
    characters = await get_characters_by_session(session_id)
    if len(characters) < 2:
        await update.message.reply_text("❌ Не найдены персонажи сессии.")
        return
    
    name_a = characters[0]["name"]
    name_b = characters[1]["name"]
    
    msgs_a = [m for m in all_messages if m.sender_name == name_a]
    msgs_b = [m for m in all_messages if m.sender_name == name_b]
    
    try:
        memories_a = await extract_memories(name_a, name_b, msgs_a)
        await update_character_memories(characters[0]["id"], memories_a)
        
        memories_b = await extract_memories(name_b, name_a, msgs_b)
        await update_character_memories(characters[1]["id"], memories_b)
        
        # Style analysis
        await update.message.reply_text("⏳ Анализирую стиль общения...")
        
        from parser.style_analyzer import analyze_style
        from db.repository import update_character_style_analyzer
        
        try:
            style_a = await analyze_style(name_a, name_b, msgs_a, all_messages=all_messages)
            await update_character_style_analyzer(characters[0]["id"], style_a)
            
            style_b = await analyze_style(name_b, name_a, msgs_b, all_messages=all_messages)
            await update_character_style_analyzer(characters[1]["id"], style_b)
            
            logger.info("Style re-analysis complete")
        except Exception as e:
            logger.warning("Style re-analysis failed: %s", e)
        
        await update.message.reply_text("✅ Память персонажей обновлена!")
    except Exception as e:
        logger.exception("Memory re-extraction failed")
        await update.message.reply_text(f"❌ Ошибка: {e}")
