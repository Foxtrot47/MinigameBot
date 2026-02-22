"""
Telegram userbot — Mini-game word solver.

Flow:
  1. Start Telethon client (prompts for OTP on first run, then uses .session).
  2. Listen for new messages in TARGET_CHANNEL from TARGET_BOT.
  3. When the trigger message with a spoiler image arrives:
       - Download the image to a temp file
       - Send it to the Gemini Vision API with a focused prompt
       - Reply with the recognised word immediately
"""

import asyncio
import base64
import logging
import os
import re
import tempfile
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mingame_bot")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

API_ID: int = int(os.environ["API_ID"])
API_HASH: str = os.environ["API_HASH"]
PHONE: str = os.environ["PHONE"]
TARGET_CHANNEL: str = os.environ["TARGET_CHANNEL"]
TARGET_BOT: str = os.environ["TARGET_BOT"]
SESSION_NAME: str = os.getenv("SESSION_NAME", "mingame_session")
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# Case-insensitive substring that identifies the trigger message.
TRIGGER_PHRASE = "be the first to write the word shown in the photo"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key={key}"
)


def _parse_entity(value: str) -> str | int:
    """Return int if value is a numeric ID, otherwise return the string as-is."""
    stripped = value.strip().lstrip("@")
    try:
        return int(stripped)
    except ValueError:
        return value.strip()


CHANNEL_ENTITY = _parse_entity(TARGET_CHANNEL)
BOT_ENTITY = _parse_entity(TARGET_BOT)

# ---------------------------------------------------------------------------
# Vision OCR via Gemini
# ---------------------------------------------------------------------------

async def gemini_ocr(session: aiohttp.ClientSession, image_path: str) -> str | None:
    """
    Send *image_path* to Gemini 2.0 Flash Vision and return the game word.

    Gemini understands the image semantically — no bounding-box tricks needed.
    The prompt instructs it to ignore logos/UI chrome and return only the
    prominent word shown in the centre of the image.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64,
                    }
                },
                {
                    "text": (
                        "This image is from a word-guessing game. "
                        "What is the large word displayed prominently in the centre? "
                        "Ignore any logos, bot names, or UI elements. "
                        "Reply with only that single word in uppercase, nothing else."
                    )
                },
            ]
        }],
        "generationConfig": {
            "temperature": 0,        # deterministic — no creativity needed
            "maxOutputTokens": 200,  # thinking models use extra tokens internally
        },
    }

    url = GEMINI_URL.format(key=GEMINI_API_KEY)
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json(content_type=None)
    except Exception as exc:
        log.error("Gemini request failed: %s", exc)
        return None

    # Extract the text from the response
    try:
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        log.error("Unexpected Gemini response structure: %s | data: %s", exc, data)
        return None

    # Strip anything that isn't a letter (in case Gemini adds punctuation)
    word = re.sub(r"[^A-Za-z]", "", raw).upper()

    if not word:
        log.warning("Gemini returned no usable text. Raw: %r", raw)
        return None

    log.info("Gemini OCR result: %r", word)
    return word

# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

async def download_photo(client: TelegramClient, message) -> str | None:
    """
    Download the photo from *message* to a temp .jpg file.

    Returns the file path (caller must delete it), or None on failure.
    Spoiler images are identical to normal photos at the MTProto layer —
    no special handling needed; Telethon downloads them transparently.
    """
    media = message.media
    if media is None:
        return None

    is_photo = isinstance(media, MessageMediaPhoto)
    is_image_doc = (
        isinstance(media, MessageMediaDocument)
        and getattr(media.document, "mime_type", "").startswith("image/")
    )

    if not (is_photo or is_image_doc):
        log.warning("Message %d: unsupported media type %s", message.id, type(media).__name__)
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        downloaded = await client.download_media(message.media, file=tmp_path)
        log.info("Image downloaded to %s", downloaded)
        return str(downloaded)
    except Exception as exc:
        log.error("Failed to download media from message %d: %s", message.id, exc)
        Path(tmp_path).unlink(missing_ok=True)
        return None

# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

async def resolve_peer_ids(client: TelegramClient) -> tuple[int, int]:
    """Resolve channel and bot entities to numeric peer IDs."""
    channel = await client.get_entity(CHANNEL_ENTITY)
    bot = await client.get_entity(BOT_ENTITY)
    log.info("Channel: %r → id=%d", TARGET_CHANNEL, channel.id)
    log.info("Bot:     %r → id=%d", TARGET_BOT, bot.id)
    return channel.id, bot.id

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(phone=PHONE)
    me = await client.get_me()
    log.info("Connected as %s (id=%d)", me.username or me.first_name, me.id)

    channel_id, bot_id = await resolve_peer_ids(client)

    # Single aiohttp session shared across all handlers (reuses connections).
    http = aiohttp.ClientSession()

    @client.on(events.NewMessage(chats=channel_id))
    async def handle(event: events.NewMessage.Event) -> None:
        message = event.message

        # Sender filter — broadcast channel posts have sender_id=None (posted as
        # the channel itself). We accept None or the exact bot_id. Using
        # from_users= in the decorator is unreliable for broadcast channels.
        sender_id = message.sender_id
        if sender_id is not None and sender_id != bot_id:
            return

        # Trigger text check
        text = (message.text or message.message or "").lower()
        if TRIGGER_PHRASE not in text:
            return

        if message.media is None:
            log.warning("Trigger message %d has no media — skipping.", message.id)
            return

        log.info("Trigger detected in message %d. Processing image...", message.id)

        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, message)
            if tmp_path is None:
                return

            word = await gemini_ocr(http, tmp_path)

            if not word:
                log.error("Could not extract a word from message %d image.", message.id)
                return

            log.info("Replying to message %d with: %r", message.id, word)
            await message.reply(word)

        except Exception as exc:
            log.error("Error handling message %d: %s", message.id, exc, exc_info=True)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    # -----------------------------------------------------------------------
    # Test command: reply to any message with .ocr
    # Deletes your command instantly; prints Gemini result to console only.
    # -----------------------------------------------------------------------
    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.ocr$"))
    async def test_ocr(event: events.NewMessage.Event) -> None:
        await event.delete()  # remove the command from chat immediately

        if not event.is_reply:
            log.info("[.ocr] Use this as a reply to a message that contains an image.")
            return

        replied = await event.get_reply_message()

        if replied is None or replied.media is None:
            log.info("[.ocr] Replied message has no image.")
            return

        log.info("[.ocr] Downloading image from message %d...", replied.id)

        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, replied)
            if tmp_path is None:
                log.info("[.ocr] Could not download image.")
                return

            word = await gemini_ocr(http, tmp_path)

            if word:
                log.info("[.ocr] Result: %r", word)
            else:
                log.info("[.ocr] No word could be extracted from the image.")

        except Exception as exc:
            log.error("[.ocr] Error: %s", exc, exc_info=True)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    log.info(
        "Bot running — watching channel id=%d for bot id=%d. Press Ctrl+C to stop.",
        channel_id,
        bot_id,
    )
    try:
        await client.run_until_disconnected()
    finally:
        await http.close()


if __name__ == "__main__":
    asyncio.run(main())
