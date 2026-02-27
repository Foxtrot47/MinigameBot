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

# Trigger substrings — matched case-insensitively against the message text.
WORD_TRIGGER = "be the first to write the word shown in the photo"
MATH_TRIGGER = "be the first to write the result of the calculation"

# Gemini prompts — kept focused so the model returns only what we need.
WORD_PROMPT = (
    "This image is from a word-guessing game. "
    "What is the large word displayed prominently in the centre? "
    "Ignore any logos, bot names, or UI elements. "
    "Reply with only that single word in uppercase, nothing else."
)

MATH_PROMPT = (
    "This image shows a maths problem: two numbers with an operator "
    "(×, ÷, +, or −) between them. "
    "Calculate the result. "
    "Reply with only the numerical answer, nothing else. "
    "If the result is a whole number write it without decimals."
)

EMOJI_TRIGGER = "identify the emoji written in the photo"

EMOJI_PROMPT = (
    "This image shows a single emoji or symbol displayed in the centre inside a decorative frame. "
    "What is that emoji? Reply with only the emoji character itself, nothing else."
)

COUNTRY_TRIGGER = "be the first to guess the country from the flag"

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
# Gemini Vision — generic helper
# ---------------------------------------------------------------------------

async def gemini_vision(
    session: aiohttp.ClientSession,
    image_path: str,
    prompt: str,
) -> str | None:
    """
    Send *image_path* + *prompt* to Gemini and return the raw response text.

    Returns None on network error, API error, or empty response.
    Callers are responsible for cleaning up the returned string.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "temperature": 0,        # deterministic
            "maxOutputTokens": 200,  # thinking models use tokens internally
        },
    }

    url = GEMINI_URL.format(key=GEMINI_API_KEY)
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            data = await resp.json(content_type=None)
    except Exception as exc:
        log.error("Gemini request failed: %s", exc)
        return None

    if data.get("error"):
        log.error("Gemini API error: %s", data["error"].get("message"))
        return None

    try:
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        finish = data.get("candidates", [{}])[0].get("finishReason", "unknown")
        log.error("Gemini returned no text (finishReason=%s). Full response: %s", finish, data)
        return None

    return raw


def parse_word(raw: str) -> str | None:
    """Extract a single uppercase word from Gemini's response."""
    word = re.sub(r"[^A-Za-z]", "", raw).upper()
    return word or None


def parse_number(raw: str) -> str | None:
    """Extract a numeric result from Gemini's response (integer or decimal)."""
    # Keep digits, minus sign, and decimal point.
    number = re.sub(r"[^0-9\-.]", "", raw)
    return number or None


def parse_emoji(raw: str) -> str | None:
    """Return Gemini's response stripped of surrounding whitespace."""
    emoji = raw.strip()
    return emoji if emoji else None


async def click_button(message, target: str) -> bool:
    """
    Scan *message*'s inline keyboard and click the first button whose text
    contains *target*. Returns True if clicked, False if none matched.
    """
    if not message.buttons:
        log.warning("Message %d has no inline buttons.", message.id)
        return False

    for row in message.buttons:
        for button in row:
            if target in button.text:
                await button.click()
                log.info("Clicked button %r (matched %r)", button.text, target)
                return True

    available = [b.text for row in message.buttons for b in row]
    log.warning("No button found containing %r. Available: %s", target, available)
    return False

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

        text = (message.text or message.message or "").lower()

        # Determine game type from trigger phrase
        if WORD_TRIGGER in text:
            prompt, parse, game = WORD_PROMPT, parse_word, "word"
        elif MATH_TRIGGER in text:
            prompt, parse, game = MATH_PROMPT, parse_number, "math"
        elif EMOJI_TRIGGER in text:
            prompt, parse, game = EMOJI_PROMPT, parse_emoji, "emoji"
        elif COUNTRY_TRIGGER in text:
            prompt, parse, game = None, None, "country"  # prompt built dynamically
        else:
            return

        if message.media is None:
            log.warning("Trigger message %d has no media — skipping.", message.id)
            return

        # Country game: build a dynamic prompt from capital hint + button options.
        if game == "country":
            capital_match = re.search(
                r"capital is ([^\.\n!]+)", message.text or "", re.IGNORECASE
            )
            capital = capital_match.group(1).strip() if capital_match else None
            options = ", ".join(
                b.text.lstrip("»").strip()
                for row in (message.buttons or [])
                for b in row
            )
            capital_hint = f"The capital city is {capital}. " if capital else ""
            prompt = (
                "This image shows a country flag. "
                f"{capital_hint}"
                f"Which of these countries does it belong to: {options}? "
                "Reply with only the exact country name from the list, nothing else."
            )
            log.info(
                "Game=country detected in message %d (capital=%r, options=%r).",
                message.id, capital, options,
            )
        else:
            log.info("Game=%s detected in message %d. Processing image...", game, message.id)

        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, message)
            if tmp_path is None:
                return

            raw = await gemini_vision(http, tmp_path, prompt)
            if not raw:
                log.error("Gemini returned nothing for message %d.", message.id)
                return

            answer = (parse_emoji if game == "country" else parse)(raw)
            if not answer:
                log.error("Could not parse answer from %r (message %d).", raw, message.id)
                return

            if game in ("emoji", "country"):
                await click_button(message, answer)
            else:
                log.info("Replying to message %d with: %r", message.id, answer)
                await message.reply(answer)

        except Exception as exc:
            log.error("Error handling message %d: %s", message.id, exc, exc_info=True)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    # -----------------------------------------------------------------------
    # Test commands — delete instantly, print result to console only.
    #   .ocr   → reply to a word-game image
    #   .calc  → reply to a math-game image
    # -----------------------------------------------------------------------
    async def _test(event: events.NewMessage.Event, prompt: str, parse, label: str) -> None:
        await event.delete()
        if not event.is_reply:
            log.info("[%s] Use this as a reply to a game message with an image.", label)
            return
        replied = await event.get_reply_message()
        if replied is None or replied.media is None:
            log.info("[%s] Replied message has no image.", label)
            return
        log.info("[%s] Downloading image from message %d...", label, replied.id)
        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, replied)
            if tmp_path is None:
                log.info("[%s] Could not download image.", label)
                return
            raw = await gemini_vision(http, tmp_path, prompt)
            answer = parse(raw) if raw else None
            if answer:
                log.info("[%s] Result: %r", label, answer)
            else:
                log.info("[%s] Could not extract an answer (raw=%r).", label, raw)
        except Exception as exc:
            log.error("[%s] Error: %s", label, exc, exc_info=True)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.ocr$"))
    async def test_ocr(event: events.NewMessage.Event) -> None:
        await _test(event, WORD_PROMPT, parse_word, ".ocr")

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.calc$"))
    async def test_calc(event: events.NewMessage.Event) -> None:
        await _test(event, MATH_PROMPT, parse_number, ".calc")

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.emoji$"))
    async def test_emoji(event: events.NewMessage.Event) -> None:
        await _test(event, EMOJI_PROMPT, parse_emoji, ".emoji")

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.country$"))
    async def test_country(event: events.NewMessage.Event) -> None:
        await event.delete()
        if not event.is_reply:
            log.info("[.country] Use this as a reply to a country game message.")
            return
        replied = await event.get_reply_message()
        if replied is None or replied.media is None:
            log.info("[.country] Replied message has no image.")
            return
        msg_text = replied.text or replied.message or ""
        capital_match = re.search(r"capital is ([^\.\n!]+)", msg_text, re.IGNORECASE)
        capital = capital_match.group(1).strip() if capital_match else None
        options = ", ".join(
            b.text.lstrip("»").strip()
            for row in (replied.buttons or [])
            for b in row
        )
        capital_hint = f"The capital city is {capital}. " if capital else ""
        prompt = (
            "This image shows a country flag. "
            f"{capital_hint}"
            f"Which of these countries does it belong to: {options}? "
            "Reply with only the exact country name from the list, nothing else."
        )
        log.info("[.country] capital=%r, options=%r", capital, options)
        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, replied)
            if tmp_path is None:
                log.info("[.country] Could not download image.")
                return
            raw = await gemini_vision(http, tmp_path, prompt)
            answer = parse_emoji(raw) if raw else None  # just strips whitespace
            if answer:
                log.info("[.country] Result: %r", answer)
            else:
                log.info("[.country] Could not extract an answer (raw=%r).", raw)
        except Exception as exc:
            log.error("[.country] Error: %s", exc, exc_info=True)
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
