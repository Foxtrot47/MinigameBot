"""
Telegram userbot — Mini-game solver.

Supports four game types:
  - word    : identify the large word in the image
  - math    : calculate the result of the shown equation
  - emoji   : identify the emoji and click the matching button
  - country : identify the flag country and click the matching button

OCR backends (set OCR_BACKEND in .env):
  - gemini  (default): cloud vision via Gemini — accurate, no local resources
  - local             : EasyOCR running on-device — no API calls for word/math
                        (emoji + country always fall back to Gemini)
"""

import asyncio
import base64
import logging
import operator as _op
import os
import re
import tempfile
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto

# Optional — only needed when OCR_BACKEND=local
try:
    import easyocr
    import numpy as np
    from PIL import Image, ImageEnhance, ImageOps
    _LOCAL_DEPS = True
except ImportError:
    _LOCAL_DEPS = False

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
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OCR_BACKEND: str = os.getenv("OCR_BACKEND", "gemini").lower()  # "gemini" | "local"

# ---------------------------------------------------------------------------
# Triggers + prompts
# ---------------------------------------------------------------------------
WORD_TRIGGER    = "be the first to write the word shown in the photo"
MATH_TRIGGER    = "be the first to write the result of the calculation"
EMOJI_TRIGGER   = "identify the emoji written in the photo"
COUNTRY_TRIGGER = "be the first to guess the country from the flag"

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
EMOJI_PROMPT = (
    "This image shows a single emoji or symbol displayed in the centre "
    "inside a decorative frame. "
    "What is that emoji? Reply with only the emoji character itself, nothing else."
)
# Country prompt is built dynamically (needs capital + button options).

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key={key}"
)

# ---------------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------------

def _parse_entity(value: str) -> str | int:
    stripped = value.strip().lstrip("@")
    try:
        return int(stripped)
    except ValueError:
        return value.strip()

CHANNEL_ENTITY = _parse_entity(TARGET_CHANNEL)
BOT_ENTITY     = _parse_entity(TARGET_BOT)

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_word(raw: str) -> str | None:
    word = re.sub(r"[^A-Za-z]", "", raw).lower()
    return word or None

def parse_number(raw: str) -> str | None:
    number = re.sub(r"[^0-9\-.]", "", raw)
    return number or None

def parse_emoji(raw: str) -> str | None:
    emoji = raw.strip()
    return emoji if emoji else None

# ---------------------------------------------------------------------------
# Local OCR (EasyOCR) — only used when OCR_BACKEND=local
# ---------------------------------------------------------------------------

_MATH_OPS: dict = {
    "×": _op.mul, "x": _op.mul, "X": _op.mul, "*": _op.mul,
    "÷": _op.truediv, "/": _op.truediv,
    "+": _op.add,
    "−": _op.sub, "-": _op.sub,
}


def init_local_ocr():
    if not _LOCAL_DEPS:
        raise RuntimeError(
            "OCR_BACKEND=local requires: "
            "pip install easyocr Pillow  "
            "(and optionally: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu)"
        )
    log.info("Initialising EasyOCR (may take a few seconds on first run)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    log.info("EasyOCR ready.")
    return reader


def _preprocess(image_path: str) -> "np.ndarray":
    """Grayscale + auto-invert + contrast boost + upscale for cleaner OCR."""
    img  = Image.open(image_path).convert("RGB")
    gray = img.convert("L")
    if float(np.median(np.array(gray))) < 128:
        gray = ImageOps.invert(gray)
    gray = ImageEnhance.Contrast(gray).enhance(2.5)
    if gray.height < 64:
        scale = 64 / gray.height
        gray = gray.resize(
            (int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS
        )
    return np.array(gray.convert("RGB"))


def _safe_calc(texts: list[str]) -> str | None:
    """Extract two numbers and an operator from OCR segments and compute."""
    nums: list[float] = []
    found_op = None
    for t in texts:
        t = t.strip()
        try:
            nums.append(float(t))
            continue
        except ValueError:
            pass
        if t in _MATH_OPS:
            found_op = t
    if len(nums) == 2 and found_op:
        try:
            result = _MATH_OPS[found_op](nums[0], nums[1])
        except ZeroDivisionError:
            return "0"
        return str(int(result)) if result == int(result) else str(round(result, 4))
    return None


def local_extract(reader, image_path: str, game: str) -> str | None:
    """Run EasyOCR on *image_path* and return the answer for *game*."""
    processed = _preprocess(image_path)
    results   = reader.readtext(processed, detail=1)

    if not results:
        log.warning("[local] EasyOCR returned no results.")
        return None

    if game == "word":
        candidates: list[tuple[str, float]] = []
        for bbox, text, conf in results:
            cleaned = re.sub(r"[^A-Za-z]", "", text).strip()
            if not cleaned or conf < 0.2:
                continue
            xs   = [p[0] for p in bbox]
            ys   = [p[1] for p in bbox]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            candidates.append((cleaned.lower(), area))
        if not candidates:
            return None
        best, _ = max(candidates, key=lambda x: x[1])
        log.info("[local] word result: %r", best)
        return best

    if game == "math":
        texts  = [text for _, text, conf in results if conf >= 0.2]
        answer = _safe_calc(texts)
        log.info("[local] math result: %r (from segments %s)", answer, texts)
        return answer

    return None  # emoji + country not supported locally

# ---------------------------------------------------------------------------
# Gemini vision
# ---------------------------------------------------------------------------

async def gemini_vision(
    session: aiohttp.ClientSession,
    image_path: str,
    prompt: str,
) -> str | None:
    """Send image + prompt to Gemini and return the raw response text."""
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY is not set.")
        return None

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 200},
    }

    url = GEMINI_URL.format(key=GEMINI_API_KEY)
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            data = await resp.json(content_type=None)
    except Exception as exc:
        log.error("Gemini request failed: %s", exc)
        return None

    if data.get("error"):
        log.error("Gemini API error: %s", data["error"].get("message"))
        return None

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        finish = data.get("candidates", [{}])[0].get("finishReason", "unknown")
        log.error("Gemini returned no text (finishReason=%s). Response: %s", finish, data)
        return None

# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

async def call_vision(
    http: aiohttp.ClientSession,
    reader,                     # EasyOCR reader or None
    loop: asyncio.AbstractEventLoop,
    image_path: str,
    game: str,
    prompt: str,
    parse_fn,
) -> str | None:
    """
    Route to the configured OCR backend and return the final parsed answer.

    Gemini mode  → gemini_vision → parse_fn
    Local mode   → local_extract (word/math only; emoji+country use Gemini)
    """
    if OCR_BACKEND == "local" and game in ("word", "math"):
        raw = await loop.run_in_executor(None, local_extract, reader, image_path, game)
        return raw  # already parsed by local_extract
    else:
        if OCR_BACKEND == "local":
            log.info("[local] %s game — falling back to Gemini.", game)
        raw = await gemini_vision(http, image_path, prompt)
        return parse_fn(raw) if raw else None

# ---------------------------------------------------------------------------
# Button clicking
# ---------------------------------------------------------------------------

async def click_button(message, target: str) -> bool:
    """Click the first inline button whose text contains *target*."""
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
    """Download the photo from *message* to a temp .jpg file."""
    media = message.media
    if media is None:
        return None
    is_photo     = isinstance(media, MessageMediaPhoto)
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
    channel = await client.get_entity(CHANNEL_ENTITY)
    bot     = await client.get_entity(BOT_ENTITY)
    log.info("Channel: %r → id=%d", TARGET_CHANNEL, channel.id)
    log.info("Bot:     %r → id=%d", TARGET_BOT, bot.id)
    return channel.id, bot.id

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    log.info("OCR backend: %s", OCR_BACKEND.upper())

    reader = init_local_ocr() if OCR_BACKEND == "local" else None
    loop   = asyncio.get_event_loop()

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(phone=PHONE)
    me = await client.get_me()
    log.info("Connected as %s (id=%d)", me.username or me.first_name, me.id)

    channel_id, bot_id = await resolve_peer_ids(client)

    http = aiohttp.ClientSession()

    # -------------------------------------------------------------------
    # Game handler
    # -------------------------------------------------------------------
    @client.on(events.NewMessage(chats=channel_id))
    async def handle(event: events.NewMessage.Event) -> None:
        message   = event.message
        sender_id = message.sender_id
        if sender_id is not None and sender_id != bot_id:
            return

        text = (message.text or message.message or "").lower()

        if WORD_TRIGGER in text:
            prompt, parse_fn, game = WORD_PROMPT, parse_word, "word"
        elif MATH_TRIGGER in text:
            prompt, parse_fn, game = MATH_PROMPT, parse_number, "math"
        elif EMOJI_TRIGGER in text:
            prompt, parse_fn, game = EMOJI_PROMPT, parse_emoji, "emoji"
        elif COUNTRY_TRIGGER in text:
            prompt, parse_fn, game = None, parse_emoji, "country"
        else:
            return

        if message.media is None:
            log.warning("Trigger message %d has no media — skipping.", message.id)
            return

        # Build the dynamic country prompt
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
                "Game=country | message=%d | capital=%r | options=%r",
                message.id, capital, options,
            )
        else:
            log.info("Game=%s | message=%d", game, message.id)

        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, message)
            if tmp_path is None:
                return

            answer = await call_vision(http, reader, loop, tmp_path, game, prompt, parse_fn)
            if not answer:
                log.error("No answer for message %d.", message.id)
                return

            if game in ("emoji", "country"):
                await click_button(message, answer)
            else:
                log.info("Replying with: %r", answer)
                await message.reply(answer)

        except Exception as exc:
            log.error("Error handling message %d: %s", message.id, exc, exc_info=True)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    # -------------------------------------------------------------------
    # Test commands (console only, no chat output)
    # -------------------------------------------------------------------
    async def _test(
        event: events.NewMessage.Event,
        game: str,
        prompt: str | None,
        parse_fn,
        label: str,
    ) -> None:
        await event.delete()
        if not event.is_reply:
            log.info("[%s] Use as a reply to a game message with an image.", label)
            return
        replied = await event.get_reply_message()
        if replied is None or replied.media is None:
            log.info("[%s] Replied message has no image.", label)
            return

        # Build country prompt dynamically
        if game == "country":
            msg_text     = replied.text or replied.message or ""
            capital_match = re.search(r"capital is ([^\.\n!]+)", msg_text, re.IGNORECASE)
            capital      = capital_match.group(1).strip() if capital_match else None
            options      = ", ".join(
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
            log.info("[%s] capital=%r | options=%r", label, capital, options)

        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, replied)
            if tmp_path is None:
                log.info("[%s] Could not download image.", label)
                return
            answer = await call_vision(http, reader, loop, tmp_path, game, prompt, parse_fn)
            if answer:
                log.info("[%s] Result: %r", label, answer)
            else:
                log.info("[%s] No answer extracted.", label)
        except Exception as exc:
            log.error("[%s] Error: %s", label, exc, exc_info=True)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.ocr$"))
    async def test_ocr(event):
        await _test(event, "word", WORD_PROMPT, parse_word, ".ocr")

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.calc$"))
    async def test_calc(event):
        await _test(event, "math", MATH_PROMPT, parse_number, ".calc")

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.emoji$"))
    async def test_emoji(event):
        await _test(event, "emoji", EMOJI_PROMPT, parse_emoji, ".emoji")

    @client.on(events.NewMessage(outgoing=True, pattern=r"^\.country$"))
    async def test_country(event):
        await _test(event, "country", None, parse_emoji, ".country")

    log.info(
        "Bot running — channel=%d | bot=%d | backend=%s. Ctrl+C to stop.",
        channel_id, bot_id, OCR_BACKEND.upper(),
    )
    try:
        await client.run_until_disconnected()
    finally:
        await http.close()


if __name__ == "__main__":
    asyncio.run(main())
