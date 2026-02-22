"""
Telegram userbot — Mini-game word solver.

Flow:
  1. Start Telethon client (prompts for OTP on first run, then uses .session).
  2. Initialise EasyOCR reader once at startup (expensive model load, amortised).
  3. Listen for new messages in TARGET_CHANNEL from TARGET_BOT.
  4. When the trigger message with a spoiler image arrives:
       - Download the image to a temp file
       - Run OCR (in a thread so the event loop stays unblocked)
       - Reply with the recognised word immediately
"""

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

import easyocr
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageOps
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

# Case-insensitive substring that identifies the trigger message.
TRIGGER_PHRASE = "be the first to write the word shown in the photo"


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
# OCR
# ---------------------------------------------------------------------------

def init_ocr() -> easyocr.Reader:
    """Load EasyOCR once at startup. First run downloads model weights (~300 MB)."""
    log.info("Initialising EasyOCR — may take a few seconds on first run...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    log.info("EasyOCR ready.")
    return reader


def preprocess(image_path: str) -> np.ndarray:
    """
    Prepare the image for OCR:
      1. Grayscale — removes color noise from decorative fonts/gradients.
      2. Auto-invert — ensures text is dark on a light background.
         Game images often use light/white text on dark backgrounds;
         EasyOCR performs better on dark-on-light.
      3. Contrast boost — sharpens faint or anti-aliased edges.
      4. Upscale — helps with small images (EasyOCR works best >= 32px tall).

    Returns a numpy array (H×W×3 RGB) that EasyOCR accepts directly,
    avoiding a redundant disk write.
    """
    img = Image.open(image_path).convert("RGB")

    gray = img.convert("L")

    # Auto-invert: if the image is predominantly dark, the text is likely light.
    # Median pixel value < 128 → dark background → invert so text becomes dark.
    median = float(np.median(np.array(gray)))
    if median < 128:
        gray = ImageOps.invert(gray)

    # Contrast boost — 2.5x is aggressive but effective on styled text.
    gray = ImageEnhance.Contrast(gray).enhance(2.5)

    # Upscale small images to at least 64px tall for better glyph detection.
    if gray.height < 64:
        scale = 64 / gray.height
        gray = gray.resize(
            (int(gray.width * scale), int(gray.height * scale)),
            Image.LANCZOS,
        )

    # Convert back to RGB (EasyOCR expects 3-channel input).
    return np.array(gray.convert("RGB"))


def extract_word(reader: easyocr.Reader, image_path: str) -> str | None:
    """
    Run OCR on *image_path* and return the most likely single word.

    EasyOCR returns [(bounding_box, text, confidence), ...].
    We filter by confidence, strip non-alpha characters, and return the
    segment with the highest confidence score.
    """
    try:
        processed = preprocess(image_path)
        results = reader.readtext(processed, detail=1)
    except Exception as exc:
        log.error("EasyOCR error: %s", exc)
        return None

    if not results:
        log.warning("EasyOCR returned no results for %s", image_path)
        return None

    log.debug("Raw OCR results: %s", results)

    # Each result: (bounding_box, text, confidence)
    # bounding_box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (4 corners).
    # We pick by largest bounding box area, not confidence — the target word
    # is always the biggest text in the image; logo/watermark text is small.
    candidates: list[tuple[str, float, float]] = []  # (text, confidence, area)
    for bbox, text, confidence in results:
        cleaned = re.sub(r"[^A-Za-z]", "", text).strip()
        if not cleaned or confidence < 0.2:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        candidates.append((cleaned, confidence, area))

    if not candidates:
        log.warning("No usable text after cleaning. Raw results: %s", results)
        return None

    best_text, best_conf, best_area = max(candidates, key=lambda x: x[2])
    log.info("OCR result: %r (confidence=%.2f, area=%.0f)", best_text, best_conf, best_area)
    return best_text

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
    # Initialise OCR before connecting so model loading doesn't delay first reply
    ocr_reader = init_ocr()

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(phone=PHONE)
    me = await client.get_me()
    log.info("Connected as %s (id=%d)", me.username or me.first_name, me.id)

    channel_id, bot_id = await resolve_peer_ids(client)

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

        # Must have an image
        if message.media is None:
            log.warning("Trigger message %d has no media — skipping.", message.id)
            return

        log.info("Trigger detected in message %d. Processing image...", message.id)

        tmp_path: str | None = None
        try:
            tmp_path = await download_photo(client, message)
            if tmp_path is None:
                return

            # OCR is synchronous/CPU-bound — run in thread pool to keep loop free
            loop = asyncio.get_event_loop()
            word = await loop.run_in_executor(None, extract_word, ocr_reader, tmp_path)

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
    # Deletes your command instantly; prints OCR result to console only.
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

            loop = asyncio.get_event_loop()
            word = await loop.run_in_executor(None, extract_word, ocr_reader, tmp_path)

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
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
