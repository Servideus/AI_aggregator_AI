from __future__ import annotations

import asyncio
import os
import sys
import json
import re
import datetime as dt
from typing import List
from dataclasses import dataclass

import yaml
from telethon import TelegramClient
from telethon.tl.custom.message import Message

from llm_adapter import complete_json, LLMError


# ==========================
#  НАСТРОЙКИ
# ==========================

SOURCES_PATH = os.getenv("SOURCES_PATH", "sources.yml")
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "user_session_llm")
DIGEST_PEER = os.getenv("DIGEST_PEER")  # Optional destination chat or channel

# Сколько часов назад собираем новости
FETCH_HOURS = int(os.getenv("FETCH_HOURS", "24"))

# Configurable offset for digest dates
LOCAL_TZ = dt.timezone(dt.timedelta(hours=float(os.getenv("UTC_OFFSET_HOURS", "0"))))

# Ограничения по длине текста для LLM
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "4000"))

# Фильтры: бренды и мультилинки
RU_BRAND_PATTERN = re.compile(
    r"\b(яндекс|сбер|тинькофф|авито|мтс|ozon|озон|wildberries|вб|битрикс|bitrix24|точка банк)\b",
    re.IGNORECASE,
)

MULTILINK_PATTERN = re.compile(
    r"(подборка ссылок|дайджест|ежедневная сводка|много интересного по ссылке)",
    re.IGNORECASE,
)

MULTILINK_MAX_LINKS = int(os.getenv("MULTILINK_MAX_LINKS", "4"))

DEFAULT_LLM_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

# Повторы при временной недоступности LLM: ждем 5 минут между попытками,
# но не дольше получаса суммарно, иначе зависший запуск заблокирует следующие.
LLM_RETRY_SECONDS = int(os.getenv("LLM_RETRY_SECONDS", "300"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))  # 0 = без лимита
LLM_MODELS = [
    model.strip()
    for model in os.getenv("LLM_MODELS", ",".join(DEFAULT_LLM_MODELS)).split(",")
    if model.strip()
]
if not LLM_MODELS:
    LLM_MODELS = DEFAULT_LLM_MODELS

RETRYABLE_LLM_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
)

PERMANENT_LLM_MARKERS = (
    "400",
    "401",
    "403",
    "404",
    "API_KEY_INVALID",
    "INVALID_ARGUMENT",
    "MODEL_NOT_FOUND",
    "NOT_FOUND",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
)


# ==========================
#  УТИЛИТЫ ДЛЯ ТЕКСТА
# ==========================

def clean_text(text: str) -> str:
    """
    Простая чистка текста: убираем лишние пробелы, проверяем пустоту.
    """
    if not text:
        return ""
    # Убираем артефакты типа невидимых символов и лишних пробелов
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(msg: Message) -> str:
    raw = msg.message or msg.raw_text or ""
    return raw or ""


def guess_title(text: str) -> str:
    """
    Берем первую строку поста без механического обрезания.
    Это идет ТОЛЬКО во входной JSON для модели.
    В финальный дайджест попадает title из ответа модели.
    """
    if not text:
        return "Без названия"
    first = text.splitlines()[0].strip()
    for pref in ("- ", "* ", "• "):
        if first.startswith(pref):
            first = first[len(pref):].strip()
            break
    return first[:200] if first else "Без названия"


def count_links(text: str) -> int:
    return len(re.findall(r"https?://", text))


def is_ru_brand_post(text: str) -> bool:
    """
    Грубый фильтр рекламных интеграций российских брендов.
    """
    if RU_BRAND_PATTERN.search(text):
        return True
    return False


def is_multilink_digest(text: str) -> bool:
    """
    Фильтр "солянок" с кучей ссылок.
    """
    if MULTILINK_PATTERN.search(text):
        return True
    if count_links(text) > MULTILINK_MAX_LINKS:
        return True
    return False


def is_retryable_llm_error(error: LLMError) -> bool:
    text = str(error).upper()
    if any(marker in text for marker in RETRYABLE_LLM_MARKERS):
        return True
    if any(marker in text for marker in PERMANENT_LLM_MARKERS):
        return False
    return True


def limit_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# ==========================
#  ЗАГРУЗКА ИСТОЧНИКОВ
# ==========================

def load_sources(path: str = SOURCES_PATH) -> List[Channel]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[error] missing {path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[error] failed to parse {path}: {e}", file=sys.stderr)
        return []

    # Поддерживаем и формат с корнем dict, и список
    if isinstance(data, dict):
        items = data.get("sources") or data.get("channels") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    channels: List[Channel] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # можно отключать канал флагом enabled: false
        if not item.get("enabled", True):
            continue

        cid = item.get("id") or item.get("username") or item.get("channel")
        if not cid:
            continue
        title = item.get("title") or item.get("name") or str(cid)

        channels.append(Channel(id=str(cid), title=str(title)))

    print(f"[sources] loaded {len(channels)} channels")
    return channels


# ==========================
#  ВРЕМЯ И МОДЕЛИ ДАННЫХ
# ==========================

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def now_local() -> dt.datetime:
    return now_utc().astimezone(LOCAL_TZ)


@dataclass
class Channel:
    id: str
    title: str


@dataclass
class Post:
    channel_id: str
    channel_title: str
    message_id: int
    date: dt.datetime
    url: str
    text: str


# ==========================
#  ВСПОМОГАТЕЛЬНОЕ ДЛЯ ENV
# ==========================

def get_env(*names: str, required: bool = True, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    if required:
        raise RuntimeError("Нужно установить одну из переменных: " + ", ".join(names))
    return default


# ==========================
#  ДАЙДЖЕСТЫ ПРОШЛЫХ ДНЕЙ
# ==========================

def _load_digest_for_date(date: dt.date) -> str | None:
    """
    Загружает текст дайджеста за указанный календарный день (локальное время).
    Имя файла: digest_YYYYMMDD.txt. Если файл не найден или не читается, возвращает None.
    """
    fname = f"digest_{date.strftime('%Y%m%d')}.txt"
    try:
        with open(fname, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[warn] failed to load digest file {fname}: {e}", file=sys.stderr)
        return None


def load_yesterday_digest() -> str | None:
    """
    Загружает текст дайджеста за предыдущий календарный день (относительно локального времени).
    Если файла нет, возвращает None.
    """
    y_date = (now_local() - dt.timedelta(days=1)).date()
    return _load_digest_for_date(y_date)


# ==========================
#  TELEGRAM FETCH
# ==========================

async def fetch_posts(
    client: TelegramClient,
    channels: List[Channel],
    since_utc: dt.datetime,
) -> List[Post]:
    posts: List[Post] = []

    for ch in channels:
        print(f"[{ch.id}] fetch...")
        try:
            async for msg in client.iter_messages(
                ch.id,
                offset_date=since_utc,
                reverse=True,
            ):
                if not isinstance(msg, Message):
                    continue
                if not msg.date:
                    continue

                # Telethon выдает локальное время, приводим к UTC
                msg_dt = msg.date
                if msg_dt.tzinfo is None:
                    msg_dt = msg_dt.replace(tzinfo=LOCAL_TZ)
                msg_dt_utc = msg_dt.astimezone(dt.UTC)

                if msg_dt_utc < since_utc:
                    continue

                raw_text = extract_text(msg)
                text = clean_text(raw_text)
                if not text:
                    continue

                if is_ru_brand_post(text):
                    continue
                if is_multilink_digest(text):
                    continue

                url = f"https://t.me/{ch.id}/{msg.id}"
                posts.append(
                    Post(
                        channel_id=ch.id,
                        channel_title=ch.title,
                        message_id=msg.id,
                        date=msg_dt_utc,
                        url=url,
                        text=text,
                    )
                )
        except Exception as e:
            print(f"[error] failed to fetch from {ch.id}: {e}", file=sys.stderr)

    print(f"[total] fetched={len(posts)} after filters (RU brands + multilink)")
    return posts


# ==========================
#  LLM: ВЫБОР ТОП-НОВОСТЕЙ
# ==========================

async def llm_select_top10(posts: List[Post], yesterday_digest: str | None = None) -> tuple[str, int]:
    """
    Берем сырые посты, передаем их в complete_json и получаем JSON вида:
    [
      {"id": int, "title": str, "summary": str},
      ...
    ]
    Берем максимум 10 валидных объектов (но не добиваем до 10 искусственно).
    Возвращаем (markdown, N), где N — реальное количество новостей.
    """
    MAX_TOTAL_CHARS = 4096  # Telegram limit

    if not posts:
        return "Нет новостей.", 0

    # Более свежие выше
    posts_sorted = sorted(posts, key=lambda p: p.date, reverse=True)

    # Кандидаты для модели
    items = []
    for i, p in enumerate(posts_sorted, start=1):
        snippet = p.text.strip()
        if len(snippet) > MAX_TEXT_CHARS:
            snippet = snippet[:MAX_TEXT_CHARS] + "\n\n[текст обрезан]"
        items.append(
            {
                "id": i,
                "channel": p.channel_title,
                "source_title": guess_title(p.text),
                "text": snippet,
                "url": p.url,
                "time_utc": p.date.isoformat(),
            }
        )

    today = now_local().strftime("%d-%m-%Y")

    # Рассчитываем примерный лимит на summary
    # Заголовок "Топ X новостей · DD-MM-YYYY\n\n" ~30 символов
    # Каждая новость: "N. [title](url)\nsummary\n\n"
    # url в среднем ~40 символов, title ~80, разметка ~10
    # Итого на одну новость без summary: ~140 символов
    MAX_ITEMS = 10
    HEADER_RESERVE = 50  # заголовок дайджеста
    PER_ITEM_OVERHEAD = 140  # title + url + разметка

    available_for_summaries = MAX_TOTAL_CHARS - HEADER_RESERVE - (MAX_ITEMS * PER_ITEM_OVERHEAD)
    max_summary_length = max(100, available_for_summaries // MAX_ITEMS)  # минимум 100 символов

    system_prompt = (
        "Ты строгий редактор ежедневного AI-дайджеста для технической аудитории.\n\n"
        "Тебе переданы посты за последние 24 часа в формате JSON. Каждый объект содержит поля:\n"
        "id, channel, source_title, text, url, time_utc.\n\n"
        "ТВОЯ ЗАДАЧА:\n"
        "1) Выбрать НЕ БОЛЕЕ 10 самых важных новостей об ИИ.\n"
        "2) Удалить ВСЕ повторы одной и той же новости в текущем списке (даже если разные каналы и формулировки).\n"
        "3) Игнорировать любые рекламные интеграции российских компаний (Яндекс, Сбер, Авито, Тинькофф, МТС, Точка банк, "
        "Bitrix24, Ozon, Wildberries и т.п.).\n"
        "4) Игнорировать солянки и дайджесты с кучей разнородных ссылок.\n"
        "5) Приоритизировать глобальные новости об ИИ: новые модели (текста, кода, изображений, мультимодальные), чипы, "
        "крупные сделки, прорывные исследования, регуляцию, инфраструктуру, важные события крупных игроков "
        "(OpenAI, Anthropic, Google, Meta, Microsoft, Nvidia и т.п.).\n"
        "6) Локальные новости, саморазвитие, личные блоги и нишевые посты ставь ниже приоритета.\n"
        "7) Если во входных данных есть вчерашний дайджест, НЕ ВКЛЮЧАЙ в новый список новости, которые по сути описывают те же "
        "события, релизы, продукты или исследования, что и новости из вчерашнего дайджеста. Исключение: если сегодня есть "
        "значимое развитие вчерашней новости (например, вчера был только анонс, а сегодня выпуск кода, открытие весов или "
        "официальный релиз сервиса), такую новость можно включить как новую.\n\n"
        "ТРЕБОВАНИЯ К ВЫВОДУ:\n"
        "• Количество объектов в ответе не должно превышать 10.\n"
        "• Каждый объект в ответе должен иметь вид:\n"
        "[\n"
        "  {\n"
        "    \"id\": <int>,           // id из входного массива\n"
        f"    \"title\": \"краткий информативный заголовок, СТРОГО не длиннее 100 символов\",\n"
        f"    \"summary\": \"краткое описание на русском, 1–2 предложения, СТРОГО не более {max_summary_length} символов\"\n"
        "  }, ...\n"
        "]\n"
        "• Заголовок (title) можно слегка переформулировать, если исходный source_title слишком длинный.\n"
        "  Не сокращай смысл и не превращай заголовок в кликбейт.\n"
        f"• КРИТИЧЕСКИ ВАЖНО: summary должен быть не длиннее {max_summary_length} символов. Это жесткое ограничение.\n"
        "• Не добавляй никаких других полей, кроме id, title и summary.\n"
        "• Не пиши ничего до или после JSON."
    )

    yesterday_block = ""
    if yesterday_digest:
        yesterday_block = (
            "Вот дайджест новостей за предыдущий день (вчера). Используй его только для того, чтобы не включать повторы. "
            "Если какая-либо из сегодняшних новостей описывает то же событие, что и любая из вчерашних новостей без значимого "
            "развития, эту сегодняшнюю новость включать НЕ нужно.\n\n"
            "Вчерашний дайджест:\n"
            f"{yesterday_digest}\n\n"
        )

    user_prompt = (
        f"Сегодня {today}. Ниже, возможно, приведен вчерашний дайджест и JSON со всеми кандидатами в новости.\n"
        "Сформируй топ новостей по правилам выше и верни только JSON.\n\n"
        + yesterday_block
        + "JSON кандидатов:\n"
        + json.dumps(items, ensure_ascii=False)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    attempt = 0
    while True:
        attempt += 1
        last_error: LLMError | None = None
        has_retryable_error = False
        for model in LLM_MODELS:
            try:
                print(f"[llm] attempt {attempt}, model={model}")
                data = complete_json(messages, model=model)
                if not isinstance(data, list):
                    raise LLMError(f"Unexpected JSON type: {type(data)}")
                break
            except LLMError as e:
                last_error = e
                if is_retryable_llm_error(e):
                    has_retryable_error = True
                print(
                    f"[llm] error on attempt {attempt}, model={model}: {e}",
                    file=sys.stderr,
                )
        else:
            if not has_retryable_error:
                if last_error is not None:
                    raise last_error
                raise LLMError("All LLM models failed with non-retryable errors.")
            if LLM_MAX_RETRIES > 0 and attempt >= LLM_MAX_RETRIES:
                if last_error is not None:
                    raise last_error
                raise LLMError("All LLM models failed.")
            print(
                f"[llm] all models failed; retrying in {LLM_RETRY_SECONDS} seconds...",
                file=sys.stderr,
            )
            await asyncio.sleep(LLM_RETRY_SECONDS)
            continue

        break

    posts_by_id = {i + 1: p for i, p in enumerate(posts_sorted)}

    chosen: List[dict] = []
    used_ids: set[int] = set()
    MAX_ITEMS = 10

    # Берем только то, что реально вернула модель, без добивки
    for item in data:
        if len(chosen) >= MAX_ITEMS:
            break
        if not isinstance(item, dict):
            continue
        pid = item.get("id")
        if not isinstance(pid, int):
            continue
        if pid in used_ids:
            continue
        if pid not in posts_by_id:
            continue

        title = item.get("title")
        summary = item.get("summary")
        if not isinstance(title, str) or not isinstance(summary, str):
            continue

        title = limit_text(title, 100)
        summary = limit_text(summary, max_summary_length)
        if not title or not summary:
            continue

        used_ids.add(pid)
        chosen.append({"id": pid, "title": title, "summary": summary})

    # Строим markdown
    lines_md: List[str] = []
    n_items = len(chosen)
    for idx, item in enumerate(chosen, start=1):
        pid = item["id"]
        p = posts_by_id[pid]
        title = item["title"]
        summary = item["summary"]
        url = p.url

        lines_md.append(f"{idx}. [{title}]({url})")
        lines_md.append(summary)
        lines_md.append("")

    body = "\n".join(lines_md).rstrip()
    if not body:
        body = "Нет новостей, которые модель сочла достаточно важными для дайджеста."
    return body, n_items


async def send_digest(
    api_id: int,
    api_hash: str,
    text: str,
) -> None:
    if not DIGEST_PEER:
        print("[warn] DIGEST_PEER не задан, дайджест только в файл и stdout")
        return

    client = TelegramClient(SESSION_NAME, api_id, api_hash)
    async with client:
        # Ошибку отправки не глушим: иначе неудачная публикация выглядит как успех.
        await client.send_message(DIGEST_PEER, text, link_preview=False)
        print(f"[sent] digest to {DIGEST_PEER}")


# ==========================
#  MAIN
# ==========================

async def run_digest() -> None:
    channels = load_sources()
    if not channels:
        print("[error] no channels")
        return

    api_id_str = get_env("TG_API_ID", "TELEGRAM_API_ID", "API_ID")
    api_hash = get_env("TG_API_HASH", "TELEGRAM_API_HASH", "API_HASH")

    api_id = int(api_id_str)  # type: ignore[arg-type]
    since_utc = now_utc() - dt.timedelta(hours=FETCH_HOURS)
    print(f"[time] fetching since {since_utc}")

    yesterday_digest = load_yesterday_digest()

    client = TelegramClient(SESSION_NAME, api_id, api_hash)  # type: ignore[arg-type]
    async with client:
        posts = await fetch_posts(client, channels, since_utc)

    if not posts:
        date_s = now_local().strftime("%d-%m-%Y")
        final = f"Новости · {date_s}\n\nНет новостей за последние 24 часа после фильтров.\n"
        print(final)
        await send_digest(api_id, api_hash, final)
        return

    digest_body, n_items = await llm_select_top10(posts, yesterday_digest=yesterday_digest)

    date_s = now_local().strftime("%d-%m-%Y")

    if n_items > 0:
        header = f"Топ {n_items} новостей · {date_s}"
        final = f"{header}\n\n{digest_body}\n"
    else:
        header = f"Новости · {date_s}"
        final = f"{header}\n\n{digest_body}\n"

    print(final)

    # Сохраняем до отправки: если публикация упадет, текст не потеряется
    # и завтрашний дедуп получит вчерашний дайджест.
    out_name = f"digest_{now_local().strftime('%Y%m%d')}.txt"
    try:
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(final)
        print(f"[saved] {out_name}")
    except Exception as e:
        print(f"[error] failed to save digest file: {e}", file=sys.stderr)

    await send_digest(api_id, api_hash, final)


if __name__ == "__main__":
    asyncio.run(run_digest())
