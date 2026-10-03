# -*- coding: utf-8 -*-
"""
LLM-адаптер для Gemini через google-genai.

Основные принципы:
- Ленивая инициализация клиента (ключи из .env)
- Строгий JSON-режим (response_mime_type="application/json")
- Аккуратный разбор: убираем ``` и мусор вокруг JSON
- Salvage: если массив обрублен, собираем валидный префикс
- Текстовый режим complete_text для отладки и "жестких" ретраев

По умолчанию используется модель gemini-flash-latest.
Можно переопределить через переменную окружения LLM_MODEL.

max_output_tokens в адаптере намеренно не используется.
Аргументы max_output_tokens оставлены только для совместимости сигнатуры.
"""

import os
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Не установлен google-genai. Установи: pip install google-genai"
    ) from e

load_dotenv(Path(__file__).with_name(".env"))


class LLMError(Exception):
    pass


_CLIENT: Optional["genai.Client"] = None
_MODEL: Optional[str] = None


def _get_client() -> ("genai.Client", str):
    """
    Ленивая инициализация клиента Gemini.
    """
    global _CLIENT, _MODEL
    if _CLIENT is None:
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise LLMError(
                "Не найден ключ API. "
                "Задай хотя бы одну переменную: LLM_API_KEY, GEMINI_API_KEY или GOOGLE_API_KEY."
            )
        _MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT, _MODEL  # type: ignore[return-value]


def _to_genai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Простое отображение {role, content} -> [{role, parts:[{text}]}].

    Для нашего сценария достаточно схлопывать роли в user.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        text = (m.get("content") or "").strip()
        if not text:
            continue
        out.append(
            {
                "role": "user",
                "parts": [{"text": text}],
            }
        )
    if not out:
        out = [{"role": "user", "parts": [{"text": ""}]}]
    return out


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE | re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s.strip())


def _remove_invisibles(s: str) -> str:
    # Убираем zero-width и контрольные символы, чтобы не ломать json.loads
    return "".join(ch for ch in s if ch.isprintable() or ch in "\r\n\t")


def _extract_json_fragment(s: str) -> str:
    """
    Вырезаем JSON-массив или объект из текста.
    В приоритете массив.
    """
    s = _strip_code_fences(_remove_invisibles(s))
    if not s:
        raise LLMError("Пустой ответ от модели.")
    # массив
    start = s.find("[")
    if start != -1:
        end = s.rfind("]")
        if end != -1 and end > start:
            return s[start : end + 1]
        return s[start:]
    # объект
    start = s.find("{")
    if start != -1:
        end = s.rfind("}")
        if end != -1 and end > start:
            return s[start : end + 1]
        return s[start:]
    raise LLMError("В ответе не найден JSON.")


def _salvage_json_array_prefix(s: str) -> Optional[list]:
    """
    Пытаемся собрать валидный префикс JSON-массива из обрубленного текста.
    Возвращаем list или None.
    """
    s = s.strip()
    if not s.startswith("["):
        return None

    items = []
    i = 1
    n = len(s)

    def skip_ws(pos: int) -> int:
        while pos < n and s[pos] in " \t\r\n":
            pos += 1
        return pos

    i = skip_ws(i)
    if i < n and s[i] == "]":
        return []

    while i < n:
        i = skip_ws(i)
        if i >= n or s[i] == "]":
            break

        start = i
        depth_obj = 0
        depth_arr = 0
        in_str = False
        esc = False
        finished = False

        while i < n:
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue

            if ch == '"':
                in_str = True
                i += 1
                continue

            if ch == "{":
                depth_obj += 1
            elif ch == "}":
                if depth_obj > 0:
                    depth_obj -= 1
            elif ch == "[":
                depth_arr += 1
            elif ch == "]":
                if depth_arr > 0:
                    depth_arr -= 1
                else:
                    finished = True
                    i += 1
                    break
            elif ch == "," and depth_obj == 0 and depth_arr == 0:
                finished = True
                i += 1
                break

            i += 1

        if not finished and (depth_obj or depth_arr or in_str):
            break

        elem_text = s[start:i].rstrip().rstrip(",").strip()
        if not elem_text:
            continue

        try:
            item = json.loads(elem_text)
            items.append(item)
        except Exception:
            break

        i = skip_ws(i)
        if i < n and s[i] == "]":
            break

    return items or None


def _safe_get_text(res: Any) -> Optional[str]:
    """
    Аккуратно достаем res.text, не падая, если quick accessor кидает исключение.
    """
    try:
        text_attr = getattr(res, "text", None)
    except Exception:
        return None
    if not text_attr:
        return None
    try:
        return text_attr.strip()
    except Exception:
        return None


def complete_json(
    messages: List[Dict[str, Any]],
    max_output_tokens: int = 1200,
    response_schema: Optional[Any] = None,
    model: Optional[str] = None,
) -> Any:
    """
    Просим у модели JSON.

    max_output_tokens намеренно не передается в GenerateContentConfig
    и используется только для совместимости сигнатуры.
    Если ответ обрублен, пытаемся собрать валидный префикс массива.
    """
    client, default_model = _get_client()
    selected_model = model or default_model
    try:
        cfg_kwargs: Dict[str, Any] = dict(
            response_mime_type="application/json",
            temperature=0.1,
            top_p=0.0,
        )
        if response_schema is not None:
            cfg_kwargs["response_schema"] = response_schema

        cfg = GenerateContentConfig(**cfg_kwargs)

        res = client.models.generate_content(
            model=selected_model,
            contents=_to_genai(messages),
            config=cfg,
        )

        if hasattr(res, "parsed") and res.parsed is not None:
            return res.parsed

        raw = _safe_get_text(res)
        if not raw:
            if hasattr(res, "to_dict"):
                raw = json.dumps(res.to_dict(), ensure_ascii=False)
            else:
                raw = str(res)

        frag = _extract_json_fragment(raw)
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            salvage = _salvage_json_array_prefix(frag)
            if salvage is not None:
                return salvage
            raise

    except Exception as e:
        raise LLMError(f"LLM JSON error: {e}") from e


def complete_text(
    messages: List[Dict[str, Any]],
    max_output_tokens: int = 1600,
    model: Optional[str] = None,
) -> str:
    """
    Текстовый режим для ретраев и отладки.

    max_output_tokens намеренно игнорируется и не передается в конфиг.
    """
    client, default_model = _get_client()
    selected_model = model or default_model
    try:
        cfg = GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.1,
            top_p=0.0,
        )
        res = client.models.generate_content(
            model=selected_model,
            contents=_to_genai(messages),
            config=cfg,
        )
        text = _safe_get_text(res)
        if text is None:
            if hasattr(res, "to_dict"):
                return json.dumps(res.to_dict(), ensure_ascii=False)
            return str(res)
        return text
    except Exception as e:
        raise LLMError(f"LLM text fallback error: {e}") from e
