import json
import re
from typing import Dict, List, Tuple


INVALID_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def _extract_candidate_json_strings(content: str) -> List[str]:
    candidates: List[str] = []

    # Prefer fenced JSON blocks when present.
    fence_pattern = r'```json\s*([\s\S]*?)\s*```'
    fenced_matches = re.findall(fence_pattern, content, re.DOTALL | re.IGNORECASE)
    for match in fenced_matches:
        if match.strip():
            candidates.append(match.strip())

    # Fallback: try parsing the whole response.
    if content.strip():
        candidates.append(content.strip())

    # Last-resort fallback: extract the outermost object payload.
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        obj_slice = content[first_brace:last_brace + 1].strip()
        if obj_slice:
            candidates.append(obj_slice)

    # Deduplicate while preserving order.
    unique_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _remove_invalid_control_chars(text: str) -> str:
    # Keep legal JSON whitespace (\t, \n, \r), remove other control chars.
    return INVALID_CONTROL_CHAR_RE.sub('', text)


def parse_json_str(content: str) -> Tuple[Dict, str]:
    content = content.strip()

    # GLM-4.5V
    if 'begin_of_box' in content: 
        pattern = r'\<\|begin_of_box\|\>([\s\S]*?)\<\|end_of_box\|\>'
        content_match = re.search(pattern, content, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
    
        else:
            content = content.strip().split('begin_of_box')[1].strip()
            if 'end_of_box' in content:
                content = content.split('end_of_box')[0].strip()

    last_error: Exception | None = None
    for json_candidate in _extract_candidate_json_strings(content):
        parse_attempts = [json_candidate]
        cleaned_candidate = _remove_invalid_control_chars(json_candidate)
        if cleaned_candidate != json_candidate:
            parse_attempts.append(cleaned_candidate)

        for parse_attempt in parse_attempts:
            try:
                return json.loads(parse_attempt), parse_attempt
            except Exception as e:
                last_error = e

    if last_error is not None:
        raise last_error
    raise ValueError('Failed to extract JSON content from model response.')
