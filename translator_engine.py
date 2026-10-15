import sys
import json
import re
import time
import random
import threading

if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import srt

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


def normalize_gemini_model(model_name: Optional[str]) -> str:
    """Returns normalized model name prioritizing ultra-fast models."""
    if not model_name:
        return "gemini-3.5-flash-lite"
    m = str(model_name).strip()
    if any(bad in m for bad in ["2.5-flash", "2.0-flash", "1.5-flash"]):
        return "gemini-3.5-flash-lite"
    return m


def test_gemini_api_key(api_key: str, model_name: str = "gemini-flash-lite-latest") -> Tuple[bool, str]:
    """
    Tests a Gemini API key with all supported Google Studio models.
    Returns (is_valid, message).
    """
    if not genai:
        return False, "បណ្ណាល័យ 'google-genai' មិនទាន់ត្រូវបានដំឡើងទេ។"

    api_key = api_key.strip().strip('"\'')
    if not api_key:
        return False, "សូមបញ្ចូល API Key ជាមុនសិន។"

    norm_model = normalize_gemini_model(model_name)
    models_to_try = [
        norm_model,
        "gemini-3.8-flash",
        "gemini-flash-lite-latest",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ]
    # De-duplicate preserving order
    seen = set()
    models = [m for m in models_to_try if not (m in seen or seen.add(m))]

    client = genai.Client(api_key=api_key)
    last_err = ""

    for m in models:
        try:
            response = client.models.generate_content(
                model=m,
                contents="Say 'OK' in one word.",
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                    temperature=0.1
                )
            )
            resp_text = response.text.strip() if response and response.text else "OK"
            return True, f"✅ API Key ត្រឹមត្រូវ និងដំណើរការបានល្អជាមួយ {m}! (ឆ្លើយតប: {resp_text})"
        except Exception as e:
            err_str = str(e)
            last_err = err_str

            # If quota exceeded or temporary 503, the key IS VALID and authenticated
            if "429" in err_str or "resource_exhausted" in err_str.lower():
                return True, (
                    f"✅ API Key ត្រឹមត្រូវ! (ទទួលស្គាល់ដោយ Google Studio)\n\n"
                    f"⚠️ ចំណាំ៖ ម៉ូដែល {m} កំពុងជាប់ Rate Limit (Quota 429) បន្តិចបន្តួច។\n"
                    f"កម្មវិធីនឹងធ្វើការ Retry និងផ្ទេរម៉ូដែលដោយស្វ័យប្រវត្តិពេលបកប្រែ។"
                )
            elif "503" in err_str or "unavailable" in err_str.lower():
                continue  # Try next model in list
            elif "401" in err_str or "unauthenticated" in err_str.lower():
                return False, (
                    "❌ កំហុស 401 UNAUTHENTICATED:\n"
                    "API Key មិនត្រឹមត្រូវ ឬផុតកំណត់។ សូមពិនិត្យមើល Key ក្នុង Google AI Studio ឡើងវិញ។"
                )
            elif "403" in err_str or "permission_denied" in err_str.lower() or "denied access" in err_str.lower():
                return False, (
                    "❌ កំហុស 403 PERMISSION_DENIED:\n"
                    "គម្រោង/គណនី Google នេះត្រូវបានបិទ (Project Denied Access)។ សូមលុប Key នេះចេញពីបញ្ជី Settings។"
                )
            elif "404" in err_str or "not_found" in err_str.lower() or "invalid_argument" in err_str.lower():
                continue

    return False, f"❌ កំហុសក្នុងការភ្ជាប់ទៅ Google Studio: {last_err[:200]}"


def detect_and_read_file(filepath: str) -> Tuple[str, str]:
    encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'big5', 'utf-16', 'cp936']
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            return text, enc
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode('utf-8', errors='replace'), 'utf-8 (fallback)'


def parse_srt_content(content: str) -> List[srt.Subtitle]:
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    try:
        subtitles = list(srt.parse(content))
        return subtitles
    except Exception:
        clean_subtitles = []
        pattern = re.compile(
            r'(\d+)\n'
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\n'
            r'([\s\S]*?)(?=\n\d+\n\d{2}:\d{2}:\d{2}|\Z)',
            re.MULTILINE
        )
        for match in pattern.finditer(content):
            idx = int(match.group(1))
            start = srt.srt_timestamp_to_timedelta(match.group(2).replace('.', ','))
            end = srt.srt_timestamp_to_timedelta(match.group(3).replace('.', ','))
            text = match.group(4).strip()
            clean_subtitles.append(srt.Subtitle(index=idx, start=start, end=end, content=text))
        return clean_subtitles


def format_timestamp(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def export_srt_file(
    subtitles: List[Any],
    output_filepath: str,
    translations: Optional[Dict[int, Dict[str, Any]]] = None,
    include_gender_tag: bool = True,
) -> bool:
    """
    Exports subtitle list to a valid UTF-8 SRT file with optional translated Khmer text
    and speaker gender tags [មនុស្សប្រុស] / [មនុស្សស្រី].
    """
    try:
        lines = []
        translations = translations or {}
        for sub in subtitles:
            sid = sub.index
            start_tc = format_timestamp(sub.start)
            end_tc = format_timestamp(sub.end)

            t_data = translations.get(sid, {})
            khmer_txt = t_data.get("khmer", "").strip()
            gender = t_data.get("gender", "ប្រុស").strip()

            if khmer_txt:
                clean_txt = re.sub(r"^\[[^\]]*\]\s*", "", khmer_txt).strip()
                if include_gender_tag:
                    tag = f"[មនុស្ស{gender}]" if gender in ["ប្រុស", "ស្រី"] else f"[{gender}]"
                    content_str = f"{tag} {clean_txt}"
                else:
                    content_str = clean_txt
            else:
                content_str = sub.content

            lines.append(f"{sid}\n{start_tc} --> {end_tc}\n{content_str}\n")

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return True
    except Exception as e:
        print(f"Error exporting SRT file: {e}")
        return False

FOREIGN_WORDS_MAP = {
    # English leaked terms
    "arrogant": "ក្អេងក្អាង",
    "Arrogant": "ក្អេងក្អាង",
    "ARROGANT": "ក្អេងក្អាង",
    "proud": "អួតអាង",
    "confident": "ជឿជាក់",
    "hello": "សួស្តី",
    "hi": "សួស្តី",
    "bye": "លាហើយ",
    "goodbye": "លាហើយ",
    "ok": "យល់ព្រម",
    "okay": "យល់ព្រម",
    "yes": "បាទ",
    "no": "ទេ",
    "sorry": "សុំទោស",
    "thanks": "អរគុណ",
    "thank you": "អរគុណ",
    "wechat": "វីឆាត",
    "tiktok": "ទិកតុក",
    "facebook": "ហ្វេសប៊ុក",
    "telegram": "តេឡេក្រាម",
    "boss": "មេ",
    "bro": "បងប្អូន",
    # Japanese katakana / corruptions occasionally output by LLMs
    "រエ": "ឬ",
    "អエ": "អឺ",
    "エ": "ឺ",
    "អλαើន": "ក្អេងក្អាង",
}

FILM_CORRECTIONS_MAP = {
    # Chinese idiom mistranslations
    "កុំក្រអឺតក្រអឺតអី": "កុំគិតច្រើនអី ញ៉ាំឱ្យអស់ដៃទៅ",
    "កុំក្រអឺតក្រទមអី": "កុំគិតច្រើនអី ញ៉ាំឱ្យអស់ដៃទៅ",
    "បាយអូសំប៉ែត": "អែបអប",
    "សេចក្តីស្ងោរចិត្ត": "ស្ទាក់ស្ទើរក្នុងចិត្ត",
    "ដាក់ខលបែនចោល": "រាយការណ៍ថាជ្រុះបាត់",
    "វាយកាតយានដ្ឋាន": "ព្រីនសំបុត្រចតឡាន",
    "វិក្កយបត្រសប្តាហ៍នេះ": "វិក្កយបត្រតុនេះ",
    "គ្មានភ្នែកមើលសោះហ៎": "មិនចេះមើលកាលៈទេសៈសោះហើយ",
    "មីនាង ស៊ូ": "មុយយាន",
    "ម្តាយមីងម្នាក់នេះហ៎": "អីយ៉ា បងស្រីអើយ",
    # Corrupt Arabic glyph transliterations (免单 / 免费 -> ហ្វ្រី / ឥតគិតថ្លៃ)
    "ហريស": "ឥតគិតថ្លៃ",
    "ហري": "ឥតគិតថ្លៃ",
    "ري": "",
    "ហ្វ្រីមិនគិតលុយ": "ឥតគិតថ្លៃ",
    "ឥតគិតថ្លៃមិនគិតលុយ": "ឥតគិតថ្លៃ",
    "មិនអាចហ្វ្រីបានទេ": "មិនអាចឥតគិតថ្លៃបានទេ",
    "មិនអាចហ្វ្រី": "មិនអាចឥតគិតថ្លៃ",
    # Phonetic glitches for Chinese particles (过来 / 咯 / 够了 -> ហ្ហ៎ / ហ្ហី / ល្មមហើយ)
    # Phonetic glitches for Chinese particles (过来 / 咯 / 够了 -> ហ្ហ៎ / ហ្ហី / ល្មមហើយ)
    "មិនយល់ទៀតហ្គោល?": "មិនទាន់យល់ទៀតហ្ហ៎?",
    "មិនយល់ទៀតហ្គោល": "មិនទាន់យល់ទៀតហ្ហ៎",
    "យល់ទៀតហ្គោល?": "យល់ទៀតហ្ហ៎?",
    "យល់ទៀតហ្គោល": "យល់ទៀតហ្ហ៎",
    "ទៀតហ្គោល?": "ទៀតហ្ហ៎?",
    "ទៀតហ្គោល": "ទៀតហ្ហ៎",
    "ហ្ហី? ទេ": "ហ្ហី?",
    "ហ្ហ៎? ទេ": "ហ្ហ៎?",
    # Common machine-translation corruptions
    "លគេអ្នក": "អួតគេ",
    "ថក្រោយ": "ថយក្រោយ",
    "អអួត": "អួតអាង",
    "អអុញ": "ក្អេងក្អាង",
    "ទះតុមួយខ្សឹប": "ទះតុមួយទំហឹង",
    "គ្មានកែងតម្រង់អៀនខ្មាស": "គ្មានកេរ្តិ៍ខ្មាសសោះ",
    "កែវភ្នែកជាដួងចិត្ត": "កែវភ្នែកបេះដូង",
    "ស្ងប់ស្ងាត់ដូចជាមិនមានជីវិត": "ស្ងាត់ជ្រងំដូចគ្មានមនុស្ស",
    "សាច់គោ ុ": "សាច់គោ",
    # Military title (督军 -> លោកមេទ័ព)
    "លោកទុជុន": "លោកមេទ័ព",
    "ទុជុន": "មេទ័ព",
    # Star curse / evil spirit (煞星 -> គ្រោះចង្រៃ / មនុស្សចង្រៃ)
    "ជា សាត ទុកមិនបានឡើយ": "ជាគ្រោះចង្រៃ ទុកមិនបានឡើយ",
    "ជាសាតទុកមិនបានឡើយ": "ជាគ្រោះចង្រៃទុកមិនបានឡើយ",
    "ជា សាត": "ជាគ្រោះចង្រៃ",
    "ជាសាត": "ជាគ្រោះចង្រៃ",
    "នេះជា ង្រៃ": "នេះជាគ្រោះចង្រៃ",
    "នេះជាង្រៃ": "នេះជាគ្រោះចង្រៃ",
    "ជា ង្រៃ": "ជាគ្រោះចង្រៃ",
    "ជាង្រៃ": "ជាគ្រោះចង្រៃ",
    "ក្មេងង្រៃ": "ក្មេងចង្រៃ",
    "គ្រោះង្រៃ": "គ្រោះចង្រៃ",
    "ក្មេងសាត": "ក្មេងចង្រៃ",
    "គ្រោះសាត": "គ្រោះចង្រៃ",
    "ផ្កាយសាត": "ផ្កាយចង្រៃ",
    # Distorted animal name (鹦鹉 -> សត្វសេក)
    "សត្វសង្កែឥន្ទ្រី": "សត្វសេក",
    "សង្កែឥន្ទ្រី": "សត្វសេក",
    # Temper distortion (脾气不好 -> ចរិតមិនល្អ / កំហឹងកាច)
    "អរិយធម៌មិនល្អ": "ចរិតមិនល្អ",
    "អរិយធម៌កាច": "ចរិតកាច",
    # Clan naming consistency (北冥 -> ប៉ីមីង)
    "ត្រកូលបេយមីង": "ត្រកូលប៉ីមីង",
    "ត្រកូលបើយមីង": "ត្រកូលប៉ីមីង",
    "បេយមីង": "ប៉ីមីង",
    "បើយមីង": "ប៉ីមីង",
}

PROMO_AD_PATTERNS = [
    re.compile(r'点击(?:下方|这里|链接|关注)?', re.IGNORECASE),
    re.compile(r'关注我.*(?:完整版|下集)?', re.IGNORECASE),
    re.compile(r'点赞.*收藏', re.IGNORECASE),
    re.compile(r'投币.*转发', re.IGNORECASE),
    re.compile(r'精彩内容.*下集', re.IGNORECASE),
    re.compile(r'(?:សូម)?(?:អញ្ជើញ)?ចុច(?:នៅ)?(?:លើ)?(?:តំណ)?(?:ភ្ជាប់)?ខាងក្រោម', re.IGNORECASE),
    re.compile(r'ចុច.*(?:ខាងក្រោម|តំណ)', re.IGNORECASE),
    re.compile(r'ដើម្បី(?:រីករាយ)?ទស្សនា', re.IGNORECASE),
    re.compile(r'តាមដានខ្ញុំដើម្បីទស្សនា', re.IGNORECASE),
]


def is_promo_ad_line(chinese_text: str = "", khmer_text: str = "") -> bool:
    """Detects whether a subtitle line is social media promo/ad rather than movie story dialogue."""
    combined = f"{chinese_text} {khmer_text}"
    for pat in PROMO_AD_PATTERNS:
        if pat.search(combined):
            return True
    return False


# Regex to detect any non-Khmer text characters:
# Allowed characters:
# - Khmer Unicode block: \u1780-\u17FF
# - Khmer Symbols: \u19E0-\u19FF
# - Standard spaces and punctuation: \s .,!?:;'"“”«»()\-–—…
# - Khmer punctuation: \u17D4 (។), \u17D5 (៕), \u17D6 (៖), \u17D7 (ៗ), \u17D8 (៘), \u17D9 (៙), \u17DA (៚), \u17DB (៛)
NON_KHMER_CHAR_REGEX = re.compile(
    r'[^\u1780-\u17FF\u19E0-\u19FF\s.,!?:;\'"“”«»()\-–—…\u17D4\u17D5\u17D6\u17D7\u17D8\u17D9\u17DA\u17DB]'
)


KHMER_TO_ARABIC_DIGITS = str.maketrans({
    '០': '0', '១': '1', '២': '2', '៣': '3', '៤': '4',
    '៥': '5', '៦': '6', '៧': '7', '៨': '8', '៩': '9'
})

ONES_WORDS = {
    0: 'សូន្យ', 1: 'មួយ', 2: 'ពីរ', 3: 'បី', 4: 'បួន',
    5: 'ប្រាំ', 6: 'ប្រាំមួយ', 7: 'ប្រាំពីរ', 8: 'ប្រាំបី', 9: 'ប្រាំបួន'
}

TENS_WORDS = {
    1: 'ដប់', 2: 'ម្ភៃ', 3: 'សាមសិប', 4: 'សែសិប', 5: 'ហាសិប',
    6: 'ហុកសិប', 7: 'ចិតសិប', 8: 'ប៉ែតសិប', 9: 'កៅសិប'
}


def number_to_khmer_words(n: int) -> str:
    """Converts an integer to natural written Khmer words (មួយ ពីរ បី...)."""
    if n < 0:
        return 'ដក' + number_to_khmer_words(-n)
    if n < 10:
        return ONES_WORDS[n]
    if n < 100:
        ten, unit = divmod(n, 10)
        prefix = TENS_WORDS[ten]
        if unit == 0:
            return prefix
        if ten == 1:
            return 'ដប់' + ONES_WORDS[unit]
        return prefix + ONES_WORDS[unit]
    if n < 1000:
        hundred, rest = divmod(n, 100)
        prefix = 'មួយរយ' if hundred == 1 else ONES_WORDS[hundred] + 'រយ'
        return prefix if rest == 0 else prefix + number_to_khmer_words(rest)
    if n < 10000:
        thousand, rest = divmod(n, 1000)
        prefix = 'មួយពាន់' if thousand == 1 else ONES_WORDS[thousand] + 'ពាន់'
        return prefix if rest == 0 else prefix + number_to_khmer_words(rest)
    if n < 100000:
        tenthousand, rest = divmod(n, 10000)
        prefix = 'មួយម៉ឺន' if tenthousand == 1 else ONES_WORDS[tenthousand] + 'ម៉ឺន'
        return prefix if rest == 0 else prefix + number_to_khmer_words(rest)
    if n < 1000000:
        hundredthousand, rest = divmod(n, 100000)
        prefix = 'មួយសែន' if hundredthousand == 1 else ONES_WORDS[hundredthousand] + 'សែន'
        return prefix if rest == 0 else prefix + number_to_khmer_words(rest)
    if n < 10000000:
        million, rest = divmod(n, 1000000)
        prefix = 'មួយលាន' if million == 1 else ONES_WORDS[million] + 'លាន'
        return prefix if rest == 0 else prefix + number_to_khmer_words(rest)
    return ''.join(ONES_WORDS.get(int(d), d) for d in str(n))


def replace_numbers_with_khmer_words(text: str) -> str:
    """Replaces all numbers (Arabic or Khmer digits) with spoken Khmer words (មួយ ពីរ បី)."""
    # Replace percent signs
    text = text.replace('%', 'ភាគរយ').replace('％', 'ភាគរយ')
    # Normalize Khmer digits to Arabic digits for parsing
    text = text.translate(KHMER_TO_ARABIC_DIGITS)

    # Handle decimals: e.g. 50.5 -> ហាសិបចុចប្រាំ
    def decimal_repl(match):
        whole = int(match.group(1))
        dec = match.group(2)
        dec_words = ''.join(ONES_WORDS.get(int(d), d) for d in dec)
        return f"{number_to_khmer_words(whole)}ចុច{dec_words}"

    text = re.sub(r'(\d+)\.(\d+)', decimal_repl, text)

    # Handle whole numbers
    def whole_repl(match):
        val = int(match.group(0))
        return number_to_khmer_words(val)

    text = re.sub(r'\d+', whole_repl, text)
    return text


def sanitize_khmer_text(text: str) -> str:
    """
    Guarantees 100% pure Khmer text:
    - Removes English inside parentheses like (WeChat), (AI)
    - Converts numbers to written Khmer words (មួយ ពីរ បី...)
    - Converts '%' and '％' to 'ភាគរយ'
    - Replaces known leaked English/corrupt words (arrogant -> ក្អេងក្អាង, រエ -> ឬ)
    - Replaces Thai vocabulary
    - Strictly strips any remaining non-Khmer characters (Latin, Japanese, Greek, Thai, etc.)
    """
    if not text:
        return ""

    # 1. Remove parenthesized English words: e.g. (WeChat), （WeChat）, (AI), (App)
    text = re.sub(r'[\(（]\s*[A-Za-z0-9\s\.\-_/&]+\s*[\)）]', '', text)

    # 2. Map known foreign/corrupt words to proper Khmer equivalents
    for foreign_w, khmer_w in FOREIGN_WORDS_MAP.items():
        text = re.sub(rf'\b{re.escape(foreign_w)}\b', khmer_w, text, flags=re.IGNORECASE)
        if foreign_w in text:
            text = text.replace(foreign_w, khmer_w)

    # 2.1 Map drama subtitle idioms & phonetic glitch corrections
    for bad_w, good_w in FILM_CORRECTIONS_MAP.items():
        if bad_w in text:
            text = text.replace(bad_w, good_w)

    # Clean hallucinated 'ហ្គោល' at the end of questions/sentences (phonetic glitch of 过来/咯)
    text = re.sub(r'[\s,]*(?:ហ្គោល|ហ្គោល\?|ហ្គោល\s*\?)\s*$', 'ហ្ហ៎?', text)
    text = re.sub(r'[\s,]*(?:ហ្គោល)\s*([?!។])', r'ហ្ហ៎', text)

    # 3. Convert all numbers (Arabic or Khmer digits) to spoken Khmer words (មួយ ពីរ បី)
    text = replace_numbers_with_khmer_words(text)

    # 4. Clean stray affectionate terms like បងសម្លាញ់ / អូនសម្លាញ់ / ណា៎បង / ណា៎អូន and coarse pronouns
    text = re.sub(r'[\s,]*(?:ណា៎បង|ណា៎អូន)', 'ណា៎', text)
    text = re.sub(r'[\s,]*(?:បងសម្លាញ់|អូនសម្លាញ់)', '', text)
    text = re.sub(r'\bចាំបង\b', 'ចាំខ្ញុំ', text)
    text = re.sub(r'\bបងបញ្ចូល\b', 'ខ្ញុំបញ្ចូល', text)
    text = re.sub(r'\bបងទាញ\b', 'ខ្ញុំទាញ', text)
    text = re.sub(r'អញ(?![\u17d2])', 'ខ្ញុំ', text)
    text = re.sub(r'អ្ហែង', 'អ្នក', text)

    # 5.1 Clean prohibited unnatural mother terms (លោកម៉ែ, លោកម៉ាក់, លោកម្ដាយ, លោកម្តាយ)
    text = text.replace('លោកម៉ែ', 'ម៉ែ')
    text = text.replace('លោកម៉ាក់', 'ម៉ាក់')
    text = text.replace('លោកម្ដាយ', 'ម្តាយ')
    text = text.replace('លោកម្តាយ', 'ម្តាយ')

    # 6. Strip all parenthesized commentary or explanations (សម្គាល់: ...), (note: ...), etc.
    text = re.sub(r'[\(（]\s*(?:សម្គាល់|note|meaning|ការពន្យល់|សំគាល់|note:).*?[\)）]', '', text, flags=re.IGNORECASE)
    # Remove all parentheses characters completely to guarantee clean subtitle lines
    text = text.replace('(', '').replace(')', '').replace('（', '').replace('）', '')

    # 7. Strip any remaining leaked Arabic/foreign non-Khmer characters
    text = NON_KHMER_CHAR_REGEX.sub('', text)

    # 8. Normalize multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


class GeminiSubtitleTranslator:
    def __init__(
        self,
        api_keys: List[str],
        model_name: str = "gemini-3.5-flash-lite",
        style: str = "emotion_acting",
        global_character_roster: Optional[Dict[str, Any]] = None,
    ):
        self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        if not self.api_keys:
            raise ValueError("API Keys មិនអាចទទេបានទេ។")
        self.model_name = model_name
        self.style = style
        self.client_idx = 0
        self.client_lock = threading.Lock()
        self.key_cooldowns: Dict[str, float] = {}
        self.clients = []
        for key in self.api_keys:
            try:
                c = genai.Client(api_key=key)
                self.clients.append((c, key))
            except Exception:
                pass
        if not self.clients:
            raise ValueError("មិនអាចបង្កើត Gemini Client បានទេ។ សូមពិនិត្យមើល API Keys។")
        self.last_speaker_map: Dict[int, Dict[str, Any]] = {}
        self.last_characters: Dict[str, Dict[str, Any]] = {}
        self.global_character_roster: Dict[str, Dict[str, Any]] = global_character_roster or {}
        self.character_name_to_gender: Dict[str, str] = {}
        self.character_pronoun_rules: List[str] = []
        if self.global_character_roster:
            for c_name, c_info in self.global_character_roster.items():
                if isinstance(c_info, dict):
                    raw_g = str(c_info.get("gender", "")).lower()
                    c_gen = "ស្រី" if ("ស្រី" in raw_g or "female" in raw_g or "woman" in raw_g or "女" in str(c_info.get("gender", ""))) else "ប្រុស"
                    self.character_name_to_gender[c_name] = c_gen
                    for al in c_info.get("aliases", []):
                        if al:
                            self.character_name_to_gender[str(al).strip()] = c_gen

    def _get_next_client(self):
        with self.client_lock:
            if not self.clients:
                raise RuntimeError("គ្មាន API Key ដែលដំណើរការទៀតទេ។ គ្រប់ Keys ទាំងអស់សុទ្ធតែត្រូវបានបិទ (403) ឬមិនត្រឹមត្រូវ។")
            now = time.time()
            # Try to find a client not in cooldown
            for _ in range(len(self.clients)):
                client, key = self.clients[self.client_idx % len(self.clients)]
                self.client_idx += 1
                if now >= self.key_cooldowns.get(key, 0):
                    return client, key
            # If all clients are in cooldown, pick the one with earliest cooldown expiry
            best_client, best_key = min(self.clients, key=lambda c: self.key_cooldowns.get(c[1], 0))
            return best_client, best_key

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 30.0):
        with self.client_lock:
            self.key_cooldowns[key] = time.time() + cooldown_seconds
            print(f"[Rate Limit 429] ⚠️ Key {key[:8]}... ជាប់ Quota ផ្អាកបណ្តោះអាសន្ន {int(cooldown_seconds)}s (ប្តូរទៅ Key បន្ទាប់)")

    def mark_bad_key(self, bad_key: str, reason: str = ""):
        with self.client_lock:
            old_count = len(self.clients)
            self.clients = [c for c in self.clients if c[1] != bad_key]
            self.key_cooldowns.pop(bad_key, None)
            if len(self.clients) < old_count:
                print(f"[Key Evicted] ❌ ជម្រុះ Key ខូច ({reason}): {bad_key[:8]}... នៅសល់ {len(self.clients)} Keys ដំណើរការ")

    def _get_system_prompt(self) -> str:
        style_instructions = {
            "emotion_acting": (
                "【រចនាប័ទ្មសម្ដែងបញ្ចូលសំឡេងរឿងភាគ និងសម្រាយរឿងតាមអារម្មណ៍តួអង្គ (Emotion Acting & Natural Spoken Dubbing/Storytelling)】៖\n"
                "១. សព្វនាមទំនើប ស៊ីវិល័យ និងសុភាពរាបសារ (Modern, Polite & Cinematic Spoken Pronouns):\n"
                "   - ហាមដាច់ខាតមិនឱ្យប្រើពាក្យ 'អញ' និង 'អ្ហែង' ជាដាច់ខាត (STRICTLY PROHIBITED)!\n"
                "   - ជៀសវាងការប្រើពាក្យ 'ឯង' និង 'បង - អូន' ហួសហេតុ ឬដដែលៗ។\n"
                "   - ប្រើសព្វនាមជំនួសដែលសមរម្យ និងថ្លៃថ្នូរ:\n"
                "     * ជំនួស 'អញ' ដោយ: 'ខ្ញុំ' (គួរសមទូទៅ) ឬ 'យើង' (បញ្ជា/ម៉ឺងម៉ាត់)។\n"
                "     * ជំនួស 'ឯង' / 'អ្ហែង' ដោយ: 'អ្នក' (សុភាព), 'លោក' (ចំពោះបុរស), 'អ្នកនាង / កញ្ញា' (ចំពោះនារី), 'គ្នា / គ្នាយើង' (មិត្តជិតស្និទ្ធ) ឬ ហៅឈ្មោះតួអង្គផ្ទាល់។\n"
                "     * ជំនួស 'បង - អូន' ដោយ: 'ខ្ញុំ - អ្នក', 'លោក', 'អ្នកនាង', ឬ ហៅឈ្មោះតួអង្គផ្ទាល់ (ឧ. 'ស៊ឺវុី', 'វ៉ាន់នីង', 'លោក ឈូ')។ ចំពោះប្ដីប្រពន្ធ អាចប្រើ 'ប្ដីសម្លាញ់ / ប្រពន្ធសម្លាញ់' ឬ ឈ្មោះផ្ទាល់ ដោយមិនប្រើបងអូនញឹកញាប់ពេកឡើយ។\n"
                "   - ឪពុកម្តាយ និងកូន: 'ម៉ាក់ - កូន', 'ប៉ា/ពុក - កូន', 'កូនស្រី', 'កូនប្រុស', 'ម៉ែ - កូន'។\n"
                "   - ឋានន្តរស័ក្តិ / បុរាណ / ក្បាច់គុន: 'លោកម្ចាស់', 'អ្នកប្រុស', 'អ្នកនាង', 'លោកមេបញ្ជាការ', 'ចៅហ្វាយ', 'ទូលបង្គំ', 'ទ្រង់/ព្រះអង្គ'។\n"
                "២. ទឹកដមសំឡេង និងអារម្មណ៍តួអង្គ (Character Emotion Context):\n"
                "   - ឈុតសម្រាយរឿង / រៀបរាប់ដំណើររឿង: ប្រើភាសារៀបរាប់រលូន ជក់ចិត្ត ក្បោះក្បាយ ដូចអ្នកសម្រាយរឿងអាជីព។\n"
                "   - ឈុតផ្អែមល្ហែម / ស្នេហា / លួងលោម: ប្រើពាក្យស្រទន់ ពីរោះ រណ្តំ លួងលោម បង្កប់ក្តីស្រឡាញ់ និងភាពកក់ក្តៅ ដោយហៅឈ្មោះគ្នា ឬប្រើ 'ខ្ញុំ - អ្នក', 'ប្ដីសម្លាញ់ / ប្រពន្ធសម្លាញ់' (ឧ. 'កុំបារម្ភអីណា៎... ខ្ញុំនៅក្បែរអ្នកជានិច្ច', 'គេងឱ្យស្រួលណា៎')។\n"
                "   - ឈុតអង្វរករ / ទទូចសុំ: ប្រើពាក្យទន់ភ្លន់ ទទូច អង្វរ សុំក្តីមេត្តា (ឧ. 'ខ្ញុំសុំអង្វរ... មេត្តាលើកលែងទោសឱ្យខ្ញុំម្តងទៅ...', 'កុំធ្វើបែបនេះអី...')។\n"
                "   - ឈុតឌឺដង / ចំអក / មើលងាយ: ប្រើពាក្យចាក់ដោត បញ្ឆិតបញ្ឆៀង ឌឺដង ញាក់ចិញ្ចើម ឬចំអកឡកឡឺយ តែហាមប្រើ 'អញ'/'ឯង' (ឧ. 'អូហូ... គិតថាលោកខ្លាំងពូកែណាស់ទៅហ្ហេស?', 'ពិតជាគួរឱ្យអស់សំណើចមែន!')។\n"
                "   - ឈុតស្លន់ស្លោ / ភ័យខ្លាច / តក់ស្លុត: ប្រើពាក្យដាច់ៗ ញ័រ ឧទាន ភ័យស្លន់ស្លោ (ឧ. 'ស្អី... ស្អីគេ?!', 'មិន... មិនពិតទេ!', 'រត់... ប្រញាប់រត់ចេញទៅ!')។\n"
                "   - ឈុតបញ្ជា / អំណាច / ម៉ឺងម៉ាត់: ប្រើពាក្យម៉ឺងម៉ាត់ ថ្លៃថ្នូរ មានអំណាច មិនបត់បែន (ឧ. 'ស្តាប់បញ្ជា!', 'ធ្វើតាមភ្លាម!')។\n"
                "   - ឈុតកំប្លែង / ក្រមិចក្រមើម: ប្រើពាក្យរស់រវើក ក្រមិចក្រមើម គួរឱ្យចង់សើចបែបខ្មែរ។"
            ),
            "dramatic_conflict": (
                "【រចនាប័ទ្មក្ដៅក្រហាយ / ជម្លោះខ្លាំង (Intense Conflict)】៖ "
                "បកប្រែពាក្យពេចន៍ឱ្យមានកំហឹង ពុះកញ្ជ្រោល ក្ដៅគគុក តានតឹងខ្លាំង មុតស្រួច វាយប្រហារចំៗ និងមានអំណាច។ "
                "ហាមដាច់ខាតប្រើ 'អញ' ឬ 'អ្ហែង'! ជៀសវាង 'ឯង'! ត្រូវប្រើសព្វនាមម៉ឺងម៉ាត់ដូចជា 'អ្នក', 'លោក', 'យើង', 'ខ្ញុំ' ឬហៅឈ្មោះផ្ទាល់ ស្រែកគំហក ឬគំរាមកំហែងទៅតាមសាច់រឿងជម្លោះ។"
            ),
            "sad_tearful": (
                "【រចនាប័ទ្មកម្សត់ / ស្រក់ទឹកភ្នែក / ឈឺចាប់ (Sad Tragedy)】៖ "
                "បកប្រែពាក្យពេចន៍ឱ្យកម្សត់ សោកសៅ ស្រក់ទឹកភ្នែក ឈឺចាប់ដល់បាតបេះដូង អួលណែនក្នុងទ្រូង "
                "ប្រើសូរសំនៀងទន់ភ្លន់ អាណោចអាធ័ម ចាក់ដោតមនោសញ្ចេតនាយ៉ាងជ្រាលជ្រៅបំផុត ប្រើសព្វនាមសុភាព 'ខ្ញុំ - អ្នក', 'លោក', 'អ្នកនាង' ឬហៅឈ្មោះផ្ទាល់។ ហាមប្រើ 'អញ' ឬ 'ឯង'។"
            ),
            "sweet": (
                "【រចនាប័ទ្មផ្អែមល្ហែម / មនោសញ្ចេតនាស្នេហា】៖ "
                "ពាក្យពេចន៍ស៊ីជម្រៅ ត្រឹមត្រូវបំផុត ស្រទន់ ទន់ភ្លន់ លួងលោម ពិរោះពិសារ ចាក់ដោតដល់បេះដូង "
                "ប្រើពាក្យផ្អែមល្ហែមបែបសម័យទំនើប ហៅឈ្មោះគ្នាទៅវិញទៅមក ឬ 'ខ្ញុំ - អ្នក', 'ប្ដីសម្លាញ់ / ប្រពន្ធសម្លាញ់' ជៀសវាងការប្រើពាក្យ 'បង - អូន' ជាន់គ្នាដដែលៗគ្រប់ឃ្លា។ ហាមប្រើ 'អញ' និង 'ឯង'។"
            ),
            "historical": (
                "【រចនាប័ទ្មបុរាណ / យុទ្ធសិល្ប៍ / ក្បាច់គុនចិន】៖ "
                "បកប្រែឱ្យស៊ីជម្រៅ ត្រឹមត្រូវបំផុតតាមក្បួនខ្នាតបុរាណខ្មែរ មានអំណាច ស្វាហាប់ ឧត្តុង្គឧត្តម ប្រើពាក្យរាជសព្ទ និងពាក្យបុរាណជ្រាលជ្រៅតាមឋានៈតួអង្គ "
                "(ទូលបង្គំ, ព្រះអង្គ, ម្ចាស់ក្សត្រី, ព្រះពរ, ម្ចាស់បង, មេបញ្ជាការ, ចៅហ្វាយ, លោកម្ចាស់...)។"
            ),
            "comedy": (
                "【រចនាប័ទ្មកំប្លែង / ក្រមិចក្រមើម】៖ "
                "ត្រឹមត្រូវ ស៊ីជម្រៅ ក្រមិចក្រមើម ឆ្លាតវៃ រស់រវើក សើចសប្បាយបែបខ្មែរ។"
            ),
            "general": (
                "【រចនាប័ទ្មទូទៅ / ភាពយន្តសម័យ】៖ "
                "បកប្រែពាក្យពេចន៍ឱ្យត្រឹមត្រូវបំផុត ស៊ីជម្រៅ គ្រប់ជ្រុងជ្រោយ ពិរោះរណ្តំ ចាក់ដោតអារម្មណ៍ និងត្រូវតាមទឹកដមសំឡេងតួអង្គក្នុងវីដេអូ។"
            )
        }
        chosen_style = style_instructions.get(self.style, style_instructions.get("emotion_acting", style_instructions["general"]))

        return (
            "You are Cambodia's #1 premier cinematic scriptwriter, master dubbing director, and poetic translator specializing in Chinese to Khmer (ភាសាខ្មែរ).\n"
            "Your mission: Transform Chinese subtitles into PROFOUNDLY ACCURATE, DEEP, EXQUISITE, and 100% AUTHENTIC NATURAL SPOKEN KHMER calibrated to fit subtitle speech duration (បកប្រែពាក្យពេចន៍ឱ្យពិរោះរណ្តំ ធម្មជាតិបំផុត ពោរពេញដោយអារម្មណ៍តួអង្គ និងកម្រិតរយៈពេលនិយាយទាន់ Subtitle Timing)!\n\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            f"🎭 SELECTED DRAMA ACTING STYLE:\n{chosen_style}\n"
            "═══════════════════════════════════════════════════════════════════════════════\n\n"
            "🌟 MASTER PRINCIPLES OF AUTHENTIC KHMER DUBBING & RECAP STORYTELLING:\n"
            "1. NO PARENTHESES OR EXPLANATIONS (ហាមដាច់ខាតមិនឱ្យមានវង់ក្រចក និងការពន្យល់):\n"
            "   - Translate the EXACT dialogue/narration directly, clearly, accurately, and profoundly (បកប្រែឱ្យត្រូវតាមអត្ថន័យតួអក្សរ និងច្បាស់លាស់បំផុត)!\n"
            "   - NEVER output parentheses '(...)' or '（...）', and NEVER add translator explanations or commentary.\n"
            "2. MODERN CINEMATIC PRONOUN RULES (ច្បាប់សព្វនាមសុភាព ស៊ីវិល័យ និងពាក្យហាមឃាត់ដាច់ខាត):\n"
            "   - STRICTLY FORBIDDEN: ហាមដាច់ខាតមិនឱ្យប្រើពាក្យ 'អញ' និង 'អ្ហែង' ជាដាច់ខាត!\n"
            "   - AVOID 'ឯង': ជៀសវាងការប្រើពាក្យ 'ឯង' (ជំនួសដោយ 'អ្នក', 'លោក', 'អ្នកនាង', 'កញ្ញា', 'គ្នាយើង' ឬហៅឈ្មោះតួអង្គផ្ទាល់)។\n"
            "   - AVOID OVERUSING 'បង - អូន': ជៀសវាងការប្រើពាក្យ 'បង - អូន' ជាន់គ្នាដដែលៗគ្រប់ឃ្លា។ ជំនួសដោយ 'ខ្ញុំ - អ្នក', 'លោក', 'អ្នកនាង', ឬហៅឈ្មោះតួអង្គផ្ទាល់ (ឧ. 'ស៊ឺវុី', 'វ៉ាន់នីង', 'លោក 顾')។\n"
            "   - Married couple: ប្រើ 'ប្ដីសម្លាញ់ / ប្រពន្ធសម្លាញ់', 'លោកប្តី', ឬឈ្មោះផ្ទាល់។\n"
            "3. DIALOGUE CONTEXT, SPLIT-SENTENCE CONTINUITY & DEEP MEANING (យល់បរិបទសាច់រឿង មិនបកប្រែកាត់ទទឹង):\n"
            "   - Subtitle lines often SPLIT a single sentence across 2-3 consecutive subtitle lines. Connect their meaning seamlessly!\n"
            "     * E.g. '我让你帮客人' (Line A) + '找到真正适合他们的衣服' (Line B) -> 'ខ្ញុំចង់ឱ្យអ្នកជួយភ្ញៀវ' + 'ស្វែងរកសម្លៀកបំពាក់ដែលស័ក្តិសមនឹងពួកគេបំផុត!'\n"
            "     * E.g. When characters discuss bridesmaid list and one complains '...没有我' -> 'ចុះខ្ញុំវិញ? / គ្មានខ្ញុំនៅក្នុងហ្នឹងទេ?!' (NEVER 'បើគ្មានខ្ញុំទេ')!\n"
            "     * E.g. '你眼睛是尺子啊' -> 'ភ្នែកឯងដូចជាបន្ទាត់វាស់ចឹង?' / 'ភ្នែកអ្នកជាម៉ែត្រវាស់មែនទេ?' (NEVER broken characters like 'ែត្រហ្ហេស')!\n"
            "     * E.g. '我总不能反悔吧' -> 'ខ្ញុំមិនអាចត្រឡប់ពាក្យសម្តីវិញបានទេ!' (NEVER nonsensical 'ដោយស្បែកអី')!\n"
            "     * E.g. '茶水间' -> 'បន្ទប់សម្រាក / បន្ទប់ឆុងកាហ្វេ' (office pantry/break room, NEVER 'បន្ទប់ទឹក' bathroom)!\n"
            "     * E.g. '就是平时爱挑毛病' -> 'គ្រាន់តែជាទម្លាប់ចូលចិត្តចាប់កំហុសរើសអើងប៉ុណ្ណោះ!' (NEVER hallucinated numbers like 'មួយរយម្ភៃបី...')!\n"
            "   - NEVER TRANSLATE ROBOTICALLY OR LITERALLY WORD-BY-WORD! Understand the true intention, dramatic conflict, and character emotions first, then translate into natural, culturally fluent spoken Khmer!\n"
            "4. CHARACTER NAME CONSISTENCY (រក្សាឈ្មោះតួអង្គខ្មែរឱ្យថេរ ១០០% ពេញមួយរឿង):\n"
            "   - Transliterate Chinese names into standard, natural Khmer and KEEP THEM 100% CONSISTENT throughout all lines (e.g. 顾城舟 -> 'គូ ឆេងចូវ', 许志威 -> 'ស៊ឺ ជឺវុី' / 'ស៊ឺវុី', 唐婉宁 -> 'ថាង វ៉ាន់នីង')! NEVER invent different random names across lines!\n"
            "5. NATURAL SPOKEN FLOW & RECAP STORYTELLING (ភាសានិយាយ និងសម្រាយរឿងបែបធម្មជាតិ):\n"
            "   - Translate into rich, expressive, and melodious SPOKEN Khmer as used by professional Cambodian voice actors (មិនមែនជាភាសាសៀវភៅ ឬបកប្រែពាក្យតាមពាក្យឡើយ)!\n"
            "   - For story recap/narration, translate into captivating, natural, and smooth narrative Khmer.\n"
            "   - NEVER produce robotic literal translations or hang dialogue tags like '...និយាយថា៖' at line ends.\n"
            "   - Use authentic spoken particles ('ណា៎', 'ហ្នឹង', 'អី', 'តើ', 'ចុះ', 'ទៅណា?', 'មែនទេ?', 'អ្ហាស!', 'អញ្ចឹង')!\n"
            "5. CHINESE IDIOMS & PHRASES TRANSLATION GUIDE (ក្បួនបកប្រែពាក្យស្លោក និងកន្សោមពាក្យចិនញឹកញាប់):\n"
            "   - 大家别客气 / 别客气 -> 'កុំគិតច្រើនអី ញ៉ាំឱ្យអស់ដៃទៅ!' / 'កុំគួរសមអី!' (NEVER 'កុំក្រអឺតក្រទមអី')\n"
            "   - 本桌 / 这桌 -> 'តុនេះ' / 'វិក្កយបត្រតុនេះ' (NEVER 'សប្តាហ៍នេះ')\n"
            "   - 巴结 / 奉承 / 讨好 -> 'អែបអប' / 'លើកជើង' / 'យកចិត្តយកថ្លើម' (NEVER 'បាយអូសំប៉ែត')\n"
            "   - 挂失 -> 'រាយការណ៍ថាជ្រុះបាត់' / 'ផ្អាកកាត' (NEVER 'ដាក់ខលបែនចោល')\n"
            "   - 刷停车卡 / 打印停车票 -> 'ព្រីនសំបុត្រចតឡាន' / 'ស្កេនកាតចតឡាន' (NEVER 'វាយកាតយានដ្ឋាន')\n"
            "   - 没眼力见 / 不懂事 -> 'មិនចេះមើលកាលៈទេសៈសោះ' / 'មិនដឹងខ្យល់អី' (NEVER 'គ្មានភ្នែកមើលសោះ')\n"
            "   - 免单 / 免费 -> 'មិនគិតលុយ' / 'ឥតគិតថ្លៃ' (NEVER 'ហ្វ្រីស / ហريស')\n"
            "   - 你们还没明白过来？ / 还没懂吗？ -> 'អ្នកទាំងអស់គ្នានៅតែមិនទាន់យល់ការណ៍ពិតទៀតហ្ហ៎?' / 'ពួកអ្នកនៅតែមិនទាន់ភ្លឺភ្នែកទៀតមែនទេ?' (NEVER '...មិនយល់ទៀតហ្គោល?')\n"
            "   - 箭在弦上，不得不发 -> 'ព្រួញបាញ់ចេញពីធ្នូហើយ / ធ្លាក់ក្នុងស្ថានភាពពិបាកដកថយ'\n"
            "   - 打肿脸充胖子 -> 'គ្មានលុយ តែចង់អួតធ្វើជាអ្នកមាន / ចង់បានតែមុខមាត់'\n"
            "   - 死要面子 -> 'ខ្លាចបាក់មុខ / ចង់បានតែមុខមាត់'\n"
            "   - 帝王蟹 -> 'ក្តាមយក្សអាឡាស្កា / ក្តាមស្តេច'\n"
            "   - 鲍鱼 / 海参 -> 'ខ្យងប៉ាវហ៊ឺ / ត្រសក់សមុទ្រ'\n"
            "   - 澳龙 -> 'បង្កងអូស្ត្រាលី'\n"
            "5. NUMBERS, CURRENCY & AMOUNTS (បកប្រែចំនួនលុយ និងលេខឱ្យត្រឹមត្រូវតាមទម្លាប់ខ្មែរ):\n"
            "   - NEVER translate currency with literal decimal points like 'ចុចប្រាំមួយសូន្យសូន្យ'! Translate into real natural Khmer amounts:\n"
            "     * 313,600 / 31.36万 -> '៣១ ម៉ឺន ៣ ពាន់ ៦ រយ' (ឬ 'ជាង ៣០ ម៉ឺន')\n"
            "     * 30多万 -> 'ជាង ៣០ ម៉ឺន'\n"
            "     * 3,000 / 3千 -> '៣ ពាន់'\n"
            "     * 15,000 / 1.5万 -> '១ ម៉ឺន ៥ ពាន់'\n"
            "     * 50% / 5折 -> 'បញ្ចុះតម្លៃ ៥០%' (ឬ 'ពាក់កណ្តាលតម្លៃ')\n"
            "     * 3公斤 / 3kg -> '៣ គីឡូ'\n"
            "6. DURATION LIMIT (+70% of Subtitle Duration):\n"
            "   - Cambodian voice artists (ពិសិដ្ឋ for male / ស្រីម៉ុំ for female) speak at ~3.2 words (~15 Khmer characters) per second.\n"
            "   - Ensure your translated sentence fits comfortably within 'max_speech_sec' so voiceover timing is synchronized perfectly.\n"
            "7. ZERO-SKIP MANDATE (បកប្រែគ្រប់ជួរទាំងអស់ ហាមរំលង):\n"
            "   - Return EVERY SINGLE input subtitle ID with expressive, deep, and complete Khmer translation.\n"
            "   - For promo watermark lines (e.g. '点击查看完整视频', '完整版已上线', '点赞关注') -> 'សូមចុចទីនេះ ដើម្បីទស្សនាវីដេអូពេញ!' ឬពាក្យសមរម្យ។\n"
            "8. CHARACTER GENDER & VOICE CASTING (វិភាគភេទតួអង្គឱ្យត្រូវ ១០០%):\n"
            "   - For EACH line, accurately identify the true biological gender of the speaker:\n"
            "     * 'ស្រី' (Female): Women, ladies, brides, bridesmaids, sisters, daughters, mothers, female friends (闺蜜), female shop staff/assistants (店员).\n"
            "     * 'ប្រុស' (Male): Men, husbands, grooms, brothers, fathers, commanders, male bosses.\n"
            "   - In scenes between women, bridal shops, trying on dresses (e.g. 闺蜜, 伴娘, 小姐, 唐小姐, 许小姐/紫薇, 店员): ALL LINES ARE SPOKEN BY WOMEN ('ស្រី')! NEVER mark them as Male!\n"
            "9. OUTPUT FORMAT:\n"
            "   - Return ONLY a valid JSON object strictly matching:\n"
            "   {\n"
            "     \"translations\": [\n"
            "        {\n"
            "          \"id\": <int id matching input>,\n"
            "          \"khmer\": \"<profoundly accurate, deep, complete, and melodious translated dialogue in 100% pure Khmer>\",\n"
            "          \"gender\": \"<exact speaker gender: 'ស្រី' or 'ប្រុស'>\",\n"
            "          \"speaker\": \"<speaker name or role, e.g. 'តួស្រី', 'តួប្រុស'>\"\n"
            "        }\n"
            "     ]\n"
            "   }\n"
            "   - Keep output pure JSON with NO commentary or markdown code fences."
        )


    def extract_global_character_roster(
        self,
        items: List[Dict[str, Any]],
        status_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Global Screenplay Character Extraction (ស្កេនស្រង់តួអង្គ និងកំណត់ភេទទូទាំងរឿង):
        Reads key dialogue across the script to identify the main cast, their true biological genders,
        roles, and romantic/relational dynamics before chunk-based diarization and translation.
        Specifically solves the problem of:
        - Female characters with unisex/male-sounding names (e.g. 许志威 / 志威 / 亚男).
        - ASR homophone '他' in Chinese subtitles used for both men and women.
        - Inverted romantic pronouns (បង - អូន).
        """
        if not items:
            return {}

        if status_callback:
            try:
                status_callback("🎭 កំពុងស្កេនស្រង់តួអង្គ និងសាច់រឿងទូទាំងរឿង (Global Character Casting)...")
            except Exception:
                pass

        # Select representative lines: first 90 lines, 45 middle lines, 45 late lines
        total_len = len(items)
        sampled_items = []
        if total_len <= 250:
            sampled_items = items
        else:
            mid_start = total_len // 2 - 25
            late_start = max(0, total_len - 60)
            sampled_items.extend(items[:90])
            sampled_items.extend(items[mid_start:mid_start + 45])
            sampled_items.extend(items[late_start:])

        sample_dialogue = []
        for it in sampled_items:
            clean_d = re.sub(r"^\[[^\]]*\]\s*", "", str(it.get("text", ""))).strip()
            if clean_d:
                sample_dialogue.append({"id": it["id"], "text": clean_d})

        norm_model = normalize_gemini_model(self.model_name)
        models_to_try = [
            norm_model,
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
        ]
        seen = set()
        models = [m for m in models_to_try if not (m in seen or seen.add(m))]

        cast_system_prompt = (
            "You are an elite drama screenplay supervisor, dialogue attribution director, and casting expert.\n"
            "TASK: Analyze the provided drama dialogue lines and extract the COMPLETE CHARACTER ROSTER with their TRUE GENDERS and SPOKEN PRONOUN RULES.\n\n"
            "🌟 CRITICAL SCRIPT & SUBTITLE RULES:\n"
            "1. CHINESE ASR HOMOPHONE '他' (tā):\n"
            "   - Subtitles transcribed from spoken audio use '他' (tā) indiscriminately for BOTH female and male characters. NEVER assume a character is male simply because the subtitle writes '他'!\n"
            "2. UNISEX / MALE-SOUNDING FEMALE NAMES:\n"
            "   - Modern dramas frequently give female characters unisex or male-sounding names (e.g. 许志威 / 志威, 紫薇, 徐总, 亚男, 胜男, 丹, 楠, 凡, 婉宁).\n"
            "   - CRITICAL: In dramas where 许志威 (Xu Zhiwei) / 徐总 / 紫薇 is the female lead, bride's friend, or businesswoman, she is 100% FEMALE ('ស្រី' / 'Sreymom')! Do NOT assume she is male because of the name '志威'!\n"
            "   - ALWAYS check story anchors:\n"
            "     * 伴娘 (Bridesmaid), 小姐, 夫人, 太太, 闺蜜, 姐妹, 女员工, 小姑娘, 女朋友, 妈妈, 姐姐, 妹妹, 徐总 -> 100% FEMALE ('ស្រី' / 'Sreymom')!\n"
            "3. MALE ROLES:\n"
            "   - 顾城舟 (Gu Chengzhou), 顾廷山 (Gu Tingshan), 少爷, 先生, 男主, 董事长, 总裁, 经理, 老师, 老板, 爸爸, 哥哥, 弟弟, 兄弟 -> 100% MALE ('ប្រុស' / 'Piseth')!\n"
            "4. SPOKEN KHMER MODERN PRONOUN RULES:\n"
            "   - Strictly ban 'អញ' and 'អ្ហែង'. Avoid using 'ឯង'.\n"
            "   - Use polite terms: 'ខ្ញុំ - អ្នក', 'លោក', 'អ្នកនាង', or direct character names instead of repetitive 'បង - អូន'.\n"
            "   - Female friends / colleagues: Use polite conversational pronouns 'ខ្ញុំ - អ្នក', 'បងស្រី - ប្អូនស្រី', or direct character names, NEVER romantic 'បង - អូន'!\n\n"
            "Return strictly valid JSON matching:\n"
            "{\n"
            "  \"characters\": [\n"
            "    {\n"
            "      \"name\": \"Character Name\",\n"
            "      \"aliases\": [\"alias1\", \"alias2\"],\n"
            "      \"gender\": \"ស្រី\" | \"ប្រុស\",\n"
            "      \"role\": \"Role description in Khmer\",\n"
            "      \"romantic_pronoun\": \"អូន\" | \"បង\"\n"
            "    }\n"
            "  ],\n"
            "  \"rules\": [\n"
            "    \"Spoken pronoun and relationship rule in Khmer\"\n"
            "  ]\n"
            "}"
        )

        user_prompt = (
            f"Analyze these {len(sample_dialogue)} drama subtitle lines and return the character roster with 100% accurate biological genders:\n"
            f"{json.dumps(sample_dialogue, ensure_ascii=False, indent=1)}"
        )

        for attempt in range(1, 5):
            client, key_prev = self._get_next_client()
            active_model = models[(attempt - 1) % len(models)]
            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=cast_system_prompt,
                        temperature=0.1,
                        response_mime_type="application/json",
                        max_output_tokens=4096,
                    ),
                )
                raw_text = response.text.strip() if response and response.text else ""
                clean_json = raw_text
                if clean_json.startswith("```"):
                    clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                    clean_json = re.sub(r"\s*```$", "", clean_json)
                data = json.loads(clean_json)
                if data and isinstance(data, dict):
                    chars = data.get("characters", [])
                    rules = data.get("rules", [])
                    self.character_name_to_gender = {}
                    self.character_pronoun_rules = [str(r) for r in rules if r]
                    roster_dict = {}
                    for ch in chars:
                        c_name = ch.get("name", "").strip()
                        raw_g = str(ch.get("gender", "")).lower()
                        c_gen = "ស្រី" if ("ស្រី" in raw_g or "female" in raw_g or "woman" in raw_g or "女" in ch.get("gender", "")) else "ប្រុស"
                        c_role = ch.get("role", "")
                        c_aliases = ch.get("aliases", [])
                        if c_name:
                            roster_dict[c_name] = {
                                "gender": c_gen,
                                "role": c_role,
                                "aliases": c_aliases,
                                "romantic_pronoun": ch.get("romantic_pronoun", "អូន" if c_gen == "ស្រី" else "បង")
                            }
                            self.character_name_to_gender[c_name] = c_gen
                            for al in c_aliases:
                                al_str = str(al).strip()
                                if al_str:
                                    self.character_name_to_gender[al_str] = c_gen
                    self.global_character_roster = roster_dict
                    return roster_dict
            except Exception as e:
                err_str = str(e).lower()
                if "401" in err_str or "unauthenticated" in err_str:
                    self.mark_bad_key(key_prev, "401 Unauthenticated")
                elif "403" in err_str or "permission_denied" in err_str or "denied access" in err_str:
                    self.mark_bad_key(key_prev, "403 Permission Denied")
                time.sleep(0.3 * attempt)

        return {}

    def analyze_full_script_genders(
        self,
        items: List[Dict[str, Any]],
        concurrency: int = 6,
        chunk_size: int = 35,
        chunk_callback: Optional[Any] = None,
        status_callback: Optional[Any] = None
    ) -> Dict[int, str]:
        """
        Unified High-Precision Screenplay Diarization & Voice Casting:
        Attributes each dialogue line directly to Speaker ID ('Speaker 1', 'Speaker 2'...),
        Gender ('ប្រុស' / 'ស្រី'), Role ('មេបញ្ជាការ', 'បងស្រី'...), and Voice ('Piseth' / 'Sreymom')
        in a single unified pass with full conversational turn-taking context.
        """
        if not items:
            return {}

        total_items = len(items)
        gender_map: Dict[int, str] = {}
        self.last_speaker_map = {}
        line_speakers: Dict[int, Dict[str, str]] = {}

        # 1. Global Screenplay Character Roster Extraction (ស្កេនស្រង់តួអង្គទូទាំងរឿងជាមុន)
        if not self.global_character_roster:
            try:
                self.extract_global_character_roster(items, status_callback)
            except Exception:
                pass

        norm_model = normalize_gemini_model(self.model_name)
        models_to_try = [
            norm_model,
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
        ]
        seen = set()
        models = [m for m in models_to_try if not (m in seen or seen.add(m))]

        # -------------------------------------------------------------------------
        # UNIFIED HIGH-PRECISION SPEAKER & GENDER ANALYSIS (វិភាគ Speaker + ភេទ ប្រុស/ស្រី ដំណាលគ្នា)
        # -------------------------------------------------------------------------
        if status_callback:
            try:
                status_callback("⚡ កំពុងវិភាគសាច់រឿង តួអង្គ និងបែងចែកភេទ (ប្រុស Piseth / ស្រី Sreymom) ដោយ AI...")
            except Exception:
                pass

        # Build prompt guidelines from discovered global roster
        roster_guidelines = ""
        if self.global_character_roster:
            r_lines = []
            for c_name, c_info in self.global_character_roster.items():
                al_str = f" (Aliases: {', '.join(c_info.get('aliases', []))})" if c_info.get("aliases") else ""
                r_lines.append(f"- {c_name}{al_str}: {c_info.get('role', '')} -> GENDER: {c_info.get('gender', '')}")
            roster_guidelines = (
                "\n🌟 SCRIPT-SPECIFIC GLOBAL CHARACTER CASTING:\n"
                + "\n".join(r_lines) + "\n"
            )

        unified_system_prompt = (
            "You are an elite drama screenplay supervisor, dialogue attribution director, and voice casting director.\n"
            "TASK: For EACH subtitle line in the script, identify the EXACT speaker and biological gender (ប្រុស/ស្រី) with 100% precision.\n\n"
            + roster_guidelines + "\n"
            "🌟 CRITICAL RULES FOR ACCURATE DIARIZATION & GENDER DETECTION:\n"
            "0. SCENE-BASED SCREENPLAY BREAKDOWN (បែងចែកតាមឈុតឆាករឿង):\n"
            "   - Drama dialogue occurs in distinct scenes with specific active characters in each scene:\n"
            "   - SCENE WITH ONLY MALE CHARACTERS (e.g. 顾廷山 & 顾城舟, 董事会): ALL lines in this scene belong to MALE ('ប្រុស' / 'Piseth')! Zero female lines!\n"
            "   - SCENE WITH ONLY FEMALE CHARACTERS (e.g. 许志威 & 助理/店员, 闺蜜/伴娘): ALL lines in this scene belong to FEMALE ('ស្រី' / 'Sreymom')! Zero male lines!\n"
            "   - SCENE BETWEEN 1 MALE & 1 FEMALE (e.g. 顾城舟 & 许志威): Strictly alternate between Gu (ប្រុស) and Zhiwei (ស្រី). In proposal scenes, Gu asks and Zhiwei agrees!\n"
            "   - MONOLOGUES (e.g. 许志威 reflecting on life, monologue/narration): Belongs 100% to that character's gender ('ស្រី')!\n"
            "1. SCRIPT CONVERSATION CONTINUITY & MULTI-LINE MONOLOGUES:\n"
            "   - In subtitle files, a single speaker frequently speaks 2, 3, or more consecutive subtitle lines (split sentences or monologues).\n"
            "   - KEEP THE SAME SPEAKER AND GENDER for consecutive lines by the same person! DO NOT artificially alternate speakers on every line!\n"
            "   - Only switch speaker when another character actually speaks, replies, or reacts.\n"
            "2. ADDRESSEE VS SPEAKER (NEVER CONFUSE LISTENER WITH SPEAKER):\n"
            "   - When a character says '许小姐说的对' (Miss Xu is right) or '唐小姐您朋友...', the speaker is the STORE CLERK / ASSISTANT (店员), NEVER Miss Xu or Miss Tang! A person does NOT say 'Miss Xu is right' about themselves!\n"
            "   - When a character says '紫薇这件怎么样' (Ziwei, how is this?), the speaker is the other person (唐婉宁), talking TO Ziwei!\n"
            "   - When a line lists people '方小曼大学时借了婉宁3000块一直没还... 周茜去年说她整过鼻子... 孙婷婷是他前男友女朋友... 没有我': This is the PROTAGONIST'S INNER MONOLOGUE / NARRATION (许志威/紫薇, Female), NOT the people being mentioned!\n"
            "3. CONVERSATION TURNS & TELEPHONE CALLS:\n"
            "   - Track dialogue turns between caller and receiver. When bride complains on phone '到底你跟谁结婚？怎么什么都听他的？', the speaker is the BRIDE (Female), NOT the groom!\n"
            "4. CHINESE ASR HOMOPHONE '他' (tā):\n"
            "   - Chinese subtitle transcription tools transcribe all third-person pronouns as '他'. NEVER assume a character is male simply because the subtitle writes '他'!\n"
            "5. UNISEX & BOYISH FEMALE NAMES:\n"
            "   - Modern dramas often give female characters boyish or unisex names (e.g. 许志威 / 志威, 紫薇, 徐总, 亚男, 胜男, 丹, 楠, 凡). In this drama, 许志威 (Xu Zhiwei) / 徐总 / 紫薇 is the FEMALE LEAD, bride's best friend, and businesswoman -> 100% FEMALE ('ស្រី' / 'Sreymom')! 顾城舟 is MALE ('ប្រុស' / 'Piseth')! 顾廷山 is MALE ('ប្រុស' / 'Piseth')!\n"
            "6. SINGLE NARRATOR / RECAP SCRIPTS:\n"
            "   - If the video is a story recap, commentary, documentary, or single-voice narration, ALL narrative commentary belongs to the NARRATOR with ONE consistent gender (ប្រុស or ស្រី).\n"
            "7. EXPLICIT GENDER INDICATORS:\n"
            "   - FEMALE ('ស្រី' / 'Sreymom'):\n"
            "     * 1st-person self-referrals: 奴婢, 妾身, 小女, 本姑娘, 老娘, 本宫, 妹妹, 女儿\n"
            "     * Addressing others: 哥, 哥哥, 夫君, 老公, 爸爸, 爹, 叔叔, 少爷, 公子 (addressed to male)\n"
            "     * Roles/Family: 姐, 妹, 妈/娘, 小姐, 夫人, 太太, 闺蜜, 伴娘, 女员工, 女儿, 妻/媳妇, 奶奶, 姑姑, 店员\n"
            "     * Khmer words: បងស្រី, អ្នកនាង, ម៉ាក់, កូនស្រី, អូន, ស្រី, នាង\n"
            "   - MALE ('ប្រុស' / 'Piseth'):\n"
            "     * 1st-person self-referrals: 老子, 本王, 本座, 卑职, 末将, 为父, 为师, 哥哥, 爸爸, 俺, 老夫\n"
            "     * Addressing others: 妹, 妹妹, 夫人, 娘子, 老婆, 妈妈, 娘, 姑娘, 小姐, 宝贝女儿 (addressed to female)\n"
            "     * Roles/Family: 哥, 弟, 爸/爹, 爷, 叔, 夫/丈夫, 少爷, 公子, 总/主管, 董事长, 司令, 将军, 侍卫, 兄弟, 新郎, 老师, 老板\n"
            "     * Khmer words: បងប្រុស, លោកមេបញ្ជាការ, លោកម្ចាស់, ប្រុស, គាត់\n"
            "8. SPEAKER CONSISTENCY:\n"
            "   - Each speaker (唐婉宁, 许志威, 顾老师, 新郎, 店员) MUST maintain consistent biological gender throughout the script.\n\n"
            "Return strictly valid JSON matching:\n"
            "{\n"
            "  \"items\": [\n"
            "     {\"id\": 1, \"speaker\": \"唐婉宁\", \"gender\": \"ស្រី\", \"role\": \"新娘\", \"voice\": \"Sreymom\"},\n"
            "     {\"id\": 2, \"speaker\": \"许志威\", \"gender\": \"ស្រី\", \"role\": \"伴娘/闺蜜\", \"voice\": \"Sreymom\"},\n"
            "     {\"id\": 3, \"speaker\": \"顾老师\", \"gender\": \"ប្រុស\", \"role\": \"设计师/老板\", \"voice\": \"Piseth\"}\n"
            "  ]\n"
            "}"
        )

        def extract_speaker_name(val_str: Any) -> str:
            s = str(val_str).strip()
            if s.lower().startswith("speaker"):
                m_num = re.search(r'\d+', s)
                return f"Speaker {m_num.group(0)}" if m_num else "Speaker 1"
            m_num = re.search(r'\d+', s)
            if m_num:
                return f"Speaker {m_num.group(0)}"
            return s if s else "Speaker 1"

        def parse_unified_response(raw_text: str) -> Dict[int, Dict[str, Any]]:
            parsed: Dict[int, Dict[str, Any]] = {}
            if not raw_text:
                return parsed

            clean_json = raw_text.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json).strip()

            data = None
            try:
                data = json.loads(clean_json)
            except Exception:
                obj_m = re.search(r'\{[\s\S]*\}', raw_text)
                if obj_m:
                    try:
                        data = json.loads(obj_m.group(0))
                    except Exception:
                        pass
                if data is None:
                    list_m = re.search(r'\[[\s\S]*\]', raw_text)
                    if list_m:
                        try:
                            arr = json.loads(list_m.group(0))
                            if isinstance(arr, list):
                                data = arr
                        except Exception:
                            pass

            if data is None:
                return parsed

            raw_items = None
            if isinstance(data, list):
                raw_items = data
            elif isinstance(data, dict):
                for candidate_k in ["items", "speakers", "dialogues", "dialogue", "lines", "subtitles", "results", "data"]:
                    if candidate_k in data and isinstance(data[candidate_k], (list, dict)):
                        raw_items = data[candidate_k]
                        break
                if raw_items is None:
                    raw_items = data

            def normalize_gender(g_val: Any, v_val: Any = "") -> str:
                g_str = str(g_val or "").strip().lower()
                v_str = str(v_val or "").strip().lower()
                if (
                    "ស្រី" in g_str
                    or "女" in g_str
                    or "female" in g_str
                    or "woman" in g_str
                    or g_str in ["f", "girl"]
                    or "sreymom" in v_str
                    or "mom" in v_str
                ):
                    return "ស្រី"
                return "ប្រុស"

            # Format A: List of items
            if isinstance(raw_items, list):
                for entry in raw_items:
                    if isinstance(entry, dict):
                        raw_id = entry.get("id") or entry.get("line_id") or entry.get("sub_id") or entry.get("index")
                        if raw_id is not None:
                            try:
                                m_id = re.search(r'\d+', str(raw_id))
                                if m_id:
                                    sid_int = int(m_id.group(0))
                                    spk_name = extract_speaker_name(entry.get("speaker") or entry.get("speaker_id") or entry.get("name") or "Speaker 1")
                                    gender = normalize_gender(entry.get("gender"), entry.get("voice"))
                                    voice = "Sreymom" if gender == "ស្រី" else "Piseth"
                                    role = str(entry.get("role") or entry.get("character") or "").strip()
                                    parsed[sid_int] = {
                                        "speaker": spk_name,
                                        "gender": gender,
                                        "voice": voice,
                                        "role": role,
                                        "pitch": "+0Hz"
                                    }
                            except Exception:
                                pass

            # Format B: Dict of {id: info}
            elif isinstance(raw_items, dict):
                for k, v in raw_items.items():
                    k_str = str(k).strip()
                    try:
                        m_id = re.search(r'\d+', k_str)
                        if not m_id:
                            continue
                        sid_int = int(m_id.group(0))
                        if isinstance(v, dict):
                            spk_name = extract_speaker_name(v.get("speaker") or v.get("speaker_id") or "Speaker 1")
                            gender = normalize_gender(v.get("gender"), v.get("voice"))
                            voice = "Sreymom" if gender == "ស្រី" else "Piseth"
                            role = str(v.get("role") or v.get("character") or "").strip()
                            parsed[sid_int] = {
                                "speaker": spk_name,
                                "gender": gender,
                                "voice": voice,
                                "role": role,
                                "pitch": "+0Hz"
                            }
                        elif isinstance(v, str):
                            gender = normalize_gender(v)
                            voice = "Sreymom" if gender == "ស្រី" else "Piseth"
                            parsed[sid_int] = {
                                "speaker": "Speaker 1" if gender == "ប្រុស" else "Speaker 2",
                                "gender": gender,
                                "voice": voice,
                                "role": "",
                                "pitch": "+0Hz"
                            }
                    except Exception:
                        pass

            return parsed

        def analyze_chunk(
            segment: List[Dict[str, Any]],
            preceding_context: Optional[List[Dict[str, Any]]] = None,
            known_roster: Optional[Dict[str, Dict[str, Any]]] = None
        ) -> Dict[int, Dict[str, Any]]:
            c_data = []
            for it in segment:
                clean_d = re.sub(r"^\[[^\]]*\]\s*", "", str(it.get("text", ""))).strip()
                entry = {"id": it["id"], "dialogue": clean_d if clean_d else it.get("text", "")}
                if it.get("khmer"):
                    clean_k = re.sub(r"^\[[^\]]*\]\s*", "", str(it.get("khmer", ""))).strip()
                    if clean_k and clean_k != clean_d:
                        entry["khmer_translation"] = clean_k
                c_data.append(entry)

            prompt_parts = []
            if known_roster:
                roster_desc = ", ".join([f"{k}: {v.get('role', '')} ({v.get('gender', '')})" for k, v in known_roster.items() if v.get("role")])
                if roster_desc:
                    prompt_parts.append(f"KNOWN ACTIVE CHARACTERS IN SCRIPT:\n{roster_desc}\n")

            if preceding_context:
                ctx_lines = [f"[Line {p['id']} - {p.get('speaker', 'Speaker 1')} ({p.get('gender', 'ប្រុស')})]: {p.get('dialogue', '')}" for p in preceding_context[-12:]]
                prompt_parts.append("RECENT PRECEDING DIALOGUE CONTEXT (for conversational turn-taking continuity):\n" + "\n".join(ctx_lines) + "\n")

            prompt_parts.append(
                f"Attribute character speaker and gender (ប្រុស/ស្រី) for ALL {len(segment)} dialogue lines below.\n"
                f"Carefully track conversational continuity and speaker turns:\n"
                f"{json.dumps(c_data, ensure_ascii=False, indent=1)}\n\n"
                f"Return JSON matching: {{\"items\": [{{\"id\": {segment[0]['id']}, \"speaker\": \"Character Name or Speaker ID\", \"gender\": \"ប្រុស\"|\"ស្រី\", \"role\": \"Role description in Khmer\", \"voice\": \"Piseth\"|\"Sreymom\"}}]}}."
            )
            c_prompt = "\n".join(prompt_parts)

            for attempt in range(1, 6):
                client, key_prev = self._get_next_client()
                active_model = models[(attempt - 1) % len(models)]
                try:
                    response = client.models.generate_content(
                        model=active_model,
                        contents=c_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=unified_system_prompt,
                            temperature=0.1,
                            response_mime_type="application/json",
                            max_output_tokens=8192
                        )
                    )
                    raw_text = response.text.strip() if response and response.text else ""
                    parsed = parse_unified_response(raw_text)
                    if parsed:
                        return parsed
                except Exception as e:
                    err_str = str(e).lower()
                    if "401" in err_str or "unauthenticated" in err_str:
                        self.mark_bad_key(key_prev, "401 Unauthenticated")
                    elif "403" in err_str or "permission_denied" in err_str or "denied access" in err_str:
                        self.mark_bad_key(key_prev, "403 Permission Denied")
                    time.sleep(0.3 * attempt)
            return {}

        # Process in chunks with parallel execution across worker threads (10x - 20x speedup)
        seg_size = max(80, min(chunk_size, 150))
        chunks = []
        for i in range(0, len(items), seg_size):
            seg = items[i:i + seg_size]
            preceding = []
            if i > 0:
                for p_item in items[max(0, i - 10):i]:
                    d_txt = re.sub(r"^\[[^\]]*\]\s*", "", str(p_item.get("text", ""))).strip()
                    preceding.append({
                        "id": p_item["id"],
                        "dialogue": d_txt,
                        "speaker": "Speaker",
                        "gender": "Unknown"
                    })
            chunks.append((seg, preceding))

        all_line_analysis: Dict[int, Dict[str, Any]] = {}
        analysis_lock = threading.Lock()
        completed_count = 0

        max_workers = min(concurrency or 6, len(chunks)) if chunks else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(analyze_chunk, seg, prec, dict(self.global_character_roster)): seg
                for seg, prec in chunks
            }
            for future in as_completed(future_to_chunk):
                seg = future_to_chunk[future]
                try:
                    res = future.result()
                except Exception:
                    res = {}
                with analysis_lock:
                    all_line_analysis.update(res)
                    completed_count += len(seg)
                    if chunk_callback:
                        try:
                            cur_map = {sid: info["gender"] for sid, info in all_line_analysis.items()}
                            chunk_callback(cur_map, min(completed_count, total_items), total_items)
                        except Exception:
                            pass

        # High-precision line-level refinement and fallback
        strict_female_self = ["本宫", "老娘", "妾身", "奴婢", "小女", "本姑娘", "小姐", "妹妹", "妈妈", "夫人"]
        strict_male_self = ["老子", "末将", "卑职", "本王", "本座", "爷", "俺", "老夫", "为父", "为师", "哥哥", "爸爸"]

        global_speaker_stats: Dict[str, Dict[str, Any]] = {}
        prev_gender = "ប្រុស"
        prev_speaker = "Speaker 1"

        for it in items:
            sid = it["id"]
            zh_text = str(it.get("text", ""))
            km_text = str(it.get("khmer", ""))

            if sid in all_line_analysis:
                info = all_line_analysis[sid]
                spk = info.get("speaker", "Speaker 1")
                g = info.get("gender", "ប្រុស")
                r = info.get("role", "")
            else:
                # Missing from AI analysis -> intelligent contextual fallback
                if any(w in zh_text for w in strict_female_self) or any(w in km_text for w in ["ចង់ឱ្យបងស្រី", "ដើរតាមបងស្រី", "បងស្រីនឹង", "នាង", "អ្នកនាង"]):
                    spk = "Speaker 2"
                    g = "ស្រី"
                    r = "តួស្រី"
                elif any(w in zh_text for w in strict_male_self) or any(w in km_text for w in ["បងប្រុស", "លោកម្ចាស់", "គាត់"]):
                    spk = "Speaker 1"
                    g = "ប្រុស"
                    r = "តួប្រុស"
                else:
                    # PRESERVE continuity: Keep previous speaker and gender! DO NOT artificially flip!
                    spk = prev_speaker
                    g = prev_gender
                    r = f"តួអង្គ {spk}"

            # Strict 1st-person self-referral override (unequivocal)
            if any(w in zh_text for w in ["本宫", "老娘", "妾身", "奴婢", "小女", "本姑娘"]):
                g = "ស្រី"
            elif any(w in zh_text for w in ["老子", "末将", "卑职", "本王", "本座", "老夫"]):
                g = "ប្រុស"

            # Guaranteed character name override from Global Screenplay Roster
            # Match SPEAKER name (spk) or character role (r) - NEVER spoken dialogue text (zh_text)!
            if self.character_name_to_gender:
                matched_gen = None
                # 1. Match SPEAKER name (spk) to known character roster
                for c_name, c_gen in self.character_name_to_gender.items():
                    if spk and (c_name == spk or c_name in spk):
                        matched_gen = c_gen
                        break
                # 2. Match role description if it explicitly names the character
                if not matched_gen and r:
                    for c_name, c_gen in self.character_name_to_gender.items():
                        if c_name in r:
                            matched_gen = c_gen
                            break
                # 3. Only match in zh_text if it is an explicit dialogue speaker prefix (e.g. "婉宁：" or "[婉宁]")
                if not matched_gen:
                    for c_name, c_gen in self.character_name_to_gender.items():
                        if re.match(rf"^(?:\[)?{re.escape(c_name)}(?:\])?\s*[:：]", zh_text):
                            matched_gen = c_gen
                            break
                if matched_gen:
                    g = matched_gen

            prev_gender = g
            prev_speaker = spk

            v = "Sreymom" if g == "ស្រី" else "Piseth"
            p = "+0Hz"

            gender_map[sid] = g
            self.last_speaker_map[sid] = {
                "speaker": spk,
                "gender": g,
                "role": r,
                "voice": v,
                "pitch": p
            }

            # Accumulate speaker metadata
            if spk not in global_speaker_stats:
                global_speaker_stats[spk] = {"roles": [], "genders": [], "count": 0}
            global_speaker_stats[spk]["count"] += 1
            global_speaker_stats[spk]["genders"].append(g)
            if r:
                global_speaker_stats[spk]["roles"].append(r)

        # Build consistent character roster
        roster: Dict[str, Dict[str, Any]] = {}
        female_role_keywords = ["ស្រី", "female", "woman", "girl", "女", "娘", "妹", "姐", "闺蜜", "伴娘", "小姐", "太太", "夫人", "新娘"]
        male_role_keywords = ["ប្រុស", "male", "man", "boy", "男", "哥", "弟", "爷", "爸", "夫", "郎", "叔", "伯", "少爷", "公子", "新郎", "先生"]
        for spk, stats in global_speaker_stats.items():
            # If this speaker is in the global roster, use the roster's definitive gender!
            if spk in self.character_name_to_gender:
                spk_g = self.character_name_to_gender[spk]
            elif any(c_name in spk for c_name in self.character_name_to_gender):
                matched = [self.character_name_to_gender[c_name] for c_name in self.character_name_to_gender if c_name in spk]
                spk_g = matched[0]
            elif any(any(kw in r for kw in female_role_keywords) for r in stats["roles"]):
                spk_g = "ស្រី"
            elif any(any(kw in r for kw in male_role_keywords) for r in stats["roles"]):
                spk_g = "ប្រុស"
            else:
                female_votes = stats["genders"].count("ស្រី")
                male_votes = stats["genders"].count("ប្រុស")
                spk_g = "ស្រី" if female_votes > male_votes else "ប្រុស"
            spk_v = "Sreymom" if spk_g == "ស្រី" else "Piseth"
            common_role = max(set(stats["roles"]), key=stats["roles"].count) if stats["roles"] else f"តួអង្គ {spk}"
            roster[spk] = {
                "role": common_role,
                "gender": spk_g,
                "voice": spk_v,
                "pitch": "+0Hz"
            }

        # CRITICAL: Harmonize all lines belonging to the same named character so they have 100% consistent gender!
        # DO NOT blindly override generic "Speaker 1" if the AI explicitly assigned different genders to different lines!
        for sid, sinfo in self.last_speaker_map.items():
            spk = sinfo.get("speaker", "Speaker 1")
            # Only harmonize if it's a specific named character, NOT a generic placeholder like "Speaker 1" or "Speaker"
            if spk in roster and not re.match(r"^Speaker\s*\d*$", spk, re.IGNORECASE):
                consistent_g = roster[spk]["gender"]
                sinfo["gender"] = consistent_g
                sinfo["voice"] = roster[spk]["voice"]
                gender_map[sid] = consistent_g

        self.last_characters = roster
        self.global_character_roster = roster

        if chunk_callback:
            try:
                chunk_callback(gender_map, total_items, total_items)
            except Exception:
                pass

        return gender_map


    def _call_gemini_raw(
        self,
        items: List[Dict[str, Any]],
        gender_hints: Optional[Dict[int, str]] = None,
        max_retries: int = 8,
        preceding_context: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[int, Dict[str, str]]:
        prompt_data = []
        for item in items:
            p_entry = {"id": item["id"], "chinese": item["text"]}
            if "duration_sec" in item:
                p_entry["duration_sec"] = item["duration_sec"]
                p_entry["max_speech_sec"] = item.get("max_speech_sec", round(item["duration_sec"] * 1.7, 2))
            if gender_hints and item["id"] in gender_hints:
                p_entry["gender_hint"] = gender_hints[item["id"]]
            prompt_data.append(p_entry)

        roster_prompt_section = ""
        if self.global_character_roster:
            r_items = []
            for c_name, c_info in self.global_character_roster.items():
                if isinstance(c_info, dict):
                    r_items.append(f"- {c_name}: {c_info.get('gender', '')} ({c_info.get('role', '')}) -> Romantic Pronoun: {c_info.get('romantic_pronoun', '')}")
            
            rules_text = ""
            if self.character_pronoun_rules:
                rules_text = "\nSPECIAL SCRIPT RULES:\n" + "\n".join([f"* {r}" for r in self.character_pronoun_rules])

            roster_prompt_section = (
                "\n🌟 CRITICAL CHARACTER ROSTER & POLITE PRONOUN RULES (ច្បាប់តួអង្គ និងសព្វនាមសុភាពរាបសារ):\n"
                + "\n".join(r_items)
                + rules_text + "\n"
                + "1. STRICT PROHIBITION: NEVER use 'អញ' or 'អ្ហែង'! Avoid using 'ឯង' (use 'អ្នក', 'លោក', 'អ្នកនាង', or direct character names instead).\n"
                + "2. AVOID REPETITIVE 'បង - អូន': Modern cinematic style strictly avoids overusing 'បង - អូន'. Prefer polite pronouns 'ខ្ញុំ - អ្នក', 'លោក', 'អ្នកនាង', or address characters by name directly (e.g. ហៅឈ្មោះផ្ទាល់)!\n"
                + "3. WIFE ADDRESSING HUSBAND: When a woman addresses her husband ('老公') -> 'ប្ដីសម្លាញ់' / 'លោកប្តី' / direct name, NEVER 'បងប្រុស'!\n"
                + "4. FEMALE FRIENDS & COLLEAGUES: Use polite conversational pronouns 'ខ្ញុំ - អ្នក', 'គ្នា', 'បងស្រី', or direct character names, NEVER romantic or vulgar terms!\n"
            )

        ctx_section = ""
        if preceding_context:
            ctx_lines = [f"[Line {p['id']}]: {p.get('text', '')}" for p in preceding_context[-6:] if p.get('text')]
            if ctx_lines:
                ctx_section = "\nPRECEDING LINES (for dialogue flow continuity only, do NOT re-translate):\n" + "\n".join(ctx_lines) + "\n"

        user_prompt = (
            f"Translate ALL the following {len(items)} Chinese subtitle lines into natural, expressive, and authentic spoken Khmer (ភាសាខ្មែរ).\n"
            f"{roster_prompt_section}\n"
            f"{ctx_section}\n"
            f"CRITICAL TIMING, CHARACTER EMOTION & SPEECH DURATION CONSTRAINTS (ក្បួនកម្រិតរយៈពេលនិយាយ និងអារម្មណ៍តួអង្គក្នុងវីដេអូ):\n"
            f"1. DURATION LIMIT: For each line, the speech duration of Piseth (ពិសិដ្ឋ) or Sreymom (ស្រីម៉ុំ) MUST NOT EXCEED 'max_speech_sec' (duration_sec + 70%). Calibrate your Khmer phrasing to match (~3 words or ~15 characters per second of max_speech_sec)!\n"
            f"2. DYNAMIC CHARACTER EMOTION (ទម្រង់និយាយពាក្យពេចន៍ត្រូវនឹងអារម្មណ៍តួអង្គក្នុងវីដេអូ):\n"
            f"   - Match the emotional tone and spoken intensity of each character (កំហឹង, សោកសៅ, ឌឺដង, អង្វរករ, ភ័យខ្លាច, ផ្អែមល្ហែម, បញ្ជា, សប្បាយរីករាយ) so the spoken Khmer dialogue sounds 100% natural, expressive, and deeply authentic to the scene!\n"
            f"3. PROFOUND ACCURACY & DRAMATIC CONTINUITY (បកប្រែឱ្យត្រូវអត្ថន័យ និងយល់សាច់រឿងបន្តគ្នា):\n"
            f"   - NEVER translate words in isolation or mechanically word-by-word! Understand the true drama scene, character motives, and what each line truly means in context.\n"
            f"   - When lines are split sentences (e.g. Line A is the first half, Line B is the second half), connect them seamlessly in natural, grammatically correct Khmer!\n"
            f"   - Strictly calibrate sentence length so it fits within max_speech_sec!\n"
            f"4. You MUST return exactly {len(items)} items matching every single input id with zero skips or omissions.\n"
            f"5. Strictly maintain the exact order row-by-row.\n\n"
            f"{json.dumps(prompt_data, ensure_ascii=False, indent=2)}"
        )

        system_instruction = self._get_system_prompt()
        last_err = None

        # Fallback model list across multiple independent Google server clusters to eliminate 503
        norm_model = normalize_gemini_model(self.model_name)
        models_to_try = [
            norm_model,
            "gemini-3.8-flash",
            "gemini-flash-lite-latest",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
        ]
        seen = set()
        fallback_models = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for attempt in range(1, max_retries + 1):
            client, key_prev = self._get_next_client()
            active_model = fallback_models[(attempt - 1) % len(fallback_models)]

            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                        response_mime_type="application/json",
                        max_output_tokens=8192,
                    )
                )

                raw_text = response.text.strip() if response and response.text else ""
                clean_json = raw_text
                if clean_json.startswith("```"):
                    clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                    clean_json = re.sub(r"\s*```$", "", clean_json)

                data = None
                try:
                    data = json.loads(clean_json)
                except Exception:
                    obj_m = re.search(r'\{[\s\S]*\}', raw_text)
                    if obj_m:
                        try:
                            data = json.loads(obj_m.group(0))
                        except Exception:
                            pass
                    if not data:
                        list_m = re.search(r'\[[\s\S]*\]', raw_text)
                        if list_m:
                            try:
                                arr = json.loads(list_m.group(0))
                                if isinstance(arr, list):
                                    data = {"translations": arr}
                            except Exception:
                                pass

                if not data or not isinstance(data, dict):
                    raise ValueError(f"Could not parse valid JSON from Gemini response: {raw_text[:180]}")

                translations_list = data.get("translations", [])
                if not translations_list and isinstance(data.get("items"), list):
                    translations_list = data.get("items")

                batch_result = {}
                expected_ids = [item["id"] for item in items]
                expected_id_set = set(expected_ids)
                matched_ids = set()
                unmatched_translations = []

                # Step 1: Match strictly by expected ID
                for t in translations_list:
                    raw_id = t.get("id")
                    if raw_id is not None:
                        try:
                            tid = int(raw_id)
                            if tid in expected_id_set and tid not in matched_ids:
                                khmer_text = sanitize_khmer_text(str(t.get("khmer", "")))
                                g_raw = str(t.get("gender", "")).strip()
                                spk = str(t.get("speaker", "")).strip()

                                # Priority: explicit hint > roster lookup for speaker > model gender output > keywords > fallback
                                if gender_hints and tid in gender_hints:
                                    gender = gender_hints[tid]
                                elif self.character_name_to_gender and any(c_name in spk for c_name in self.character_name_to_gender if c_name):
                                    gender = next(self.character_name_to_gender[c_name] for c_name in self.character_name_to_gender if c_name in spk)
                                elif "ស្រី" in g_raw or "female" in g_raw.lower() or "woman" in g_raw.lower() or "女" in g_raw or g_raw.upper() in ["F", "WOMAN"]:
                                    gender = "ស្រី"
                                elif "ប្រុស" in g_raw or "male" in g_raw.lower() or "man" in g_raw.lower() or "男" in g_raw or g_raw.upper() in ["M", "MAN"]:
                                    gender = "ប្រុស"
                                elif self.character_name_to_gender:
                                    gender = ""
                                    item_zh = str(next((it["chinese"] for it in prompt_data if it["id"] == tid), ""))
                                    for c_name, c_g in self.character_name_to_gender.items():
                                        if c_name in spk or c_name in item_zh:
                                            gender = c_g
                                            break
                                    if not gender:
                                        if any(w in item_zh for w in ["小姐", "夫人", "太太", "闺蜜", "伴娘", "妹妹", "姐姐", "妈妈", "女", "妹", "姐"]):
                                            gender = "ស្រី"
                                        elif any(w in item_zh for w in ["少爷", "先生", "哥哥", "弟弟", "爸爸", "男", "哥", "叔"]):
                                            gender = "ប្រុស"
                                        else:
                                            gender = "ស្រី" if ("ស្រី" in g_raw or "female" in g_raw.lower()) else "ប្រុស"
                                else:
                                    item_zh = str(next((it["chinese"] for it in prompt_data if it["id"] == tid), ""))
                                    if any(w in item_zh for w in ["小姐", "夫人", "太太", "闺蜜", "伴娘", "妹妹", "姐姐", "妈妈", "女", "妹", "姐"]):
                                        gender = "ស្រី"
                                    else:
                                        gender = "ស្រី" if ("ស្រី" in g_raw or "female" in g_raw.lower()) else "ប្រុស"

                                batch_result[tid] = {
                                    "khmer": khmer_text,
                                    "gender": gender,
                                    "speaker": spk
                                }
                                matched_ids.add(tid)
                                continue
                        except Exception:
                            pass
                    unmatched_translations.append(t)

                # Step 2: Positional fallback for items where LLM renumbered (e.g. 1..N) or omitted IDs
                unmatched_idx = 0
                for item in items:
                    iid = item["id"]
                    if iid not in batch_result:
                        if unmatched_idx < len(unmatched_translations):
                            t = unmatched_translations[unmatched_idx]
                            unmatched_idx += 1
                            khmer_text = sanitize_khmer_text(str(t.get("khmer", "")))
                            g_raw = str(t.get("gender", "")).strip()
                            spk = str(t.get("speaker", "")).strip()

                            if gender_hints and iid in gender_hints:
                                gender = gender_hints[iid]
                            elif self.character_name_to_gender and any(c_name in spk for c_name in self.character_name_to_gender if c_name):
                                gender = next(self.character_name_to_gender[c_name] for c_name in self.character_name_to_gender if c_name in spk)
                            elif "ស្រី" in g_raw or "female" in g_raw.lower() or "woman" in g_raw.lower() or "女" in g_raw or g_raw.upper() in ["F", "WOMAN"]:
                                gender = "ស្រី"
                            elif "ប្រុស" in g_raw or "male" in g_raw.lower() or "man" in g_raw.lower() or "男" in g_raw or g_raw.upper() in ["M", "MAN"]:
                                gender = "ប្រុស"
                            elif self.character_name_to_gender:
                                gender = ""
                                item_zh = str(item.get("text", ""))
                                for c_name, c_g in self.character_name_to_gender.items():
                                    if c_name in spk or c_name in item_zh:
                                        gender = c_g
                                        break
                                if not gender:
                                    if any(w in item_zh for w in ["小姐", "夫人", "太太", "闺蜜", "伴娘", "妹妹", "姐姐", "妈妈", "女", "妹", "姐"]):
                                        gender = "ស្រី"
                                    elif any(w in item_zh for w in ["少爷", "先生", "哥哥", "弟弟", "爸爸", "男", "哥", "叔"]):
                                        gender = "ប្រុស"
                                    else:
                                        gender = "ស្រី" if ("ស្រី" in g_raw or "female" in g_raw.lower()) else "ប្រុស"
                            else:
                                item_zh = str(item.get("text", ""))
                                if any(w in item_zh for w in ["小姐", "夫人", "太太", "闺蜜", "伴娘", "妹妹", "姐姐", "妈妈", "女", "妹", "姐"]):
                                    gender = "ស្រី"
                                else:
                                    gender = "ស្រី" if ("ស្រី" in g_raw or "female" in g_raw.lower()) else "ប្រុស"

                            batch_result[iid] = {
                                "khmer": khmer_text,
                                "gender": gender,
                                "speaker": spk
                            }

                return batch_result

            except Exception as e:
                last_err = e
                err_str = str(e).lower()

                if "401" in err_str or "unauthenticated" in err_str:
                    self.mark_bad_key(key_prev, "401 Unauthenticated")
                    if not self.clients:
                        raise RuntimeError(f"កំហុស API Key មិនត្រឹមត្រូវ (401 Unauthenticated): {e}")
                    continue

                if "403" in err_str or "permission_denied" in err_str or "denied access" in err_str:
                    self.mark_bad_key(key_prev, "403 Permission Denied")
                    if not self.clients:
                        raise RuntimeError(f"កំហុស API Key ត្រូវបានបិទ (403 Permission Denied): {e}")
                    continue

                is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                if is_rate_limit:
                    self.mark_rate_limited(key_prev, cooldown_seconds=30.0)

                is_server_busy = "503" in err_str or "unavailable" in err_str or "overloaded" in err_str

                if is_server_busy:
                    time.sleep(random.uniform(0.2, 0.5))
                elif len(self.clients) > 1 and attempt < max_retries:
                    time.sleep(0.3)
                else:
                    sleep_time = min(2.5, 0.8 * attempt) if is_rate_limit else 0.5
                    time.sleep(sleep_time + random.uniform(0.1, 0.4))

        raise RuntimeError(f"បរាជ័យក្នុងការហៅ Gemini API: {last_err}")

    def translate_single_line_direct(self, text: str, gender_hint: str = "ប្រុស", max_retries: int = 5) -> str:
        """
        Ultra-resilient direct line translator for stubborn or missing rows.
        Retries across multiple models until pure, eloquent Khmer is generated.
        Never skips a line!
        """
        text = (text or "").strip()
        if not text:
            return ""

        prompt = (
            "You are Cambodia's #1 premier cinematic scriptwriter, master dubbing director, and poetic translator specializing in Chinese to Khmer.\n"
            "Task: Translate this dialogue or narration line from Chinese into precise, natural, and authentic spoken Khmer (បកប្រែឱ្យត្រូវតាមអត្ថន័យតួអក្សរច្បាស់លាស់បំផុត គ្មានវង់ក្រចក និងគ្មានការពន្យល់ឡើយ)!\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the pure Khmer translated text with ZERO explanations, commentary, parentheses, quotes, or JSON.\n"
            "2. Modern Polite Dialogue & Pronouns: STRICTLY FORBIDDEN to use 'អញ' or 'អ្ហែង'! Avoid 'ឯង' and avoid overusing 'បង - អូន'. Use polite, modern cinematic pronouns: 'ខ្ញុំ - អ្នក', 'លោក', 'អ្នកនាង', 'ម៉ាក់', 'ប៉ា', or address characters by name directly.\n"
            "3. Numbers & Currency: Translate amounts into natural Khmer phrasing (e.g. 31.36万 -> '៣១ ម៉ឺន ៣ ពាន់ ៦ រយ' / 'ជាង ៣០ ម៉ឺន', 3000 -> '៣ ពាន់').\n"
            "4. Zero foreign characters: 100% pure Khmer text.\n"
            "5. Keep character names consistent and natural.\n"
            f"Line to translate: {text}"
        )

        norm_model = normalize_gemini_model(self.model_name)
        models_to_try = [
            norm_model,
            "gemini-3.8-flash",
            "gemini-flash-lite-latest",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
        ]
        seen = set()
        models = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for attempt in range(1, max_retries + 1):
            client, key_prev = self._get_next_client()
            active_model = models[(attempt - 1) % len(models)]
            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=2048,
                    )
                )
                if response and response.text:
                    res_text = response.text.strip()
                    res_text = re.sub(r"^```(?:json)?\s*", "", res_text)
                    res_text = re.sub(r"\s*```$", "", res_text).strip()
                    res_text = res_text.strip('"\n\r ')
                    clean_res = sanitize_khmer_text(res_text)
                    if clean_res:
                        return clean_res
            except Exception as e:
                err_str = str(e).lower()
                if "401" in err_str or "unauthenticated" in err_str:
                    self.mark_bad_key(key_prev, "401 Unauthenticated")
                elif "403" in err_str or "permission_denied" in err_str or "denied access" in err_str:
                    self.mark_bad_key(key_prev, "403 Permission Denied")
                elif "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    self.mark_rate_limited(key_prev, cooldown_seconds=30.0)
                time.sleep(0.2 * attempt)

        return ""

    def translate_batch(
        self,
        items: List[Dict[str, Any]],
        gender_hints: Optional[Dict[int, str]] = None,
        preceding_context: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[int, Dict[str, str]]:
        """
        Translates a batch of subtitle items with AI diarization and direct single-line fallback.
        Ensures 100% of rows are translated with high accuracy.
        """
        if not items:
            return {}

        results = self._call_gemini_raw(
            items,
            gender_hints=gender_hints,
            preceding_context=preceding_context,
            **kwargs
        )

        # Fallback for any missing items in batch
        for item in items:
            iid = item["id"]
            if iid not in results or not results[iid].get("khmer"):
                g = gender_hints.get(iid, "ប្រុស") if gender_hints else "ប្រុស"
                fb_km = self.translate_single_line_direct(item.get("text", ""), gender_hint=g)
                results[iid] = {
                    "khmer": fb_km,
                    "gender": g,
                    "speaker": ""
                }

        return results

