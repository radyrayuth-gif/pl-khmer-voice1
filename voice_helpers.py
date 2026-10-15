import os
import sys
import re
import asyncio
from pydub import AudioSegment

try:
    import mutagen.mp3
except ImportError:
    mutagen = None

def get_file_duration_ms(file_path: str) -> int:
    """
    Ultra-lightweight duration reader in pure Python.
    Uses mutagen (reads MP3 header in ~0.5ms with 0% CPU, no external processes).
    Falls back to pydub AudioSegment only if mutagen fails.
    """
    if mutagen:
        try:
            mp3 = mutagen.mp3.MP3(file_path)
            if mp3 and mp3.info and mp3.info.length > 0:
                return int(mp3.info.length * 1000)
        except Exception:
            pass
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio)
    except Exception:
        return 0

def ms_to_timestamp(ms: int) -> str:
    """Converts milliseconds to HH:MM:SS,mmm format."""
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_voice_and_echo_from_text(text: str, default_voice: str = "Global"):
    match = re.match(r'^\[(.*?)\]', text)
    voice = default_voice
    echo = False
    
    if match:
        tag_content = match.group(1).strip()
        if "ស្រី" in tag_content or "Sreymom" in tag_content or "Female" in tag_content or "female" in tag_content:
            voice = "Sreymom"
        elif "ប្រុស" in tag_content or "Piseth" in tag_content or "Male" in tag_content or "male" in tag_content or "士兵" in tag_content:
            voice = "Piseth"
            
        if "Echo" in tag_content or "echo" in tag_content or "ECHO" in tag_content or "សំឡេង" in tag_content:
            echo = True
            
    return voice, echo

COMMON_KHMER_REDUPLICATED_WORDS = sorted([
    'បន្តិចបន្តួច', 'តិចតួច', 'រាល់ថ្ងៃ', 'រាល់ខែ', 'រាល់ឆ្នាំ', 'រាល់ពេល', 'ញឹកញាប់', 'ជួនកាល', 'យូរយារ',
    'ស្រស់ស្អាត', 'ត្រជាក់ត្រជុំ', 'កក់ក្ដៅ', 'ស្ងាត់ជ្រងំ', 'កំប៉ិកកំប៉ុក', 'រញ៉េរញ៉ៃ', 'រប៉ាត់រប៉ាយ',
    'រដិបរដុប', 'រលោងស្រិល', 'ក្រអូបឈ្ងុយ', 'ផ្អែមល្ហែម', 'សប្បាយរីករាយ', 'កម្សត់កម្រ',
    'ឆ្ងាញ់', 'ផ្អែម', 'ជូរ', 'ចត់', 'ល្វីង', 'ប្រៃ', 'សាប', 'ហឹរ', 'ឈ្ងុយ', 'ស្អុយ',
    'លឿន', 'យឺត', 'រហ័ស', 'ឆាប់', 'យូរ', 'ញឹក', 'ញ៉យ', 'មុន', 'ក្រោយ', 'ភ្លាម',
    'ធំ', 'តូច', 'វែង', 'ខ្លី', 'ខ្ពស់', 'ទាប', 'ធាត់', 'ស្គម', 'ស្គាំង', 'មូល', 'កោង', 'វៀច', 'ត្រង់', 'ស្មើ',
    'ច្រើន', 'តិច', 'បន្តិច', 'ខ្លាំង', 'ខ្សោយ', 'ណាស់', 'ពេក', 'ក្រែល', 'ស្ទើរ',
    'ជិត', 'ឆ្ងាយ', 'ក្បែរ', 'លើ', 'ក្រោម', 'ក្នុង', 'ក្រៅ', 'ដើម', 'ចុង', 'មុខ',
    'ល្អ', 'អាក្រក់', 'ស្អាត', 'ថ្មី', 'ចាស់', 'ស្រាល', 'ធ្ងន់', 'ងាយ', 'ស្រួល', 'ពិបាក',
    'ស្ងាត់', 'ស្ងប់', 'ភ្លឺ', 'ងងឹត', 'ក្ដៅ', 'ត្រជាក់', 'រឹង', 'ទន់', 'រលោង', 'គគ្រើម',
    'ពិត', 'ជាក់', 'ច្បាស់', 'ផ្ទាល់', 'ប្រាកដ', 'ប្លែក',
    'សប្បាយ', 'រីករាយ', 'កើតទុក្ខ', 'កម្សត់', 'សោកសៅ', 'ឈឺ', 'ចុក', 'ផ្សា', 'រមាស់',
    'ហត់', 'នឿយ', 'ឃ្លាន', 'ស្រេក', 'ឆ្អែត', 'ស្រួលខ្លួន', 'ខ្លាច', 'ភ័យ', 'ស្លន់', 'តក់ស្លុត', 'ញ័រ',
    'ក្មេង', 'ចាស់', 'ស្រី', 'ប្រុស', 'មនុស្ស', 'បង', 'ប្អូន', 'កូន', 'ម៉ែ', 'ឪ',
    'សើច', 'យំ', 'ស្រែក', 'ដើរ', 'រត់', 'ហោះ', 'ហើរ', 'លោត', 'មើល', 'គិត', 'ស្តាប់',
    'និយាយ', 'ធ្វើ', 'ញ៉ាំ', 'ហូប', 'ផឹក', 'ដេក', 'ដួល', 'អង្គុយ', 'ឈរ',
    'ហា', 'ហាស', 'ហ៊ឺ', 'ហ៊ា', 'ហិ', 'ហុ', 'អ្ហា', 'អូ', 'វ៉ាវ', 'គ្រុក', 'ដុក', 'ភឹប',
    'ផ្សេង', 'គ្រប់', 'ខ្លះ', 'ឯង', 'មួយ', 'ពីរ', 'បី', 'បួន', 'ប្រាំ', 'ម្តង', 'អី', 'ណា', 'នោះ', 'នេះ', 'ហ្នឹង', 'ចុះ', 'ឡើង'
], key=len, reverse=True)

def expand_khmer_lekh_tho(text: str) -> str:
    """Intelligently expands the Khmer Lekh Tho repetition mark 'ៗ' (U+17D7)."""
    if not text or 'ៗ' not in text:
        return text

    khmer_char = r'[\u1780-\u17D3\u17D8-\u17DD\u17E0-\u17E9A-Za-z0-9]'

    def expand_token(m):
        prefix_boundary = m.group(1)
        token = m.group(2)
        for w in COMMON_KHMER_REDUPLICATED_WORDS:
            if token.endswith(w):
                stem = token[:-len(w)]
                return f"{prefix_boundary}{stem}{w} {w}"
        if len(token) > 6:
            clusters = re.findall(
                r'[\u1780-\u17A2](?:[\u17D2][\u1780-\u17A2]|[\u17B6-\u17C5]|[\u17C6-\u17D3])*',
                token,
            )
            if clusters:
                last_word = ''.join(clusters[-2:]) if len(clusters) >= 2 else clusters[-1]
                stem = token[:-len(last_word)]
                return f"{prefix_boundary}{stem}{last_word} {last_word}"
        return f"{prefix_boundary}{token} {token}"

    pattern = rf'(\s|^|[^\u1780-\u17D3\u17D8-\u17DD\u17E0-\u17E9A-Za-z0-9])({khmer_char}+)\s*ៗ'
    prev = None
    curr = text
    while prev != curr and 'ៗ' in curr:
        prev = curr
        curr = re.sub(pattern, expand_token, curr)
        if 'ៗ' in curr and prev == curr:
            curr = re.sub(rf'({khmer_char}+)\s*ៗ', r'\1 \1', curr)
            if prev == curr:
                curr = curr.replace('ៗ', '')
                break
    return re.sub(r'[ \t]+', ' ', curr).strip()

def clean_tts_text(text: str) -> str:
    cleaned = re.sub(r'^\[[^\]]+\]\s*', '', text)
    cleaned = expand_khmer_lekh_tho(cleaned)
    return cleaned

def trim_leading_trailing_silence(sound, silence_threshold=-40.0, chunk_size=10, margin_ms=20):
    try:
        from pydub.silence import detect_leading_silence
        trim_start = detect_leading_silence(sound, silence_threshold=silence_threshold, chunk_size=chunk_size)
        trim_end = detect_leading_silence(sound.reverse(), silence_threshold=silence_threshold, chunk_size=chunk_size)
        trim_start = max(0, trim_start - margin_ms)
        trim_end = max(0, trim_end - margin_ms)
        duration = len(sound)
        if trim_start + trim_end >= duration:
            return sound
        return sound[trim_start:duration - trim_end]
    except Exception:
        return sound

def apply_echo_effect(audio, delay_ms=180, decay=0.5, num_echos=3, echo_reduction_percent=70):
    extra_attenuation_db = 10.46
    tail_length = delay_ms * num_echos
    extended_audio = audio + AudioSegment.silent(duration=tail_length, frame_rate=audio.frame_rate)
    combined = extended_audio
    for i in range(1, num_echos + 1):
        attenuation = i * 6 + extra_attenuation_db
        silence = AudioSegment.silent(duration=i * delay_ms, frame_rate=audio.frame_rate)
        echo_clip = silence + audio.apply_gain(-attenuation)
        combined = combined.overlay(echo_clip)
    return combined
