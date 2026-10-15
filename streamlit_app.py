import os
import sys
import json
import time
import asyncio
import re
import tempfile
from typing import List, Dict, Any, Optional
from datetime import timedelta
import pandas as pd
import streamlit as st
import edge_tts
from pydub import AudioSegment
import srt

# Base directory & path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRT_DIR = os.path.join(BASE_DIR, "SRT")
if os.path.exists(SRT_DIR) and SRT_DIR not in sys.path:
    sys.path.insert(0, SRT_DIR)

AUDIO_OUTPUT_DIR = os.path.join(BASE_DIR, "audio_output")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# Import engine helpers
try:
    from voice_helpers import (
        apply_echo_effect,
        clean_tts_text,
        expand_khmer_lekh_tho,
        get_file_duration_ms,
        parse_voice_and_echo_from_text,
        trim_leading_trailing_silence,
    )
except ImportError:
    from SRT.voice_engine import (
        apply_echo_effect,
        clean_tts_text,
        expand_khmer_lekh_tho,
        get_file_duration_ms,
        parse_voice_and_echo_from_text,
        trim_leading_trailing_silence,
    )

try:
    from translator_engine import (
        GeminiSubtitleTranslator,
        format_timestamp,
        is_promo_ad_line,
        normalize_gemini_model,
        parse_srt_content,
        replace_numbers_with_khmer_words,
        sanitize_khmer_text,
        test_gemini_api_key,
    )
except ImportError:
    from SRT.translator_engine import (
        GeminiSubtitleTranslator,
        format_timestamp,
        is_promo_ad_line,
        normalize_gemini_model,
        parse_srt_content,
        replace_numbers_with_khmer_words,
        sanitize_khmer_text,
        test_gemini_api_key,
    )

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PL Khmer Voice - SRT & TTS Studio",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "sub_items" not in st.session_state:
    st.session_state.sub_items = []
if "filename" not in st.session_state:
    st.session_state.filename = "subtitles.srt"
if "master_audio_bytes" not in st.session_state:
    st.session_state.master_audio_bytes = None
if "master_audio_path" not in st.session_state:
    st.session_state.master_audio_path = None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### ⚙️ ការកំណត់ (Settings)")

    st.markdown("#### 🤖 Google Gemini AI")
    api_key_input = st.text_input(
        "Gemini API Key:",
        type="password",
        help="អាចដាក់ Key មួយ ឬច្រើនដោយខណ្ឌដោយក្បៀស (,)",
        placeholder="AIzaSy..."
    )

    gemini_model = st.selectbox(
        "Gemini Model:",
        options=["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-1.5-flash"],
        index=0
    )

    trans_style = st.selectbox(
        "Translation Style:",
        options=[
            ("emotion_acting", "🎭 សម្រួលតាមអារម្មណ៍តួអង្គ (Emotion & Acting)"),
            ("natural_story", "📖 រៀបរាប់សាច់រឿងបែបធម្មជាតិ (Natural Storytelling)"),
            ("concise", "⚡ សង្ខេបខ្លីល្មមតាមពេលវេលា (Concise/Time-fitted)"),
            ("literal", "🎯 បកប្រែចំអត្ថន័យដើម (Direct/Literal)")
        ],
        format_func=lambda x: x[1]
    )[0]

    st.divider()

    st.markdown("#### 🎙️ ការកំណត់សំឡេង (TTS Settings)")
    global_rate = st.select_slider(
        "ល្បឿននិយាយសរុប (Global Rate):",
        options=["-20%", "-10%", "+0%", "+10%", "+20%", "+30%", "+40%", "+50%"],
        value="+20%"
    )

    global_pitch = st.select_slider(
        "កម្រិតស្រួច/ធ្ងន់ (Pitch):",
        options=["-10Hz", "-5Hz", "+0Hz", "+5Hz", "+10Hz"],
        value="+0Hz"
    )

    mp3_bitrate = st.selectbox(
        "គុណភាព MP3 Bitrate:",
        options=["320k", "256k", "192k", "128k"],
        index=0
    )

    st.markdown("---")
    st.caption("🌟 PL Khmer Voice • Web Edition on Streamlit Cloud")


# --- TTS HELPER FUNCTION ---
async def generate_single_audio_async(sub_id: int, text: str, target_ms: int, voice: str, gender: str, echo: str, speed: str, rate: str, pitch: str, bitrate: str):
    clean_text = clean_tts_text(text)
    clean_text = expand_khmer_lekh_tho(clean_text)
    clean_text = replace_numbers_with_khmer_words(clean_text)
    clean_text = sanitize_khmer_text(clean_text)

    # Resolve Voice
    if voice == "Piseth" or (voice == "Auto" and gender != "ស្រី"):
        resolved_voice = "km-KH-PisethNeural"
    else:
        resolved_voice = "km-KH-SreymomNeural"

    # Rate calculation
    if speed == "Auto" or not speed:
        est_sec = max(0.5, len(clean_text) * 0.088)
        safe_target_sec = max(0.8, (target_ms / 1000.0) * 1.55)
        try:
            base_rate_pct = int(re.sub(r'[^0-9\-+]', '', rate)) if rate else 20
        except ValueError:
            base_rate_pct = 20
        if est_sec > safe_target_sec:
            calc_pct = int(((est_sec / safe_target_sec) - 1.0) * 100)
            rate_to_use = f"{max(base_rate_pct, min(90, calc_pct)):+d}%"
        else:
            rate_to_use = f"{base_rate_pct:+d}%"
    else:
        rate_to_use = speed if speed != "Default" else rate

    filename = f"line_{sub_id:03d}_{int(time.time() * 1000)}.mp3"
    filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)

    has_content = bool(re.search(r'[\u1780-\u17FFa-zA-Z0-9]', clean_text))
    export_params = ["-q:a", "0", "-ar", "48000"]

    if not has_content:
        silence_dur = max(200, min(target_ms, 5000)) if target_ms > 0 else 500
        silent = AudioSegment.silent(duration=silence_dur)
        silent.export(filepath, format="mp3", bitrate=bitrate, parameters=export_params)
        return {"filepath": filepath, "duration_ms": silence_dur, "rate_used": rate_to_use}

    comm = edge_tts.Communicate(text=clean_text, voice=resolved_voice, rate=rate_to_use, pitch=pitch)
    await comm.save(filepath)

    if os.path.exists(filepath):
        try:
            raw_audio = AudioSegment.from_file(filepath)
            trimmed = trim_leading_trailing_silence(raw_audio)
            if echo == "Yes":
                trimmed = apply_echo_effect(trimmed)
            trimmed.export(filepath, format="mp3", bitrate=bitrate, parameters=export_params)
        except Exception:
            pass

    dur_ms = get_file_duration_ms(filepath)
    return {"filepath": filepath, "duration_ms": dur_ms, "rate_used": rate_to_use}


# --- MASTER AUDIO COMPILER ---
def compile_master_audio(items: List[Dict[str, Any]], out_filename: str, bitrate: str = "320k") -> Optional[str]:
    valid_clips = []
    for it in items:
        fpath = it.get("audio_filepath")
        if fpath and os.path.exists(fpath):
            start_ms = it.get("start_ms", 0)
            end_ms = it.get("end_ms", start_ms + 1000)
            dur_ms = it.get("duration_ms", end_ms - start_ms)
            valid_clips.append((fpath, start_ms, end_ms, dur_ms))

    if not valid_clips:
        return None

    clean_base = re.sub(r'\.[^.]+$', '', out_filename).strip()
    out_mp3_name = f"{clean_base}_master.mp3" if clean_base else f"master_{int(time.time())}.mp3"
    out_path = os.path.join(AUDIO_OUTPUT_DIR, out_mp3_name)

    max_dur = max([max(start_ms + dur_ms, end_ms) for _, start_ms, end_ms, dur_ms in valid_clips] or [1000])
    master = AudioSegment.silent(duration=max_dur + 500)

    for path, start_ms, _, _ in valid_clips:
        try:
            clip = AudioSegment.from_file(path)
            master = master.overlay(clip, position=start_ms)
        except Exception as e:
            print(f"Overlay error for {path}: {e}")

    master.export(out_path, format="mp3", bitrate=bitrate, parameters=["-q:a", "0", "-ar", "48000"])
    return out_path


# --- MAIN HEADER ---
st.markdown('<div class="main-title">🌟 PL Khmer Voice - SRT & TTS Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">កម្មវិធីបកប្រែ Subtitle (ចិន/អង់គ្លេស ➡️ ខ្មែរ) និងបង្កើតសំឡេងអានបែបភាពយន្ត (Khmer Neural TTS) លើ Cloud</div>', unsafe_allow_html=True)

# --- STEP 1: FILE UPLOAD ---
col_up, col_info = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "📂 ជ្រើសរើសឯកសារ Subtitle (.srt ឬ .progress.json):",
        type=["srt", "json"],
        help="ទាញឯកសារ SRT ឬ Progress JSON មកដាក់ទីនេះ"
    )

if uploaded_file is not None and (not st.session_state.sub_items or st.session_state.filename != uploaded_file.name):
    st.session_state.filename = uploaded_file.name
    content_bytes = uploaded_file.read()

    try:
        if uploaded_file.name.endswith(".json"):
            json_data = json.loads(content_bytes.decode("utf-8", errors="ignore"))
            st.session_state.sub_items = json_data.get("items", [])
            st.success(f"✅ បានផ្ទុក {len(st.session_state.sub_items)} ជួរពី Session Progress File!")
        else:
            text_content = content_bytes.decode("utf-8", errors="ignore")
            parsed_subs = parse_srt_content(text_content)
            items = []
            for sub in parsed_subs:
                clean_orig = is_promo_ad_line(sub["text"], "")
                orig_display = sub["text"] if not clean_orig else f"⚠️ [AD] {sub['text']}"
                gender_hint, echo_hint, stripped_txt = parse_voice_and_echo_from_text(sub["text"])

                items.append({
                    "id": sub["id"],
                    "start_ms": sub["start_ms"],
                    "end_ms": sub["end_ms"],
                    "start_time": sub["start_time"],
                    "end_time": sub["end_time"],
                    "target_ms": sub["duration_ms"],
                    "orig_text": orig_display,
                    "khmer_text": stripped_txt if any(ord(c) >= 0x1780 and ord(c) <= 0x17FF for c in stripped_txt) else "",
                    "gender": gender_hint if gender_hint != "Auto" else "ប្រុស",
                    "voice": "Auto",
                    "echo": "Yes" if echo_hint else "No",
                    "speed": "Auto",
                    "status": "រង់ចាំបកប្រែ",
                    "audio_filepath": None,
                    "duration_ms": 0
                })
            st.session_state.sub_items = items
            st.success(f"✅ បានផ្ទុក {len(items)} ជួរដោយជោគជ័យ!")
    except Exception as e:
        st.error(f"❌ កំហុសក្នុងការអានឯកសារ: {e}")

with col_info:
    total_lines = len(st.session_state.sub_items)
    if total_lines > 0:
        total_dur_ms = st.session_state.sub_items[-1]["end_ms"]
        total_min = total_dur_ms / 60000.0
        st.metric("ចំនួនបន្ទាត់សរុប (Lines)", f"{total_lines} ជួរ")
        st.metric("រយៈពេលសរុប (Duration)", f"{total_min:.1f} នាទី")
    else:
        st.info("👈 សូម Upload ឯកសារ SRT ដើម្បីចាប់ផ្ដើម។")

if st.session_state.sub_items:
    st.divider()

    # --- STEP 2: ACTION BUTTONS (TRANSLATE & DUB) ---
    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 1])

    with btn_col1:
        run_translate = st.button("🚀 ១. បកប្រែ Subtitle (AI Translate)", use_container_width=True, type="primary")

    with btn_col2:
        run_tts = st.button("🎙️ ២. បង្កើតសំឡេង & ផ្គុំ Master MP3", use_container_width=True)

    with btn_col3:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.sub_items = []
            st.session_state.master_audio_bytes = None
            st.session_state.master_audio_path = None
            st.rerun()

    # --- EXECUTE TRANSLATION ---
    if run_translate:
        keys_list = [k.strip() for k in api_key_input.replace("\n", ",").split(",") if k.strip()]
        if not keys_list:
            st.error("⚠️ សូមបញ្ចូល Gemini API Key នៅក្នុង Sidebar ខាងឆ្វេងជាមុនសិន!")
        else:
            prog_bar = st.progress(0, text="🤖 កំពុងវិភាគតួអង្គ និងបកប្រែ Subtitle...")
            translator = GeminiSubtitleTranslator(
                api_keys=keys_list,
                model_name=gemini_model,
                style=trans_style
            )

            # Analyze Character Genders
            prog_bar.progress(10, text="👥 [Stage 1] កំពុងវិភាគភេទតួអង្គ (Character Casting)...")
            items_for_cast = [{"id": it["id"], "text": it["orig_text"]} for it in st.session_state.sub_items if it["orig_text"]]
            gender_map = translator.analyze_full_script_genders(items_for_cast, concurrency=4, chunk_size=80)

            for it in st.session_state.sub_items:
                sid = it["id"]
                if sid in gender_map:
                    it["gender"] = gender_map[sid]

            # Batch Translation
            batch_size = 25
            total_items = len(st.session_state.sub_items)
            for i in range(0, total_items, batch_size):
                chunk = st.session_state.sub_items[i : i + batch_size]
                b_data = [{"id": it["id"], "text": it["orig_text"], "duration_sec": it["target_ms"] / 1000.0, "max_speech_sec": (it["target_ms"] / 1000.0) * 1.6} for it in chunk]
                prec = [{"id": it["id"], "text": it["orig_text"]} for it in st.session_state.sub_items[max(0, i-5):i]]

                pct = int(15 + ((i + 1) / total_items) * 80)
                prog_bar.progress(pct, text=f"⚡ កំពុងបកប្រែជួរ {i+1} ដល់ {min(i+batch_size, total_items)}...")

                res_map = translator.translate_batch(b_data, gender_hints=gender_map, preceding_context=prec)
                for it in chunk:
                    sid = it["id"]
                    if sid in res_map:
                        it["khmer_text"] = res_map[sid]
                        it["status"] = "បកប្រែរួច"

            prog_bar.progress(100, text="🎉 បកប្រែរួចរាល់ ១០០%!")
            st.success("🎉 ការបកប្រែ Subtitle ត្រូវបានបញ្ចប់ជោគជ័យ!")
            time.sleep(1)
            st.rerun()

    # --- EXECUTE TTS & MASTER AUDIO ---
    if run_tts:
        prog_bar = st.progress(0, text="🎙️ កំពុងបង្កើតសំឡេងនិយាយខ្មែរ (Edge-TTS)...")
        total_items = len(st.session_state.sub_items)

        async def run_all_tts():
            sem = asyncio.Semaphore(10)
            completed = 0

            async def process_item(it):
                nonlocal completed
                async with sem:
                    res = await generate_single_audio_async(
                        sub_id=it["id"],
                        text=it.get("khmer_text") or it.get("orig_text", ""),
                        target_ms=it.get("target_ms", 2000),
                        voice=it.get("voice", "Auto"),
                        gender=it.get("gender", "ប្រុស"),
                        echo=it.get("echo", "No"),
                        speed=it.get("speed", "Auto"),
                        rate=global_rate,
                        pitch=global_pitch,
                        bitrate=mp3_bitrate
                    )
                    it["audio_filepath"] = res["filepath"]
                    it["duration_ms"] = res["duration_ms"]
                    it["speed"] = res["rate_used"]
                    it["status"] = "សំឡេងរួចរាល់"
                    completed += 1
                    pct = int((completed / total_items) * 85)
                    prog_bar.progress(pct, text=f"🎙️ បានបង្កើតសំឡេង {completed}/{total_items} ជួរ ({pct}%)...")

            tasks = [process_item(it) for it in st.session_state.sub_items]
            await asyncio.gather(*tasks)

        asyncio.run(run_all_tts())

        # Compile Master MP3
        prog_bar.progress(90, text="🎵 កំពុងផ្គុំ Master Audio MP3...")
        out_master = compile_master_audio(st.session_state.sub_items, st.session_state.filename, bitrate=mp3_bitrate)

        if out_master and os.path.exists(out_master):
            st.session_state.master_audio_path = out_master
            with open(out_master, "rb") as f:
                st.session_state.master_audio_bytes = f.read()

        prog_bar.progress(100, text="🎉 រួចរាល់ ១០០%!")
        st.success("🎉 បង្កើតសំឡេង និងផ្គុំ Master Audio MP3 រួចរាល់!")
        time.sleep(1)
        st.rerun()

    # --- STEP 3: MASTER AUDIO PLAYER & DOWNLOADS ---
    if st.session_state.master_audio_bytes:
        st.markdown("### 🎧 Master Voiceover Audio")
        st.audio(st.session_state.master_audio_bytes, format="audio/mp3")

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            clean_base = re.sub(r'\.[^.]+$', '', st.session_state.filename).strip()
            st.download_button(
                "⬇️ Download Master MP3",
                data=st.session_state.master_audio_bytes,
                file_name=f"{clean_base}_master.mp3",
                mime="audio/mpeg",
                use_container_width=True,
                type="primary"
            )

        with col_dl2:
            # Generate SRT Content
            sub_objs = []
            for it in st.session_state.sub_items:
                txt = (it.get("khmer_text") or it.get("orig_text") or "").strip()
                tag_parts = [f"មនុស្ស{it.get('gender', 'ប្រុស')}"]
                if it.get("echo") == "Yes":
                    tag_parts.append("សំឡេង Echo")
                full_line = f"[{' | '.join(tag_parts)}] {txt}"
                sub_objs.append(srt.Subtitle(
                    index=it["id"],
                    start=timedelta(milliseconds=it["start_ms"]),
                    end=timedelta(milliseconds=it["end_ms"]),
                    content=full_line
                ))
            srt_str = srt.compose(sub_objs)
            st.download_button(
                "⬇️ Download Subtitle (.srt)",
                data=srt_str.encode("utf-8"),
                file_name=f"{clean_base}_khmer.srt",
                mime="text/plain",
                use_container_width=True
            )

        with col_dl3:
            # Export Progress JSON
            progress_payload = json.dumps({"filename": st.session_state.filename, "items": st.session_state.sub_items}, ensure_ascii=False, indent=2)
            st.download_button(
                "💾 Save Progress (.json)",
                data=progress_payload.encode("utf-8"),
                file_name=f"{clean_base}.progress.json",
                mime="application/json",
                use_container_width=True
            )

    st.divider()

    # --- STEP 4: INTERACTIVE SUBTITLE TABLE ---
    st.markdown("### 📝 តារាងកែសម្រួល Subtitle (Interactive Table)")
    st.caption("💡 អ្នកអាចចុចកែសម្រួលអត្ថបទភាសាខ្មែរ (Khmer Text) ឬប្ដូរភេទ (Gender) បានដោយផ្ទាល់លើតារាងខាងក្រោម៖")

    df_data = []
    for it in st.session_state.sub_items:
        time_str = f"{it['start_time']} ➔ {it['end_time']}"
        dur_s = f"{(it['target_ms'] / 1000.0):.1f}s"
        df_data.append({
            "ID": it["id"],
            "ពេលវេលា (Time)": time_str,
            "ប្រវែង": dur_s,
            "អត្ថបទដើម (Original)": it["orig_text"],
            "អត្ថបទខ្មែរ (Khmer Text)": it.get("khmer_text", ""),
            "ភេទ (Gender)": it.get("gender", "ប្រុស"),
            "សំឡេង (Voice)": it.get("voice", "Auto"),
            "Echo": it.get("echo", "No"),
            "ស្ថានភាព": it.get("status", "")
        })

    df = pd.DataFrame(df_data)

    edited_df = st.data_editor(
        df,
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "ពេលវេលា (Time)": st.column_config.TextColumn("ពេលវេលា (Time)", disabled=True, width="medium"),
            "ប្រវែង": st.column_config.TextColumn("ប្រវែង", disabled=True, width="small"),
            "អត្ថបទដើម (Original)": st.column_config.TextColumn("អត្ថបទដើម (Original)", disabled=True, width="large"),
            "អត្ថបទខ្មែរ (Khmer Text)": st.column_config.TextColumn("អត្ថបទខ្មែរ (Khmer Text)", width="large"),
            "ភេទ (Gender)": st.column_config.SelectboxColumn("ភេទ (Gender)", options=["ប្រុស", "ស្រី"], width="small"),
            "សំឡេង (Voice)": st.column_config.SelectboxColumn("សំឡេង (Voice)", options=["Auto", "Piseth", "Sreymom"], width="small"),
            "Echo": st.column_config.SelectboxColumn("Echo", options=["No", "Yes"], width="small"),
            "ស្ថានភាព": st.column_config.TextColumn("ស្ថានភាព", disabled=True, width="medium")
        },
        use_container_width=True,
        num_rows="fixed",
        hide_index=True
    )

    # Sync table edits back to session state items
    for idx, row in edited_df.iterrows():
        if idx < len(st.session_state.sub_items):
            st.session_state.sub_items[idx]["khmer_text"] = row["អត្ថបទខ្មែរ (Khmer Text)"]
            st.session_state.sub_items[idx]["gender"] = row["ភេទ (Gender)"]
            st.session_state.sub_items[idx]["voice"] = row["សំឡេង (Voice)"]
            st.session_state.sub_items[idx]["echo"] = row["Echo"]
