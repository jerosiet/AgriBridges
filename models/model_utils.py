import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import logging
import whisper
import os
import re


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
# Sử dụng 'cpu' làm mặc định nếu 'mps' hoặc 'cuda' gặp vấn đề
device = torch.device("mps" if torch.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
logger.info(f"Using device: {device}")

# --- Khởi tạo và tải các mô hình trực tiếp (thay thế ModelManager) ---

# Dictionary để lưu trữ các mô hình dịch thuật và tokenizer của chúng
translation_models = {}
translation_tokenizers = {}
from transformers import VitsTokenizer, VitsModel


# MMS-TTS cho tiếng Anh và tiếng Việt
tts_models = {
    "en": "facebook/mms-tts-eng",
    "vi": "facebook/mms-tts-vie"
}

tts_processors = {}
tts_instances = {}

# Load mô hình TTS
for lang, model_name in tts_models.items():
    try:
        processor = VitsTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name).to(device)
        tts_processors[lang] = processor
        tts_instances[lang] = model
        logger.info(f"✓ Loaded MMS-TTS model for {lang}")
    except Exception as e:
        logger.exception(f"✗ Failed to load MMS-TTS model for {lang}: {e}")
# Hàm trợ giúp để tải mô hình dịch thuật
import sounddevice as sd
import numpy as np
import soundfile as sf  # đã có
import os
import subprocess
import librosa
from pydub import AudioSegment
import os

def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")
    return wav_path
def safe_load_audio(audio_path, sr=16000):
    # Nếu là file .webm → convert sang .wav tạm
    tmp_wav = None
    if audio_path.lower().endswith(".webm"):
        tmp_wav = audio_path.rsplit(".", 1)[0] + "_tmp.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", str(sr), "-ac", "1", tmp_wav
        ], check=True)
        audio_path = tmp_wav

    # Đọc file audio
    speech, sample_rate = sf.read(audio_path, dtype="float32")

    # Nếu là file tạm → xoá
    if tmp_wav and os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    return speech, sample_rate
def synthesize_speech(text, lang, output_path="output.wav", play_audio=True):
    processor = tts_processors.get(lang)
    model = tts_instances.get(lang)
    if not processor or not model:
        return f"TTS Model for '{lang}' not loaded."
    try:
        inputs = processor(text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            speech_values = model(**inputs).waveform  # [1, L]
            waveform = speech_values[0].squeeze().cpu().numpy()
            sf.write(output_path, waveform, 16000)

        # Tự động phát âm thanh
        if play_audio:
            logger.info(f"Playing audio from {output_path}...")
            sd.play(waveform, samplerate=16000)
            sd.wait()  # Chờ phát xong mới tiếp tục

        return output_path
    except Exception as e:
        logger.exception(f"TTS error for {lang}: {e}")
        return f"Lỗi sinh giọng nói ({lang}): {e}"
def _load_seq2seq_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        logger.info(f"✓ Loaded {model_name}")
        return tokenizer, model
    except Exception as e:
        logger.exception(f"✗ Failed to load {model_name}: {e}")
        return None, None

# Tải các mô hình dịch thuật
translation_model_configs = [
    "VietAI/envit5-translation",
    "./academic_to_farmer",
    "./farmer_to_academic",
]

for model_name in translation_model_configs:
    tokenizer, model = _load_seq2seq_model_and_tokenizer(model_name)
    if tokenizer and model:
        translation_tokenizers[model_name] = tokenizer
        translation_models[model_name] = model

# Tải PhoWhisper model (cho tiếng Việt)
phowhisper_model_instance = None
phowhisper_processor_instance = None
phowhisper_model_name = "vinai/PhoWhisper-base"

try:
    phowhisper_processor_instance = AutoProcessor.from_pretrained(phowhisper_model_name)
    phowhisper_model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(
        phowhisper_model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info(f"✓ PhoWhisper model ({phowhisper_model_name}) loaded successfully.")
except Exception as e:
    logger.exception(f"✗ Failed to load PhoWhisper model ({phowhisper_model_name}): {e}")
    phowhisper_model_instance = None
    phowhisper_processor_instance = None

# --- Kết thúc phần tải mô hình trực tiếp ---

def translate_text(text, model_name, src_lang=None, tgt_lang=None):
    """Generic translation function"""
    if not text or not text.strip():
        return ""

    tokenizer = translation_tokenizers.get(model_name)
    model = translation_models.get(model_name)

    if not tokenizer or not model:
        return f"Mô hình {model_name} chưa được tải."

    try:
        input_text = text.strip()
        if src_lang and tgt_lang and "envit5" in model_name.lower():
            input_text = f"{src_lang}>>{tgt_lang}: {text}"

        input_ids = tokenizer.encode(
            input_text, return_tensors="pt", max_length=512,
            truncation=True, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_length=256, num_beams=4,
                early_stopping=True, do_sample=False,
                repetition_penalty=1.2
            )

        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_text = re.sub(r"^(>>)?(vi|en|Vi|En)(>>)?\s*:? ?", "", decoded_text).strip()
        
        return cleaned_text

    except Exception as e:
        logger.error(f"Translation error with {model_name}: {e}")
        return f"Lỗi dịch thuật: {e}"

from pydub import AudioSegment
import torchaudio
from tempfile import NamedTemporaryFile
import os

def transcribe_vietnamese_audio(audio_path):
    """Transcribe Vietnamese audio using PhoWhisper"""
    global phowhisper_model_instance, phowhisper_processor_instance

    if not phowhisper_model_instance or not phowhisper_processor_instance:
        logger.warning("[DEBUG] PhoWhisper model or processor is None.")
        return "Mô hình PhoWhisper tiếng Việt chưa được tải. Vui lòng kiểm tra lại cài đặt."
    
    try:
        if not os.path.exists(audio_path):
            logger.error(f"[DEBUG] File không tồn tại: {audio_path}")
            return f"File âm thanh không tồn tại: {audio_path}"

        # Nếu là .m4a thì convert sang .wav tạm thời
        if audio_path.lower().endswith(".m4a"):
            logger.info(f"[DEBUG] Converting M4A to WAV: {audio_path}")
            tmp_wav = NamedTemporaryFile(suffix=".wav", delete=False)
            AudioSegment.from_file(audio_path, format="m4a").export(tmp_wav.name, format="wav")
            audio_path = tmp_wav.name

        # Load audio giữ nguyên sample rate gốc
        logger.info(f"[DEBUG] Loading audio from {audio_path}")
        speech, sample_rate = safe_load_audio(audio_path, sr=None)  # sr=None để lấy sample rate gốc
        logger.info(f"[DEBUG] Original Sample rate: {sample_rate}")

        # Nếu sample rate khác 16000 thì resample xuống 16000Hz
        if sample_rate != 16000:
            logger.info(f"[DEBUG] Resampling from {sample_rate}Hz to 16000Hz")
            if isinstance(speech, np.ndarray):
                speech = torch.from_numpy(speech).float()
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            speech = resampler(speech)
            sample_rate = 16000

        logger.info(f"[DEBUG] Audio length (samples): {len(speech)}, Sample rate: {sample_rate}")

        if len(speech) < 1600:
            logger.warning("[DEBUG] Audio quá ngắn.")
            return "File âm thanh quá ngắn để xử lý."

        logger.info("[DEBUG] Processing audio features...")
        input_features = phowhisper_processor_instance(
            speech, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            logger.info("[DEBUG] Generating transcription...")
            predicted_ids = phowhisper_model_instance.generate(input_features)

        transcription = phowhisper_processor_instance.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        logger.info(f"[DEBUG] Transcription result: {transcription.strip()}")
        return transcription.strip()

    except Exception as e:
        logger.exception(f"Vietnamese STT (PhoWhisper) error: {e}")
        return f"Lỗi chuyển đổi âm thanh tiếng Việt (PhoWhisper): {e}"
def transcribe_english_audio(audio_path):
    """Transcribe English audio using Whisper"""

    try:
        if not os.path.exists(audio_path):
            return f"File âm thanh không tồn tại: {audio_path}"

        # Đặt device thành "cpu" để tránh lỗi MPS nếu bạn gặp phải
        # Nếu bạn chắc chắn MPS/CUDA hoạt động tốt, có thể sử dụng biến 'device' toàn cục
        whisper_device = "cpu" if device.type == "mps" else device.type # Ưu tiên CPU cho MPS nếu có vấn đề, ngược lại giữ nguyên device
        model = whisper.load_model("base", device=whisper_device) # Load model directly, specifying device

        # Thực hiện chuyển đổi âm thanh
        result = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            verbose=False
        )

        if result and "text" in result and result["text"].strip():
            return result["text"].strip()
        else:
            return "Không thể nhận diện được nội dung âm thanh tiếng Anh."

    except Exception as e:
        logger.exception(f"English STT (Whisper) error: {e}")
        return f"Lỗi chuyển đổi âm thanh tiếng Anh (Whisper): {e}"


def transcribe_audio(audio_path, audio_lang):
    """Transcribe audio based on language"""
    try:
        if audio_lang == "vi":
            return transcribe_vietnamese_audio(audio_path)
        elif audio_lang == "en":
            return transcribe_english_audio(audio_path)
        else:
            return "Ngôn ngữ không được hỗ trợ."
    except Exception as e:
        logger.exception(f"Audio transcription error: {e}")
        return f"Lỗi trong quá trình chuyển đổi âm thanh: {e}"

# Translation functions (giữ nguyên)
def en_to_vi(text):
    """Translate English to Vietnamese"""
    return translate_text(text, "VietAI/envit5-translation", "en", "vi")

def vi_to_farmer(text):
    """Convert Vietnamese academic to farmer language"""
    return translate_text(text, "./academic_to_farmer")

def farmer_to_scientific(text):
    """Convert farmer language to Vietnamese academic"""
    return translate_text(text, "./farmer_to_academic")

def scientific_to_en(text):
    """Translate Vietnamese scientific to English"""
    return translate_text(text, "VietAI/envit5-translation", "vi", "en")
import re
import inflect
from vietnam_number import n2w  # dùng đúng theo tài liệu của họ

p = inflect.engine()

# Chuyển số thành chữ tiếng Việt bằng n2w
def number_to_vietnamese_words(text):
    def convert_number(match):
        num_str = match.group()
        try:
            return n2w(num_str)
        except Exception:
            return num_str  # fallback nếu có lỗi
    return re.sub(r'\b\d+\b', convert_number, text)

# Chuyển số thành chữ tiếng Anh bằng inflect
def number_to_english_words(text):
    def convert_number(match):
        try:
            return p.number_to_words(match.group())
        except Exception:
            return match.group()
    return re.sub(r'\b\d+\b', convert_number, text)

# Pipeline functions (giữ nguyên)
def en_to_vi_farmer_pipeline(text, tts_enabled=True):
    """English → Vietnamese → Farmer language + TTS (Vietnamese)"""
    if not text or not text.strip():
        return ""

    vi_text = en_to_vi(text)
    if vi_text.startswith(("Lỗi", "Mô hình", "Không thể")):
        return vi_text
    vi_text = number_to_vietnamese_words(vi_text)  # 🔁 chuyển số trong tiếng Việt
    farmer_text = vi_to_farmer(vi_text)
    if farmer_text.startswith(("Lỗi", "Mô hình", "Không thể")):
        return farmer_text

    if tts_enabled:
        synthesize_speech(farmer_text, lang="vi", output_path="farmer_output.wav")
        logger.info("✓ Đã sinh file farmer_output.wav từ văn bản nông dân")

    return farmer_text

def farmer_to_sci_en_pipeline(text, tts_enabled=True):
    """Farmer → Scientific Vietnamese → English + TTS (English)"""
    if not text or not text.strip():
        return ""

    sci_text = farmer_to_scientific(text)
    if sci_text.startswith(("Lỗi", "Mô hình", "Không thể")):
        return sci_text
    sci_text = number_to_vietnamese_words(sci_text)  # đảm bảo sạch trước khi dịch
    en_text = scientific_to_en(sci_text)
    if en_text.startswith(("Lỗi", "Mô hình", "Không thể")):
        return en_text

    if tts_enabled:
        synthesize_speech(en_text, lang="en", output_path="scientific_output.wav")
        logger.info("✓ Đã sinh file scientific_output.wav từ văn bản khoa học bằng tiếng Anh")

    return en_text
def vi_to_farmer_voice(text, output_path="vi_farmer_output.wav"):
    """Vietnamese Academic → Farmer + TTS (vi). Trả về text, lưu file WAV."""
    if not text or not text.strip():
        return ""

    farmer_text = vi_to_farmer(text)
    if isinstance(farmer_text, str) and farmer_text.startswith(("Lỗi", "Mô hình", "Không thể")):
        return farmer_text

    # Chuẩn hoá số trong tiếng Việt
    farmer_text = number_to_vietnamese_words(farmer_text)

    # Sinh audio tiếng Việt
    synthesize_speech(farmer_text, lang="vi", output_path=output_path)
    if 'logger' in globals():
        logger.info(f"✓ Đã sinh file {output_path} từ văn bản nông dân")

    return farmer_text


def farmer_to_scientific_voice(text, output_path="farmer_scientific_output.wav"):
    """Farmer → Vietnamese Academic + TTS (vi). Trả về text, lưu file WAV."""
    if not text or not text.strip():
        return ""

    sci_text = farmer_to_scientific(text)
    if isinstance(sci_text, str) and sci_text.startswith(("Lỗi", "Mô hình", "Không thể")):
        return sci_text

    # Chuẩn hoá số trong tiếng Việt
    sci_text = number_to_vietnamese_words(sci_text)

    # Sinh audio tiếng Việt
    synthesize_speech(sci_text, lang="vi", output_path=output_path)
    if 'logger' in globals():
        logger.info(f"✓ Đã sinh file {output_path} từ văn bản khoa học tiếng Việt")

    return sci_text