import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify, make_response

# Import functions from model_utils
try:
    from models.model_utils import (
        en_to_vi_farmer_pipeline,
        farmer_to_scientific, 
        vi_to_farmer,
        farmer_to_sci_en_pipeline,
        transcribe_audio,
        vi_to_farmer_voice,
        farmer_to_scientific_voice
    )
    print("✓ Successfully imported model functions")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Create upload folder
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Translation mapping
TRANSLATION_MAP = {
    "en_to_vi_farmer": en_to_vi_farmer_pipeline,
    "farmer_to_sci": farmer_to_scientific_voice,
    "sci_to_farmer": vi_to_farmer_voice,
    "farmer_to_sci_en": farmer_to_sci_en_pipeline
}
from openai import OpenAI
OPENAI_API_KEY = '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'
# Đảm bảo có API key
client = OpenAI(api_key=OPENAI_API_KEY)

def refine_text_with_chatgpt(text):
    """Refine text using ChatGPT API"""
    try:
        prompt = f" Bạn là một chuyên gia trong lĩnh vực nông nghiệp. Hãy dựa vào chính ngữ cảnh của đoạn để tìm các từ ngữ bị viết sai do phát âm không chuẩn. HÃY CHỈNH SỬA LẠI CHO ĐÚNG CÁC TỪ NGỮ VIẾT SAI ĐÓ. Đồng thời, viết các chữ số lại thành số (1,2,3,...). Các đơn vị viết lại với dạng ký hiệu quốc tế. LƯU Ý: các từ được nói như: trên lít, trên ngày,... của các đơn vị thì vẫn giữ nguyên. CHỈ TÌM TỪ SAI VÀ CHỈNH SỬA, KHÔNG THAY ĐỔI VĂN NÓI TRONG ĐOẠN, KHÔNG VIẾT LẠI CÂU. Chỉ trả về đoạn văn sau chỉnh sửa, không nói gì thêm. Nếu đoạn văn không có lỗi chính tả nào thì trả lại đoạn văn gốc, không nói gì thêm:\n\n{text}"
        
        response = client.chat.completions.create(
            model="gpt-4",  # hoặc "gpt-4"
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia ngôn ngữ tiếng Việt."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        refined_text = response.choices[0].message.content
        return refined_text
    
    except Exception as e:
        logger.error(f"Error refining text: {e}")
        return f"Lỗi khi gọi ChatGPT: {e}"
def process_translation(task, input_text):
    """Process translation based on task"""
    if not input_text or not input_text.strip():
        return "Vui lòng nhập văn bản."
    
    try:
        translation_func = TRANSLATION_MAP.get(task)
        if not translation_func:
            return f"Nhiệm vụ không hợp lệ: {task}"
        
        result = translation_func(input_text)
        return result if result else "Không thể dịch văn bản này."
        
    except Exception as e:
        logger.error(f"Translation error for task {task}: {e}")
        return f"Lỗi dịch thuật: {e}"

def process_audio_file(audio_file, audio_lang):
    """Save and process audio file"""
    if not audio_file or audio_file.filename == "":
        return "", "Không có file âm thanh."
    
    filename = f"{uuid.uuid4().hex}_{audio_file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        audio_file.save(filepath)
        transcribed_text = transcribe_audio(filepath, audio_lang)

        # 🌟 Thêm bước chỉnh sửa lại văn bản bằng ChatGPT
        refined_text = refine_text_with_chatgpt(transcribed_text)

        return refined_text, None
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return "", f"Lỗi xử lý file âm thanh: {e}"
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form data
            task = request.form.get("task", "en_to_vi_farmer")
            input_text = request.form.get("input_text", "").strip()
            audio_file = request.files.get("audio_file")
            audio_lang = request.form.get("audio_lang", "vi")
            
            # Process audio if provided
            if audio_file and audio_file.filename:
                transcribed_text, error = process_audio_file(audio_file, audio_lang)
                logger.info(f"[DEBUG] Transcribed text: '{transcribed_text}'")
                if error:
                    return jsonify({
                        'input': '', 'output': error, 'task': task, 'audio_lang': audio_lang
                    }), 400
                input_text = transcribed_text
            
            # Process translation
            if input_text and not input_text.startswith(("Lỗi", "Mô hình")):
                output = process_translation(task, input_text)
            elif input_text.startswith(("Lỗi", "Mô hình")):
                output = input_text
            else:
                output = "Vui lòng nhập văn bản hoặc chọn file âm thanh."
            
            return jsonify({
                'input': input_text,
                'task': task,
                'output': output,
                'audio_lang': audio_lang
            })
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return jsonify({
                'input': '',
                'output': f'Lỗi server: {str(e)}',
                'task': request.form.get("task", "en_to_vi_farmer"),
                'audio_lang': request.form.get("audio_lang", "vi")
            }), 500
    
    # GET request - render template
    return render_template("index.html", result={
        'input': '', 'output': '', 'task': 'en_to_vi_farmer', 'audio_lang': 'vi'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'input': '', 'output': 'File quá lớn. Vui lòng chọn file nhỏ hơn 16MB.',
        'task': 'en_to_vi_farmer', 'audio_lang': 'vi'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", result={
        'input': '', 'output': 'Trang không tồn tại.', 
        'task': 'en_to_vi_farmer', 'audio_lang': 'vi'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'input': '', 'output': 'Lỗi server nội bộ. Vui lòng thử lại sau.',
        'task': 'en_to_vi_farmer', 'audio_lang': 'vi'
    }), 500

if __name__ == "__main__":
    # Check if models directory exists
    if not os.path.exists("models"):
        print("Warning: models/ directory not found. Creating it...")
        os.makedirs("models", exist_ok=True)
    
    print("Starting AgriBridge application...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)