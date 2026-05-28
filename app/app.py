"""
app.py
======
Flask REST API untuk Essay Grader AI.
Model: Siamese BiLSTM + GloVe (TensorFlow)
Menerima jawaban siswa + kunci jawaban, mengembalikan skor & feedback.

Endpoint:
    POST /grade       — nilai satu jawaban
    POST /grade/batch — nilai banyak jawaban sekaligus
    GET  /health      — cek status API
    GET  /info        — informasi model yang aktif

Dependensi:
    pip install flask flask-cors tensorflow PySastrawi nltk langdetect google-generativeai

Jalankan:
    python app/app.py
"""

import os
import sys
import pickle
import logging
import traceback
import numpy as np
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from utils.preprocessing import TextPreprocessor, compare_texts

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Inisialisasi Flask ────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.getenv("MODEL_PATH",      os.path.join(BASE_PATH, "model", "essay_grader_bilstm.keras"))
TOKENIZER_PATH  = os.getenv("TOKENIZER_PATH",  os.path.join(BASE_PATH, "model", "tokenizer.pkl"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 5000))
MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE",  50))
MAX_LEN         = 100  # harus sama dengan saat training
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")  # diisi oleh rekan fullstack

# ── Setup Gemini API ──────────────────────────────────────────────────────────
gemini_model = None

def setup_gemini():
    """Inisialisasi Gemini API. Dipanggil saat startup."""
    global gemini_model
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY tidak ditemukan. Fitur AI feedback tidak aktif.")
        return
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini API berhasil diinisialisasi.")
    except Exception as e:
        logger.warning(f"Gagal inisialisasi Gemini API: {e}")
        gemini_model = None

# ── Load model & preprocessor ─────────────────────────────────────────────────
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    do_stemming=False,
    keep_numbers=True,
    min_token_len=2
)

model          = None
tokenizer_keras = None
model_type     = "baseline_jaccard"


def clean_text(text):
    """Preprocessing teks sesuai pipeline BiLSTM."""
    if not isinstance(text, str) or not text.strip():
        return ''
    return preprocessor.full_pipeline(text)


def load_model():
    """Load model BiLSTM dan tokenizer dari disk."""
    global model, tokenizer_keras, model_type

    model_ok     = os.path.exists(MODEL_PATH)
    tokenizer_ok = os.path.exists(TOKENIZER_PATH)

    if model_ok and tokenizer_ok:
        try:
            import tensorflow as tf

            # Custom layer yang dibutuhkan saat load model
            class CosineSimilarityLayer(tf.keras.layers.Layer):
                def __init__(self, **kwargs):
                    super(CosineSimilarityLayer, self).__init__(**kwargs)
                def call(self, inputs):
                    vec_a, vec_b = inputs
                    vec_a_norm   = tf.math.l2_normalize(vec_a, axis=-1)
                    vec_b_norm   = tf.math.l2_normalize(vec_b, axis=-1)
                    similarity   = tf.reduce_sum(vec_a_norm * vec_b_norm, axis=-1, keepdims=True)
                    return tf.clip_by_value(similarity, 0.0, 1.0)
                def get_config(self):
                    return super().get_config()

            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'CosineSimilarityLayer': CosineSimilarityLayer}
            )

            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer_keras = pickle.load(f)

            model_type = "bilstm_glove"
            logger.info(f"Model loaded: {model_type}")

        except Exception as e:
            logger.warning(f"Gagal load model ({e}), fallback ke Jaccard baseline.")
            model_type = "baseline_jaccard"
    else:
        logger.info("Model belum tersedia, berjalan dalam mode Jaccard baseline.")
        model_type = "baseline_jaccard"


load_model()


# ── Scoring Engine ────────────────────────────────────────────────────────────

def texts_to_sequences(texts, max_len=MAX_LEN):
    """Konversi teks ke sequence integer dengan padding."""
    import tensorflow as tf
    seqs = tokenizer_keras.texts_to_sequences(list(texts))
    return tf.keras.preprocessing.sequence.pad_sequences(
        seqs, maxlen=max_len, padding='post', truncating='post'
    )


def score_with_bilstm(student_answer: str, reference_answer: str) -> dict:
    """Scoring menggunakan model BiLSTM."""
    import numpy as np

    stu_clean = clean_text(student_answer)
    ref_clean = clean_text(reference_answer)

    stu_seq = texts_to_sequences([stu_clean])
    ref_seq = texts_to_sequences([ref_clean])

    inputs  = {'student_input': stu_seq, 'reference_input': ref_seq}
    proba   = model.predict(inputs, verbose=0)[0]
    raw_cls = int(np.argmax(proba))

    # Konversi kelas ke similarity score
    sim_map    = {0: 0.0, 1: 0.5, 2: 1.0}
    similarity = sim_map[raw_cls]

    return {
        'similarity'   : round(similarity, 4),
        'raw_class'    : raw_cls,
        'probabilities': {
            'skor_0': round(float(proba[0]), 4),
            'skor_1': round(float(proba[1]), 4),
            'skor_2': round(float(proba[2]), 4),
        },
        'method'       : 'bilstm_glove',
    }


def score_with_jaccard(student_answer: str, reference_answer: str) -> dict:
    """Baseline scoring menggunakan Jaccard similarity."""
    result     = compare_texts(student_answer, reference_answer, preprocessor)
    similarity = result["jaccard_similarity"]
    return {
        'similarity'   : round(similarity, 4),
        'raw_class'    : None,
        'probabilities': {},
        'method'       : 'jaccard',
    }


def similarity_to_score(similarity: float, max_score: float) -> float:
    """Konversi similarity (0.0–1.0) ke skor angka."""
    sim = max(0.0, min(1.0, similarity))
    if sim < 0.20:
        normalized = (sim / 0.20) * 30
    elif sim < 0.50:
        normalized = 30 + ((sim - 0.20) / 0.30) * 35
    elif sim < 0.75:
        normalized = 65 + ((sim - 0.50) / 0.25) * 20
    else:
        normalized = 85 + ((sim - 0.75) / 0.25) * 15
    return round((normalized / 100) * max_score, 1)


def score_to_grade(score: float, max_score: float) -> str:
    """Konversi skor numerik ke grade huruf."""
    pct = (score / max_score) * 100 if max_score > 0 else 0
    if pct >= 90: return "A"
    if pct >= 80: return "B+"
    if pct >= 70: return "B"
    if pct >= 60: return "C+"
    if pct >= 50: return "C"
    if pct >= 40: return "D"
    return "E"


def generate_feedback(similarity: float, reference_answer: str, student_answer: str) -> dict:
    """Generate feedback otomatis berdasarkan skor similarity."""
    ref_tokens = set(preprocessor.full_pipeline(reference_answer, return_tokens=True))
    stu_tokens = set(preprocessor.full_pipeline(student_answer,   return_tokens=True))

    keywords_found   = sorted(stu_tokens & ref_tokens)[:8]
    keywords_missing = sorted(ref_tokens - stu_tokens)[:8]

    stu_word_count = len(student_answer.split())
    ref_word_count = len(reference_answer.split())
    length_ratio   = stu_word_count / ref_word_count if ref_word_count > 0 else 0

    if similarity >= 0.75:
        relevance_label = "Sangat Relevan"
        relevance_color = "green"
        kalimat1 = "Jawaban kamu sudah sangat baik dan mencakup poin-poin utama dengan lengkap."
    elif similarity >= 0.50:
        relevance_label = "Cukup Relevan"
        relevance_color = "yellow"
        kalimat1 = "Jawaban kamu sudah cukup baik dan menunjukkan pemahaman yang memadai."
    elif similarity >= 0.30:
        relevance_label = "Kurang Relevan"
        relevance_color = "orange"
        kalimat1 = "Jawaban kamu menunjukkan pemahaman dasar, namun perlu dikembangkan lebih lanjut."
    else:
        relevance_label = "Tidak Relevan"
        relevance_color = "red"
        kalimat1 = "Jawaban kamu belum mencerminkan pemahaman yang cukup terhadap materi."

    if keywords_found:
        found_str = ", ".join([f'"{k}"' for k in keywords_found[:4]])
        kalimat2  = f"Kamu berhasil menyebutkan beberapa konsep penting seperti {found_str}."
    else:
        kalimat2 = "Belum ada kata kunci utama dari kunci jawaban yang teridentifikasi."

    if keywords_missing and similarity < 0.75:
        missing_str = ", ".join([f'"{k}"' for k in keywords_missing[:4]])
        kalimat3    = f"Coba lengkapi jawabanmu dengan konsep-konsep seperti {missing_str}."
    elif length_ratio < 0.5 and similarity < 0.75:
        kalimat3 = "Jawaban terlalu singkat. Coba jelaskan dengan lebih detail."
    else:
        kalimat3 = "Pertahankan kualitas jawabanmu."

    return {
        "message"         : f"{kalimat1} {kalimat2} {kalimat3}",
        "relevance_label" : relevance_label,
        "relevance_color" : relevance_color,
        "keywords_found"  : keywords_found,
        "keywords_missing": keywords_missing,
    }


def generate_ai_feedback(student_answer: str, reference_answer: str,
                         score: float, max_score: float, grade: str) -> str:
    """
    Generate feedback natural menggunakan Gemini API.
    Fitur tambahan (side quest) — aktif jika GEMINI_API_KEY tersedia.
    Jika Gemini tidak tersedia, return string kosong.
    """
    if gemini_model is None:
        return ""
    try:
        prompt = f"""Kamu adalah asisten guru yang membantu memberikan feedback pada jawaban siswa.

Kunci Jawaban: {reference_answer}
Jawaban Siswa: {student_answer}
Skor yang diperoleh: {score}/{max_score} (Grade: {grade})

Berikan feedback singkat (2-3 kalimat) dalam Bahasa Indonesia yang:
1. Menjelaskan kelebihan jawaban siswa
2. Menjelaskan apa yang perlu diperbaiki atau ditambahkan
3. Memberikan semangat kepada siswa

Langsung tulis feedbacknya tanpa preamble."""

        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        return ""


def grade_answer(student_answer: str, reference_answer: str, max_score: float = 100.0) -> dict:
    """Fungsi inti penilaian."""
    if model_type == "bilstm_glove" and model is not None:
        score_data = score_with_bilstm(student_answer, reference_answer)
    else:
        score_data = score_with_jaccard(student_answer, reference_answer)

    similarity = score_data["similarity"]
    method     = score_data["method"]
    score      = similarity_to_score(similarity, max_score)
    grade      = score_to_grade(score, max_score)
    feedback   = generate_feedback(similarity, reference_answer, student_answer)

    # Generate AI feedback (Gemini) — fitur tambahan
    ai_feedback = generate_ai_feedback(student_answer, reference_answer, score, max_score, grade)

    return {
        "score"           : score,
        "max_score"       : max_score,
        "grade"           : grade,
        "similarity"      : similarity,
        "scoring_method"  : method,
        "relevance_label" : feedback["relevance_label"],
        "relevance_color" : feedback["relevance_color"],
        "feedback"        : feedback["message"],
        "ai_feedback"     : ai_feedback,
        "keywords_found"  : feedback["keywords_found"],
        "keywords_missing": feedback["keywords_missing"],
    }


# ── Middleware / Helpers ──────────────────────────────────────────────────────

def validate_text(text, field_name: str):
    if not text:
        return f"Field '{field_name}' wajib diisi."
    if not isinstance(text, str):
        return f"Field '{field_name}' harus berupa string."
    if len(text.strip()) == 0:
        return f"Field '{field_name}' tidak boleh kosong."
    if len(text) > MAX_TEXT_LENGTH:
        return f"Field '{field_name}' melebihi batas {MAX_TEXT_LENGTH} karakter."
    return None


def api_response(data: dict, status: int = 200):
    return jsonify({
        "status"   : "success" if status < 400 else "error",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **data,
    }), status


def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Unhandled error: {traceback.format_exc()}")
            return api_response({"message": "Terjadi kesalahan internal server."}, 500)
    return wrapper


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return api_response({
        "api_status"   : "ok",
        "model_type"   : model_type,
        "model_ready"  : model_type == "bilstm_glove",
        "gemini_active": gemini_model is not None,
    })


@app.route("/info", methods=["GET"])
def info():
    return api_response({
        "api_version"  : "2.0.0",
        "model_type"   : model_type,
        "model_ready"  : model_type == "bilstm_glove",
        "architecture"  : "Siamese BiLSTM + GloVe Embedding",
        "gemini_active" : gemini_model is not None,
        "max_text_len" : MAX_TEXT_LENGTH,
        "max_batch"    : MAX_BATCH_SIZE,
        "endpoints"    : {
            "POST /grade"      : "Nilai satu jawaban",
            "POST /grade/batch": "Nilai banyak jawaban",
            "GET  /health"     : "Cek status API",
            "GET  /info"       : "Info model & API",
        },
    })


@app.route("/grade", methods=["POST"])
@handle_errors
def grade():
    body = request.get_json(silent=True)
    if not body:
        return api_response({"message": "Request body harus berformat JSON."}, 400)

    student_answer   = body.get("student_answer")
    reference_answer = body.get("reference_answer")
    max_score        = body.get("max_score", 100)

    err = validate_text(student_answer, "student_answer")
    if err:
        return api_response({"message": err}, 400)

    err = validate_text(reference_answer, "reference_answer")
    if err:
        return api_response({"message": err}, 400)

    if not isinstance(max_score, (int, float)) or max_score <= 0:
        return api_response({"message": "Field 'max_score' harus berupa angka positif."}, 400)

    result = grade_answer(student_answer, reference_answer, float(max_score))
    logger.info(f"/grade — score={result['score']}, method={result['scoring_method']}")
    return api_response(result)


@app.route("/grade/batch", methods=["POST"])
@handle_errors
def grade_batch():
    body = request.get_json(silent=True)
    if not body:
        return api_response({"message": "Request body harus berformat JSON."}, 400)

    items = body.get("items")
    if not items or not isinstance(items, list):
        return api_response({"message": "Field 'items' wajib diisi dan harus berupa array."}, 400)

    if len(items) > MAX_BATCH_SIZE:
        return api_response({"message": f"Maksimal {MAX_BATCH_SIZE} item per batch."}, 400)

    results, success, failed = [], 0, 0

    for idx, item in enumerate(items):
        item_id          = item.get("id", f"item_{idx+1}")
        student_answer   = item.get("student_answer")
        reference_answer = item.get("reference_answer")
        max_score        = item.get("max_score", 100)

        err = validate_text(student_answer, "student_answer")
        if not err:
            err = validate_text(reference_answer, "reference_answer")

        if err:
            results.append({"id": item_id, "error": err})
            failed += 1
            continue

        try:
            result       = grade_answer(student_answer, reference_answer, float(max_score))
            result["id"] = item_id
            results.append(result)
            success += 1
        except Exception as e:
            results.append({"id": item_id, "error": str(e)})
            failed += 1

    logger.info(f"/grade/batch — total={len(items)}, success={success}, failed={failed}")
    return api_response({
        "total"  : len(items),
        "success": success,
        "failed" : failed,
        "results": results,
    })


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return api_response({"message": "Endpoint tidak ditemukan."}, 404)

@app.errorhandler(405)
def method_not_allowed(e):
    return api_response({"message": "HTTP method tidak diizinkan."}, 405)

@app.errorhandler(500)
def internal_error(e):
    return api_response({"message": "Kesalahan internal server."}, 500)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Essay Grader API starting — port={port}, model={model_type}")
    app.run(host="0.0.0.0", port=port, debug=debug)
