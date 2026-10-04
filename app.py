import streamlit as st
import fitz
from PIL import Image
import pytesseract
import io
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="نَقَل — محقق النقل العلمي",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# ==================== التصميم ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&family=Amiri:wght@400;700&display=swap');

    /* الخلفية العامة */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0e8 100%);
        direction: rtl;
        text-align: right;
    }

    /* كل النصوص */
    html, body, [class*="css"], p, div, span, label, h1, h2, h3, h4, h5, h6 {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
    }

    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* الهيدر الرئيسي */
    .hero-header {
        background: linear-gradient(135deg, #0d4a3d 0%, #1a6b56 50%, #2d8a6f 100%);
        padding: 40px 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(13, 74, 61, 0.25);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-header h1 {
        color: white !important;
        font-size: 42px;
        font-weight: 900;
        margin: 0 0 10px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .hero-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 18px;
        margin: 0;
        font-weight: 400;
    }

    .hero-icon {
        font-size: 60px;
        margin-bottom: 10px;
    }

    /* بطاقات الحقول */
    .input-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(13, 74, 61, 0.08);
        margin-bottom: 20px;
        height: 100%;
    }

    .card-title {
        color: #0d4a3d;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e8f0e8;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* رفع الملفات */
    .stFileUploader {
        direction: rtl;
    }

    .stFileUploader > div {
        border: 2px dashed #2d8a6f !important;
        border-radius: 12px !important;
        background: #f8fbf9 !important;
        padding: 20px !important;
    }

    .stFileUploader > div:hover {
        border-color: #0d4a3d !important;
        background: #f0f7f3 !important;
    }

    /* مربع النص */
    .stTextArea textarea {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Amiri', 'Tajawal', serif !important;
        font-size: 17px !important;
        line-height: 1.9 !important;
        border-radius: 12px !important;
        border: 2px solid #e0e8e3 !important;
        padding: 15px !important;
        background: #fafcfb !important;
    }

    .stTextArea textarea:focus {
        border-color: #2d8a6f !important;
        box-shadow: 0 0 0 3px rgba(45, 138, 111, 0.1) !important;
    }

    /* الزر الرئيسي */
    .stButton button {
        background: linear-gradient(135deg, #0d4a3d 0%, #2d8a6f 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 32px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        font-family: 'Tajawal', sans-serif !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(13, 74, 61, 0.3) !important;
        margin-top: 10px !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(13, 74, 61, 0.4) !important;
    }

    /* صندوق النتيجة */
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 15px 0;
        border-right: 5px solid #0d4a3d;
    }

    /* شارة النتيجة */
    .score-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 16px;
        margin: 5px 0;
    }

    .score-high { background: #d4f5e0; color: #0d4a3d; }
    .score-mid { background: #fff3cd; color: #856404; }
    .score-low { background: #f8d7da; color: #721c24; }

    /* النص المستخرج */
    .source-text {
        direction: rtl;
        text-align: right;
        font-family: 'Amiri', serif;
        font-size: 18px;
        line-height: 2;
        background: #f8fbf9;
        padding: 18px 22px;
        border-radius: 12px;
        border-right: 4px solid #2d8a6f;
        color: #1a2e28;
        margin: 10px 0;
    }

    /* أزرار expander */
    .streamlit-expanderHeader {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Tajawal', sans-serif !important;
        font-weight: 600 !important;
        background: #f8fbf9 !important;
        border-radius: 10px !important;
        padding: 12px 18px !important;
        color: #0d4a3d !important;
    }

    .streamlit-expanderContent {
        direction: rtl !important;
        text-align: right !important;
    }

    /* التنبيهات */
    .stAlert {
        direction: rtl !important;
        text-align: right !important;
        border-radius: 12px !important;
        font-family: 'Tajawal', sans-serif !important;
    }

    /* التذييل */
    .custom-footer {
        text-align: center;
        padding: 20px;
        color: #6b7b75;
        font-size: 14px;
        margin-top: 40px;
        border-top: 1px solid #d8e5dd;
    }

    /* علامات الميزات */
    .features-row {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0 30px 0;
        flex-wrap: wrap;
    }

    .feature-pill {
        background: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        color: #0d4a3d;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e0e8e3;
    }
</style>
""", unsafe_allow_html=True)


def fix_arabic_text(text):
    if not text:
        return text
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def display_arabic(text):
    safe = text.replace('<', '&lt;').replace('>', '&gt;')
    st.markdown(f'<div class="source-text">{safe}</div>', unsafe_allow_html=True)


# ==================== الهيدر ====================
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">📖</div>
    <h1>نَقَل</h1>
    <p>محقق النقل العلمي — تحقق من دقة الاقتباس مقارنة بالمصدر الأصلي</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features-row">
    <span class="feature-pill">🔒 لا هلوسة</span>
    <span class="feature-pill">📚 تحقق مقيد بالمصدر</span>
    <span class="feature-pill">⚡ نتائج فورية</span>
    <span class="feature-pill">🎯 مطابقة دلالية</span>
</div>
""", unsafe_allow_html=True)


# ==================== المدخلات ====================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📚 الخطوة 1: ارفع المصدر الموثوق</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF أو صورة (PNG / JPG)", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✍️ الخطوة 2: الصق الاقتباس</div>', unsafe_allow_html=True)
    quote = st.text_area("النص المراد التحقق منه", height=180, label_visibility="collapsed",
                          placeholder="مثال: قال ابن كثير في تفسيره...")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# زر التحقق
if st.button("🔍 تحقق الآن", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ الرجاء رفع المصدر أولاً")
    elif not quote.strip():
        st.warning("⚠️ الرجاء إدخال الاقتباس المراد التحقق منه")
    else:
        with st.spinner("⏳ جاري التحليل الدلالي ومطابقة النص..."):
            file_bytes = uploaded_file.read()
            pages_data = []

            try:
                if uploaded_file.type == "application/pdf":
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    for i, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            pages_data.append((i + 1, text))
                    if not pages_data:
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        for i, page in enumerate(doc):
                            pix = page.get_pixmap(dpi=200)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            text = pytesseract.image_to_string(img, lang='ara+eng')
                            if text.strip():
                                pages_data.append((i + 1, text))
                else:
                    img = Image.open(io.BytesIO(file_bytes))
                    text = pytesseract.image_to_string(img, lang='ara+eng')
                    pages_data.append((1, text))
            except Exception as e:
                st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
                st.stop()

            if not pages_data:
                st.error("❌ لم نتمكن من استخراج نص. جرّب ملف PDF نصي أو صورة أوضح.")
                st.stop()

            pages_data = [(p, fix_arabic_text(t)) for p, t in pages_data]

            chunks = []
            for page_num, text in pages_data:
                text = re.sub(r'\s+', ' ', text)
                parts = re.split(r'[.\n؟!،؛:؛]', text)
                for part in parts:
                    part = part.strip()
                    if len(part) > 15:
                        chunks.append((page_num, part))

            if not chunks:
                st.error("❌ النص المستخرج قصير جداً. جرّب مصدراً أوضح.")
                st.stop()

            chunk_texts = [c[1] for c in chunks]
            chunk_emb = model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=False)
            quote_emb = model.encode([quote.strip()], convert_to_tensor=False, show_progress_bar=False)

            sims = cosine_similarity(quote_emb, chunk_emb)[0]
            top_indices = np.argsort(sims)[::-1][:3]

            best_score = float(sims[top_indices[0]]) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 📊 نتائج التحقق")

            if best_score >= 85:
                st.markdown(f'''
                <div class="result-card" style="border-right-color: #0d4a3d;">
                    <span class="score-badge score-high">✅ مطابقة عالية — {best_score:.1f}%</span>
                    <p style="color: #0d4a3d; font-weight: 600; margin-top: 10px;">
                    النقل موثوق — الاقتباس مطابق للأصل بدرجة عالية.
                    </p>
                </div>
                ''', unsafe_allow_html=True)
            elif best_score >= 65:
                st.markdown(f'''
                <div class="result-card" style="border-right-color: #f0ad4e;">
                    <span class="score-badge score-mid">⚠️ تشابه جزئي — {best_score:.1f}%</span>
                    <p style="color: #856404; font-weight: 600; margin-top: 10px;">
                    راجع النص الأصلي بدقة — قد يوجد اختلاف في الصياغة أو نقص.
                    </p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="result-card" style="border-right-color: #d9534f;">
                    <span class="score-badge score-low">❌ لا يوجد تطابق موثوق — {best_score:.1f}%</span>
                    <p style="color: #721c24; font-weight: 600; margin-top: 10px;">
                    الاقتباس غير موجود في المصدر المرفوع. قد يكون محرّفاً أو من مصدر آخر. يُنصح بالرجوع لمصدر آخر أو مراجعة المختص.
                    </p>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown("### 📌 أفضل المطابقات من المصدر:")
            for rank, idx in enumerate(top_indices, 1):
                page_num, chunk_text = chunks[idx]
                sim = float(sims[idx]) * 100

                with st.expander(f"النتيجة {rank} — تشابه {sim:.1f}% — الصفحة {page_num}", expanded=(rank == 1)):
                    st.markdown(f"**النص في المصدر (صفحة {page_num}):**")
                    display_arabic(chunk_text)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("🔒 التحقق مبني على المصدر الذي رفعته فقط — لا توليد حر، لا هلوسة. عند عدم وجود دليل كافٍ، يمتنع النظام عن الإجابة.")


# ==================== التذييل ====================
st.markdown("""
<div class="custom-footer">
    <p><strong>نَقَل</strong> — نواة تطبيق «بيان» لخدمة المحتوى الإسلامي</p>
    <p>تحدي الذكاء الاصطناعي في خدمة المحتوى الإسلامي 2026 | مؤسسة باذل الأهلية</p>
    <p>المسار: أدوات المعرفة والتحقق لتمكين المعرّفين بالإسلام</p>
</div>
""", unsafe_allow_html=True)
