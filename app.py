import streamlit as st
import fitz
from PIL import Image
import pytesseract
import io
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="نَقَل — محقق النقل العلمي", page_icon="📖", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

st.markdown("""
<style>
    .stApp { direction: rtl; text-align: right; }
    h1, h2, h3 { color: #0d4a3d; }
    .stTextArea textarea { direction: rtl; text-align: right; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("📖 نَقَل — محقق النقل العلمي")
st.caption("تحقّق من دقة النقل العلمي مقارنة بالمصدر الأصلي — بدون هلوسة، وبأمانة علمية")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📚 الخطوة 1: ارفع المصدر")
    uploaded_file = st.file_uploader("PDF أو صورة (PNG / JPG)", type=["pdf", "png", "jpg", "jpeg"])

with col2:
    st.markdown("### ✍️ الخطوة 2: الصق الاقتباس")
    quote = st.text_area("النص المراد التحقق منه", height=180, label_visibility="collapsed",
                          placeholder="مثال: قال ابن كثير في تفسيره...")

st.markdown("---")

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

            chunks = []
            for page_num, text in pages_data:
                text = re.sub(r'\s+', ' ', text)
                parts = re.split(r'[.\n؟!،؛:؛]', text)
                for part in parts:
                    part = part.strip()
                    if len(part) > 15:
                        chunks.append((page_num, part))

            if not chunks:
                st.error("❌ النص المستخرج قصير جداً.")
                st.stop()

            chunk_texts = [c[1] for c in chunks]
            chunk_emb = model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=False)
            quote_emb = model.encode([quote.strip()], convert_to_tensor=False, show_progress_bar=False)

            sims = cosine_similarity(quote_emb, chunk_emb)[0]
            top_indices = np.argsort(sims)[::-1][:3]

            st.markdown("---")
            st.subheader("📊 نتائج التحقق")

            best_score = float(sims[top_indices[0]]) * 100

            if best_score >= 85:
                st.success(f"✅ مطابقة عالية — نسبة التشابه {best_score:.1f}% — النقل موثوق")
            elif best_score >= 65:
                st.warning(f"⚠️ تشابه جزئي — نسبة التشابه {best_score:.1f}% — راجع النص الأصلي")
            else:
                st.error(f"❌ لا يوجد تطابق موثوق — أعلى نسبة {best_score:.1f}% — الاقتباس غير موجود أو محرّف")

            st.markdown("### 📌 أفضل المطابقات من المصدر:")
            for rank, idx in enumerate(top_indices, 1):
                page_num, chunk_text = chunks[idx]
                sim = float(sims[idx]) * 100
                with st.expander(f"النتيجة {rank} — تشابه {sim:.1f}% — الصفحة {page_num}", expanded=(rank == 1)):
                    st.markdown(f"**النص في المصدر (صفحة {page_num}):**")
                    st.info(chunk_text)

            st.caption("🔒 التحقق مبني على المصدر الذي رفعته فقط — لا توليد حر، لا هلوسة.")

st.markdown("---")
st.caption("تحدي الذكاء الاصطناعي في خدمة المحتوى الإسلامي 2026 | مؤسسة باذل الأهلية")
