"""Streamlit prototype. Local inference by default; API_URL enables FastAPI mode."""

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from textstyle.predict import Predictor, resolve_model_dir, validate_text
from textstyle.settings import DEFAULT_ARTIFACTS, MAX_CHARS

st.set_page_config(page_title="TextScope · Стиль і тон тексту", page_icon="✦", layout="wide")
st.markdown(
    """<style>
.block-container { max-width: 1140px; padding-top: 2.5rem; }
h1 { letter-spacing: -0.045em; font-weight: 750 !important; }
h2, h3 { letter-spacing: -0.025em; }
[data-testid="stMetricValue"] { color: #2130a4; }
[data-testid="stSidebar"] { border-right: 1px solid #e6e9f2; }
</style>""",
    unsafe_allow_html=True,
)

LABELS = {
    "academic": "Академічний",
    "business": "Діловий",
    "formal": "Офіційний",
    "informal": "Неформальний",
    "literary": "Художній",
    "aggressive": "Агресивний",
    "friendly": "Дружній",
    "neutral": "Нейтральний",
    "sarcastic": "Саркастичний",
    "urgent": "Терміновий",
}
EXAMPLES = {
    "Ділове повідомлення": "Dear colleagues, please send the revised budget before today's meeting. We need your confirmation by noon.",
    "Дружній чат": "Hey! Thanks for helping me yesterday. You made my day, and I'd love to grab a coffee with you soon!",
    "Сарказм": "Wonderful, another meeting that could have been an email. Exactly what our busy schedule needed.",
    "Науковий текст": "The observed results indicate a positive association between the variables. Further experiments are required to assess the robustness of this finding.",
}
API_URL = os.getenv("API_URL", "").rstrip("/")


@st.cache_resource(show_spinner="Завантаження нейромережі…")
def local_predictor(path: str, modified_ns: int):
    return Predictor(Path(path))


def get_local():
    path = resolve_model_dir()
    return local_predictor(str(path), (path / "metadata.json").stat().st_mtime_ns)


def infer(texts):
    if API_URL:
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.post(f"{API_URL}/predict/batch", json={"texts": texts})
            response.raise_for_status()
            return response.json()["results"]
    return get_local().predict_batch(texts)


with st.sidebar:
    st.markdown("## ✦ TextScope")
    st.caption("Стиль і тон англійського тексту")
    st.divider()
    st.markdown("**Як скористатися**")
    st.write("Вставте повідомлення англійською та оберіть потрібний аналіз.")
    st.caption("Модель оцінює мовні ознаки тексту. Контекст розмови може змінювати його сприйняття.")
    st.divider()
    st.caption("Режим: FastAPI" if API_URL else "Режим: локальна нейромережа")
    if not API_URL:
        try:
            engine = get_local()
            st.success(f"Модель готова: {engine.metadata['architecture'].upper()}")
        except (OSError, ValueError, RuntimeError) as error:
            st.error("Не вдалося завантажити навчену модель")
            st.code("python -m textstyle.train")
            st.caption(str(error))

st.caption("DEEP LEARNING / ENGLISH TEXT ANALYSIS")
st.title("Що передає ваш текст?")
st.write("Визначайте стиль повідомлення та його емоційний тон за допомогою нейромережі.")
analysis_tab, batch_tab, quality_tab, about_tab = st.tabs(
    ["Аналіз тексту", "Пакетна обробка", "Якість моделі", "Про проєкт"]
)

with analysis_tab:
    example_name = st.selectbox("Спробуйте приклад", list(EXAMPLES), key="example_choice")
    if st.button("Вставити приклад", key="use_example"):
        st.session_state["input_text"] = EXAMPLES[example_name]
    with st.form("analyze"):
        text = st.text_area(
            "Текст англійською",
            value=EXAMPLES["Ділове повідомлення"],
            height=190,
            max_chars=MAX_CHARS,
            key="input_text",
            placeholder="Paste an English message here…",
        )
        a, b, c = st.columns(3)
        both = a.form_submit_button("Визначити стиль і тон", type="primary", width="stretch")
        style_only = b.form_submit_button("Тільки стиль", width="stretch")
        tone_only = c.form_submit_button("Тільки тон", width="stretch")
    if both or style_only or tone_only:
        try:
            validate_text(text)
            with st.spinner("Аналізуємо текст…"):
                result = infer([text])[0]
            st.session_state["prediction"] = result
            st.session_state["prediction_text"] = text
            st.session_state["prediction_mode"] = "both" if both else ("style" if style_only else "tone")
        except (ValueError, OSError, RuntimeError, httpx.HTTPError) as error:
            st.session_state.pop("prediction", None)
            st.error(f"Аналіз недоступний: {error}")
    if "prediction" in st.session_state:
        result = st.session_state["prediction"]
        mode = st.session_state["prediction_mode"]
        st.subheader("Результат аналізу")
        st.caption(
            f"Модель {result['model'].upper()} · {result['token_count']} токенів · {result['latency_ms']:.1f} мс на текст"
        )
        columns = st.columns(2 if mode == "both" else 1)
        targets = ["style", "tone"] if mode == "both" else [mode]
        for column, target in zip(columns, targets, strict=True):
            with column:
                st.metric(
                    "Стиль тексту" if target == "style" else "Емоційний тон",
                    LABELS[result[target]],
                    f"Оцінка моделі {result[f'{target}_confidence']:.0%}",
                    delta_color="off",
                )
                scores = pd.DataFrame(
                    {
                        "Клас": [LABELS[k] for k in result[f"{target}_scores"]],
                        "Оцінка": list(result[f"{target}_scores"].values()),
                    }
                )
                st.bar_chart(scores.set_index("Клас"), color="#5264dd", horizontal=True, height=220)
        for warning in result["warnings"]:
            st.warning(warning)
        st.caption(
            "Оцінки показують упевненість моделі та не гарантують правильність відповіді. Температурне калібрування використовує валідаційну вибірку."
        )
        st.download_button(
            "Завантажити результат JSON",
            json.dumps(result, indent=2, ensure_ascii=False),
            file_name="textscope_prediction.json",
            mime="application/json",
        )

with batch_tab:
    st.subheader("Кілька повідомлень за один запуск")
    st.write("Завантажте CSV з колонкою text. За один запуск можна обробити до 32 текстів.")
    st.download_button(
        "Зразок CSV", 'text\n"Thanks for your help!"\n"Please send the contract today."\n', "example.csv", "text/csv"
    )
    uploaded = st.file_uploader("CSV з повідомленнями", type=["csv"])
    if uploaded is not None and st.button("Проаналізувати файл", type="primary"):
        try:
            if uploaded.size > 5 * 1024 * 1024:
                raise ValueError("Розмір CSV перевищує 5 МБ")
            frame = pd.read_csv(uploaded, keep_default_na=False)
            if "text" not in frame:
                raise ValueError("CSV повинен містити колонку text")
            if not 1 <= len(frame) <= 32:
                raise ValueError("CSV повинен містити від 1 до 32 рядків")
            texts = [validate_text(str(value)) for value in frame.text]
            predictions = infer(texts)
            output = pd.DataFrame(
                [
                    {
                        "text": text,
                        **{
                            k: row[k]
                            for k in [
                                "style",
                                "tone",
                                "style_confidence",
                                "tone_confidence",
                                "truncated",
                            ]
                        },
                    }
                    for text, row in zip(texts, predictions, strict=True)
                ]
            )
            st.session_state["batch_results"] = output
        except (ValueError, OSError, RuntimeError, httpx.HTTPError, pd.errors.ParserError) as error:
            st.session_state.pop("batch_results", None)
            st.error(str(error))
    if "batch_results" in st.session_state:
        st.dataframe(st.session_state["batch_results"], hide_index=True, width="stretch")
        st.download_button(
            "Завантажити результати CSV",
            st.session_state["batch_results"].to_csv(index=False).encode("utf-8-sig"),
            "textscope_batch.csv",
            "text/csv",
        )

with quality_tab:
    st.subheader("Перевірка на відкладених текстах")
    experiment_path = DEFAULT_ARTIFACTS / "experiment.json"
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        st.caption(
            "Результати комплектного експерименту. Вибір моделі спирається на validation macro F1, тестова вибірка залишається окремою."
        )
        records = []
        for name, run in experiment["models"].items():
            records.append(
                {
                    "Модель": name.upper(),
                    "F1 стилю": round(run["test"]["style"]["f1_macro"], 3),
                    "F1 тону": round(run["test"]["tone"]["f1_macro"], 3),
                    "Обидві відповіді правильні": round(run["test"]["joint_accuracy"], 3),
                }
            )
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
        model_name = st.selectbox("Результати моделі", list(experiment["models"]), key="quality_model")
        curve = DEFAULT_ARTIFACTS / model_name / "learning_curves.png"
        if curve.exists():
            st.image(str(curve), caption="Навчання та валідація")
        left, right = st.columns(2)
        for column, target in [(left, "style"), (right, "tone")]:
            path = DEFAULT_ARTIFACTS / model_name / "test" / f"{target}_confusion.png"
            if path.exists():
                column.image(
                    str(path), caption="Матриця помилок стилю" if target == "style" else "Матриця помилок тону"
                )
        st.info(
            "Оцінювання стосується цього невеликого навчального корпусу. Результати на реальних повідомленнях можуть відрізнятися."
        )
    else:
        st.info("Запустіть навчання, щоб отримати звіт експерименту.")

with about_tab:
    st.subheader("Глибинне навчання для аналізу текстів")
    st.write(
        "Одна нейромережа формує представлення тексту та передає його двом класифікаторам: для стилю і для тону. Проєкт містить CNN та BiLSTM, а також додатковий режим донавчання трансформера."
    )
    st.write(
        "Підготовка даних усуває суперечливі повтори, виправляє назви класів і тримає близькі тексти в одній вибірці. Словник CNN та BiLSTM використовує тільки навчальні тексти."
    )
    st.markdown("**Джерело корпусу:** [zave52/text-style-detection](https://github.com/zave52/text-style-detection)")
    st.caption(
        "Нейромережа оцінює саме текст. Вона не визначає психологічний стан автора та не має доступу до контексту розмови."
    )
