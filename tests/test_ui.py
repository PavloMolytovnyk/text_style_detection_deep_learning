from pathlib import Path

from streamlit.testing.v1 import AppTest

from textstyle.settings import ROOT


def test_streamlit_runs_and_analyzes_real_text(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(ROOT / "artifacts" / "cnn" / "model"))
    app = AppTest.from_file(Path(ROOT / "streamlit_app.py"), default_timeout=45).run()
    assert not app.exception
    app.text_area[0].set_value("Hey friend, thanks for your amazing help!")
    next(button for button in app.button if button.label == "Визначити стиль і тон").click().run()
    assert not app.exception
    assert app.session_state["prediction"]["model"] == "cnn"
    assert len(app.metric) == 2
