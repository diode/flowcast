"""FlowCast dashboard (docs/dashboard.md): the city overview for customers, and the technical
view (docs/phase5.md) behind it.

  .venv\\Scripts\\python -m streamlit run dashboard\\app.py
"""
import os
from pathlib import Path

import streamlit as st

from netstudy import sample_data

st.set_page_config(page_title="FlowCast · City traffic", page_icon=":material/traffic:", layout="wide")
if "sample_data" not in st.session_state:          # fresh clone: unpack sample-data/*.zip into data/ once
    with st.spinner("Unpacking the sample towns ..."):
        sample_data.ensure()
    st.session_state.sample_data = True
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

pages = [st.Page("overview.py", title="City overview", icon=":material/insights:", default=True),
         st.Page("technical.py", title="Technical view", icon=":material/tune:")]
# One centred, responsive column for the whole app: the header's logo, tabs and menu line up with the content
# (Streamlit's main block: 1400 px at most, 5rem side padding on wide screens, 1rem on phones). The technical
# view's tab shows its icon only (the title stays for the browser tab and screen readers).
st.markdown("""<style>
.block-container { max-width: 1400px; }
div[data-testid="stToolbar"] > div { max-width: 1400px; margin: 0 auto; padding: 0 calc(5rem - 16px); }
@media (max-width: 1100px) { div[data-testid="stToolbar"] > div { padding: 0 calc(3rem - 16px); } }
@media (max-width: 640px) { div[data-testid="stToolbar"] > div { padding: 0; } }
header[data-testid="stHeader"] { background: rgba(244,246,250,.85); backdrop-filter: blur(8px); }
a[data-testid="stTopNavLink"][href$="technical"] > :not(:first-child) { display: none; }
footer { display: none; }
/* every widget on a surface, no solid lines: cards white with a thin soft shadow, controls white with a hairline
   shadow, the selected choice a light tint (st.container(border=True) is the stVerticalBlock with a scroll behaviour) */
:root { --card:#ffffff; --soft:#f6f8fb; --tint:#e6f4f1; --accent:#0f766e;
  --shadow-xs: 0 1px 2px rgba(16,24,40,.06);
  --shadow-sm: 0 1px 2px rgba(16,24,40,.04), 0 2px 8px rgba(16,24,40,.05); }
div[data-testid="stVerticalBlock"][data-test-scroll-behavior] { border: none !important; }
div[data-testid="stVerticalBlock"][data-test-scroll-behavior]:not(:has(div[data-testid="stVerticalBlock"][data-test-scroll-behavior])) {
  background: var(--card); border-radius: 16px; box-shadow: var(--shadow-sm); padding: 18px 20px; }
div[data-testid="stTabs"] [role="tablist"]::after { display: none; }
label[data-testid="stRadioOption"]:not([data-selected="true"]) > div > div:first-child { background: #e3e8ef; }
div[data-testid="stTabs"] { background: var(--card); border-radius: 16px; box-shadow: var(--shadow-sm); padding: 8px 20px 20px; }
div[data-testid="stMetric"] { background: var(--soft); border-radius: 12px; padding: 12px 14px; }
div[data-testid="stDataFrameResizable"] { border: none !important; border-radius: 12px; box-shadow: var(--shadow-xs); }
div[data-rac][role="group"] { border-color: transparent !important; background: var(--card) !important;
  box-shadow: var(--shadow-xs); border-radius: 10px; }
button[data-variant="segmented_control"] { border: none !important; background: var(--card) !important;
  box-shadow: var(--shadow-xs); border-radius: 8px !important; margin-right: 4px; }
button[data-variant="segmented_control"][data-selected="true"] { background: var(--tint) !important; color: var(--accent) !important; }
button[data-testid^="stBaseButton"] { border-color: transparent !important; box-shadow: var(--shadow-xs); }
section[data-testid="stSidebar"] { box-shadow: 1px 0 6px rgba(16,24,40,.04); }
/* wide screens: the whole app, header included, is one flat, square column set off from a slightly deeper page: the
   header's centred strip is its top, the main block (from the header's 60 px down, at least to the window's foot) the rest */
@media (min-width: 1100px) {
  .stApp { background: #e9edf3 !important; }
  header[data-testid="stHeader"] { background: #e9edf3 !important; backdrop-filter: none; }
  div[data-testid="stToolbar"] > div { align-self: stretch; display: flex; align-items: center; margin: 0 auto;
    background: #f5f7fa; }
  .block-container { background: #f5f7fa; margin: 60px auto 0; min-height: calc(100vh - 60px); flex-shrink: 0;
    padding-top: 1.25rem !important; padding-bottom: 2.5rem !important; }
}
/* style-only markdown blocks take no row (they left a gap under the header) */
div[data-testid="stElementContainer"]:has(style) { display: none; }
/* the app menu (below) in the header's right corner, where Streamlit's own menu was */
[data-testid="stMainMenu"] { display: none; }
div[data-testid="stLayoutWrapper"]:has(> .st-key-app_menu) { position: fixed; top: 10px; z-index: 1000001; width: auto;
  right: max(16px, calc((100vw - 1400px) / 2 + 5rem - 16px)); }
.stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) div[data-testid="stLayoutWrapper"]:has(> .st-key-app_menu) {
  right: max(16px, calc((100vw - 300px - 1400px) / 2 + 5rem - 16px)); }
@media (max-width: 1100px) { div[data-testid="stLayoutWrapper"]:has(> .st-key-app_menu) { right: calc(3rem - 16px) !important; } }
@media (max-width: 640px) { div[data-testid="stLayoutWrapper"]:has(> .st-key-app_menu) { right: 16px !important; } }
.st-key-app_menu { background: transparent !important; box-shadow: none !important; padding: 0 !important; }
.st-key-app_menu button { background: transparent !important; box-shadow: none !important; }
.st-key-app_menu button > div > div[aria-hidden="true"] { display: none; }
</style>""", unsafe_allow_html=True)


def _menu() -> None:
    """The app menu: which model answers in Predict (Glimmer on this computer, or Muse Spark through the Meta Model
    API), with where to put the API key. Switching also switches the local server (assist_backend.choose)."""
    from netstudy import assist_backend as ab
    from shared import KEY_HELP

    try:                                                # a key in .streamlit/secrets.toml counts as one set
        if ab.KEY_ENV in st.secrets:
            os.environ.setdefault(ab.KEY_ENV, str(st.secrets[ab.KEY_ENV]))
    except Exception:                                   # no secrets file
        pass
    ss = st.session_state
    ss.setdefault("app_backend", ab.current())

    def switch():
        ss["app_backend_msg"] = ab.choose(ss["app_backend"])
        ss.pop("p_answer", None)                        # the last answer came from the other model

    with st.container(key="app_menu"), st.popover("", icon=":material/more_vert:"):
        st.markdown("**Traffic Assistant model**")
        st.radio("Traffic Assistant model", list(ab.CHOICES), format_func=ab.CHOICES.get, key="app_backend",
                 on_change=switch, label_visibility="collapsed")
        if ss.get("app_backend_msg"):
            st.warning(ss["app_backend_msg"])
        if ss["app_backend"] == "muse_spark":
            if ab.api_key():
                st.caption(f"The production model. Key found ({ab.KEY_ENV}) · model `{ab.API_MODEL}` · questions and "
                           "results go to Meta.")
            else:
                st.info(KEY_HELP)
        else:
            st.caption("Low-cost development option: Muse Glimmer on this computer's GPU, no API cost. Start or "
                       "stop it on the Forecast page.")


_menu()
st.navigation(pages, position="top").run()
