"""
app.py — ChurnFlow  ·  Premium 3D Dashboard
Inspired by threedimensions.webflow.io
"""

import sys, json, pathlib, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

warnings.filterwarnings("ignore")

APP_DIR  = pathlib.Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
sys.path.insert(0, str(BASE_DIR / "src"))

ART_DIR   = BASE_DIR / "artifacts"
SHAP_DIR  = ART_DIR  / "shap"
MODEL_DIR = BASE_DIR / "models"
PROC_DIR  = BASE_DIR / "data" / "processed"
RES_JSON  = ART_DIR  / "results.json"

st.set_page_config(
    page_title="ChurnFlow · 3D Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Outfit:wght@200;300;400;500;700;900&display=swap');

:root {
  --bg:      #04040a;
  --bg2:     #08081200;
  --surface: rgba(255,255,255,0.025);
  --border:  rgba(255,255,255,0.065);
  --a1:      #6e56ff;
  --a2:      #00f5d4;
  --a3:      #ff5f57;
  --a4:      #ffbe0b;
  --txt:     #eeeef5;
  --muted:   rgba(238,238,245,0.42);
  --ff-disp: 'Syne', sans-serif;
  --ff-mono: 'DM Mono', monospace;
  --ff-body: 'Outfit', sans-serif;
}

*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
html, body { background:var(--bg)!important; color:var(--txt); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarCollapsedControl"],
.stDeployButton, [data-testid="stSidebar"]
{ display:none!important; }

[data-testid="stAppViewContainer"] { background:var(--bg)!important; }
[data-testid="stMain"] { background:transparent!important; padding:0!important; }
[data-testid="block-container"] { padding: 0!important; max-width:100%!important; }

/* ── Cursor ── */
#cdot { position:fixed;top:0;left:0;z-index:99999;width:7px;height:7px;
  border-radius:50%;background:var(--a2);pointer-events:none;
  transform:translate(-50%,-50%);mix-blend-mode:difference; }
#cring { position:fixed;top:0;left:0;z-index:99998;width:34px;height:34px;
  border-radius:50%;border:1.5px solid rgba(110,86,255,0.55);pointer-events:none;
  transform:translate(-50%,-50%); }

/* ── Grain ── */
#grain { position:fixed;inset:0;z-index:9997;pointer-events:none;opacity:0.026;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size:128px; }

/* ── Ambient BG ── */
#amb { position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 70% 55% at 12% 18%, rgba(110,86,255,0.11) 0%,transparent 65%),
    radial-gradient(ellipse 55% 45% at 88% 82%, rgba(0,245,212,0.07) 0%,transparent 65%),
    radial-gradient(ellipse 45% 35% at 55% 48%, rgba(255,95,87,0.045) 0%,transparent 60%),
    var(--bg); }

/* ── Spline hero bg ── */
#spline { position:fixed;inset:0;z-index:1;pointer-events:none;opacity:0.45; }
#spline iframe { width:100%;height:100%;border:none;
  filter:hue-rotate(20deg) saturate(0.75) brightness(0.9); }

/* ── Top Nav ── */
#nav { position:fixed;top:0;left:0;right:0;z-index:1000;height:62px;
  display:flex;align-items:center;justify-content:space-between;padding:0 48px;
  background:rgba(4,4,10,0.72);backdrop-filter:blur(28px);
  border-bottom:1px solid var(--border); }
.nav-logo { font-family:var(--ff-disp);font-size:19px;font-weight:800;
  letter-spacing:-0.5px;
  background:linear-gradient(135deg,var(--a1),var(--a2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.nav-links { display:flex;gap:30px;list-style:none; }
.nav-links a { font-family:var(--ff-mono);font-size:11px;color:var(--muted);
  text-decoration:none;letter-spacing:0.8px;text-transform:uppercase;
  transition:color 0.2s; }
.nav-links a:hover { color:var(--txt); }
.nav-pill { font-family:var(--ff-mono);font-size:10px;color:var(--a2);
  border:1px solid rgba(0,245,212,0.4);padding:4px 12px;border-radius:20px;
  letter-spacing:1.2px; }

/* ── Content layer ── */
#wrap { position:relative;z-index:10; }

/* ── Hero ── */
#hero { min-height:100vh;display:flex;flex-direction:column;
  justify-content:center;align-items:flex-start;padding:120px 60px 60px; }
.h-eye { font-family:var(--ff-mono);font-size:11px;color:var(--a2);
  letter-spacing:3.5px;text-transform:uppercase;margin-bottom:22px;
  display:flex;align-items:center;gap:14px;opacity:0; }
.h-eye::before { content:'';display:block;width:28px;height:1px;background:var(--a2); }
.h-t1 { font-family:var(--ff-disp);font-size:clamp(54px,8vw,102px);font-weight:800;
  line-height:0.97;letter-spacing:-4px;color:var(--txt);display:block;opacity:0; }
.h-t2 { font-family:var(--ff-disp);font-size:clamp(54px,8vw,102px);font-weight:800;
  line-height:0.97;letter-spacing:-4px;display:block;opacity:0;
  background:linear-gradient(135deg,var(--a1) 20%,var(--a2) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.h-sub { font-family:var(--ff-body);font-size:17px;color:var(--muted);
  max-width:480px;line-height:1.75;margin:28px 0 44px;font-weight:300;opacity:0; }
.cta-row { display:flex;gap:14px;flex-wrap:wrap;opacity:0; }
.btn-p { display:inline-flex;align-items:center;gap:8px;padding:13px 30px;
  border-radius:3px;background:var(--a1);color:#fff;font-family:var(--ff-disp);
  font-weight:700;font-size:13px;letter-spacing:0.4px;
  border:none;cursor:pointer;text-decoration:none;
  transition:all 0.3s;position:relative;overflow:hidden; }
.btn-p:hover { transform:translateY(-2px);
  box-shadow:0 14px 44px rgba(110,86,255,0.42); }
.btn-g { display:inline-flex;align-items:center;gap:8px;padding:13px 30px;
  border-radius:3px;background:transparent;color:var(--txt);font-family:var(--ff-disp);
  font-weight:600;font-size:13px;cursor:pointer;border:1px solid var(--border);
  text-decoration:none;transition:all 0.3s; }
.btn-g:hover { border-color:rgba(255,255,255,0.22);background:var(--surface); }

/* ── KPI bar ── */
#kpi { display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
  background:var(--border);border:1px solid var(--border);border-radius:10px;
  overflow:hidden;margin:0 60px 72px; }
.kc { background:#07070f;padding:30px 26px;transition:background 0.3s; }
.kc:hover { background:rgba(110,86,255,0.055); }
.kl { font-family:var(--ff-mono);font-size:10px;color:var(--muted);
  letter-spacing:1.8px;text-transform:uppercase;margin-bottom:10px; }
.kv { font-family:var(--ff-disp);font-size:36px;font-weight:800;
  letter-spacing:-1.5px;color:var(--txt);line-height:1; }
.kd { margin-top:9px;font-family:var(--ff-mono);font-size:10px;color:var(--a2); }

/* ── Sections ── */
.sec { padding:72px 60px;position:relative; }
.sec-tag { font-family:var(--ff-mono);font-size:10px;color:var(--a1);
  letter-spacing:2.5px;text-transform:uppercase;margin-bottom:14px;
  display:flex;align-items:center;gap:10px; }
.sec-tag::before { content:'';width:18px;height:1px;background:var(--a1); }
.sec-h { font-family:var(--ff-disp);font-size:clamp(28px,3.5vw,48px);
  font-weight:800;letter-spacing:-2px;margin-bottom:44px;color:var(--txt); }

/* ── Glass card ── */
.gl { background:rgba(255,255,255,0.022);border:1px solid var(--border);
  border-radius:14px;backdrop-filter:blur(18px);padding:26px;
  transition:all 0.38s;position:relative;overflow:hidden; }
.gl::before { content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent); }
.gl:hover { border-color:rgba(110,86,255,0.28);transform:translateY(-3px);
  box-shadow:0 22px 64px rgba(0,0,0,0.55); }

/* ── Model cards ── */
.mgrid { display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:44px; }
.mc { background:#07070f;border:1px solid var(--border);border-radius:10px;
  padding:22px;transition:all 0.32s;position:relative;overflow:hidden; }
.mc::after { content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--a1),var(--a2));
  transform:scaleX(0);transition:transform 0.32s;transform-origin:left; }
.mc:hover,.mc.best { border-color:rgba(110,86,255,0.32); }
.mc:hover::after,.mc.best::after { transform:scaleX(1); }
.mc.best { background:rgba(110,86,255,0.055); }
.mb { font-family:var(--ff-mono);font-size:9px;letter-spacing:1.2px;
  text-transform:uppercase;padding:3px 9px;border-radius:3px;
  display:inline-block;margin-bottom:12px; }
.mb.ch { background:rgba(110,86,255,0.18);color:var(--a1); }
.mb.fn { background:rgba(0,245,212,0.1);color:var(--a2); }
.mn { font-family:var(--ff-disp);font-size:17px;font-weight:700;
  margin-bottom:14px;letter-spacing:-0.3px; }
.mm { display:flex;justify-content:space-between;margin-bottom:7px; }
.mml { font-family:var(--ff-mono);font-size:10px;color:var(--muted); }
.mmv { font-family:var(--ff-mono);font-size:12px;color:var(--txt); }
.mbar-w { height:2px;background:rgba(255,255,255,0.06);border-radius:1px;margin-top:14px; }
.mbar { height:100%;border-radius:1px;width:0;transition:width 1.4s cubic-bezier(.25,.46,.45,.94); }

/* ── Prediction result ── */
.pred-box { border-radius:12px;padding:28px;border:1px solid;margin-top:20px;
  display:grid;grid-template-columns:1fr 1fr 1fr;gap:28px;align-items:center; }
.pred-label { font-family:var(--ff-mono);font-size:10px;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:10px; }
.pred-val { font-family:var(--ff-disp);font-size:48px;font-weight:800;
  letter-spacing:-2.5px;line-height:1; }
.pred-risk { font-family:var(--ff-disp);font-size:28px;font-weight:800;
  letter-spacing:-1px;line-height:1.2; }
.pbar-w { height:3px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:22px; }
.pbar { height:100%;border-radius:2px;transition:width 1.2s ease; }

/* ── Streamlit overrides ── */
[data-testid="stSelectbox"]>div>div,
[data-testid="stNumberInput"]>div>div>input {
  background:rgba(8,8,18,0.9)!important;
  border:1px solid var(--border)!important;border-radius:6px!important;
  color:var(--txt)!important;font-family:var(--ff-mono)!important;font-size:12px!important; }
[data-testid="stSlider"] { padding-top:8px; }
label[data-testid="stWidgetLabel"] p {
  font-family:var(--ff-mono)!important;font-size:10px!important;color:var(--muted)!important;
  letter-spacing:1.5px!important;text-transform:uppercase!important; }
[data-testid="stButton"]>button {
  background:var(--a1)!important;color:#fff!important;border:none!important;
  border-radius:3px!important;padding:12px 26px!important;font-family:var(--ff-disp)!important;
  font-weight:700!important;font-size:13px!important;letter-spacing:0.5px!important;
  transition:all 0.3s!important; }
[data-testid="stButton"]>button:hover {
  transform:translateY(-2px)!important;
  box-shadow:0 12px 40px rgba(110,86,255,0.44)!important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background:transparent!important;border-bottom:1px solid var(--border)!important; }
[data-testid="stTabs"] [data-baseweb="tab"] {
  background:transparent!important;color:var(--muted)!important;
  font-family:var(--ff-mono)!important;font-size:10px!important;
  letter-spacing:1.5px!important;text-transform:uppercase!important;
  border:none!important;border-radius:0!important; }
[data-testid="stTabs"] [aria-selected="true"] {
  color:var(--txt)!important;border-bottom:2px solid var(--a1)!important; }
[data-testid="stImage"] img { border-radius:10px; }

/* Scrollbar */
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--a1);border-radius:2px; }

/* Page veil */
#veil { position:fixed;inset:0;z-index:99999;background:var(--bg);pointer-events:none; }
</style>
""", unsafe_allow_html=True)

# ── Fixed layers + GSAP ───────────────────────────────────────────────────────
st.markdown("""
<div id="veil"></div>
<div id="cdot"></div>
<div id="cring"></div>
<div id="grain"></div>
<div id="amb"></div>

<div id="spline">
  <iframe src="https://my.spline.design/interactiveai-mcalKaDv3MXFynmF2gLFnqUt/"
    frameborder="0" width="100%" height="100%" loading="lazy"></iframe>
</div>

<nav id="nav">
  <div class="nav-logo">⬡ ChurnFlow</div>
  <ul class="nav-links">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#models">Models</a></li>
    <li><a href="#predict">Predict</a></li>
    <li><a href="#shap">SHAP</a></li>
  </ul>
  <div class="nav-pill">LIVE · v2.0</div>
</nav>

<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>

<script>
gsap.registerPlugin(ScrollTrigger);

// Page veil lift
gsap.to('#veil',{opacity:0,duration:1.1,ease:'power3.inOut',
  onComplete:()=>{ document.getElementById('veil').style.display='none'; }});

// Nav
gsap.from('#nav',{y:-70,opacity:0,duration:0.9,delay:0.7,ease:'power3.out'});

// Cursor
const dot=document.getElementById('cdot'), ring=document.getElementById('cring');
document.addEventListener('mousemove',e=>{
  gsap.to(dot, {x:e.clientX,y:e.clientY,duration:0.08});
  gsap.to(ring,{x:e.clientX,y:e.clientY,duration:0.22,ease:'power2.out'});
});
document.addEventListener('mousedown',()=>gsap.to(ring,{scale:0.7,duration:0.15}));
document.addEventListener('mouseup',  ()=>gsap.to(ring,{scale:1.0,duration:0.25}));

// Hero stagger
const heroTl = gsap.timeline({delay:1.0});
heroTl
  .to('.h-eye',{opacity:1,y:0,duration:0.7,ease:'power3.out'})
  .to('.h-t1', {opacity:1,y:0,duration:0.8,ease:'power3.out'},'-=0.3')
  .to('.h-t2', {opacity:1,y:0,duration:0.8,ease:'power3.out'},'-=0.5')
  .to('.h-sub', {opacity:1,y:0,duration:0.8,ease:'power3.out'},'-=0.4')
  .to('.cta-row',{opacity:1,y:0,duration:0.7,ease:'power3.out'},'-=0.4');

// KPI counter animation
function runCounters(){
  document.querySelectorAll('[data-count]').forEach(el=>{
    const target=parseFloat(el.dataset.count);
    const dec=parseInt(el.dataset.dec||4);
    let v=0; const step=target/90;
    const tick=()=>{ v=Math.min(v+step,target);
      el.textContent=v.toFixed(dec);
      if(v<target) requestAnimationFrame(tick); };
    tick();
  });
}
setTimeout(runCounters,1500);

// Scroll reveals
setTimeout(()=>{
  gsap.utils.toArray('.rv').forEach((el,i)=>{
    gsap.fromTo(el,{opacity:0,y:44},
      {opacity:1,y:0,duration:0.85,ease:'power3.out',delay:i*0.07,
       scrollTrigger:{trigger:el,start:'top 90%'}});
  });
  gsap.utils.toArray('.rv-l').forEach(el=>{
    gsap.fromTo(el,{opacity:0,x:-44},
      {opacity:1,x:0,duration:0.85,ease:'power3.out',
       scrollTrigger:{trigger:el,start:'top 90%'}});
  });
},900);

// Animate model bars
setTimeout(()=>{
  document.querySelectorAll('.mbar[data-w]').forEach(el=>{
    el.style.width=el.dataset.w;
  });
},1800);

// Parallax ambient on mouse
document.addEventListener('mousemove',e=>{
  const xp=(e.clientX/window.innerWidth-0.5)*14;
  const yp=(e.clientY/window.innerHeight-0.5)*10;
  gsap.to('#amb',{x:xp,y:yp,duration:2.5,ease:'power1.out'});
});
</script>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    p = MODEL_DIR / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None

@st.cache_resource
def load_preprocessor():
    p = MODEL_DIR / "preprocessor.joblib"
    return joblib.load(p) if p.exists() else None

@st.cache_data
def load_results():
    if RES_JSON.exists():
        with open(RES_JSON) as f: return json.load(f)
    return {}

def dark_plotly(fig, h=420):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eeeef5", family="DM Mono, monospace"),
        margin=dict(t=16,b=16,l=16,r=16),
        legend=dict(bgcolor="rgba(7,7,15,0.85)",bordercolor="rgba(255,255,255,0.07)",borderwidth=1)
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)",zeroline=False,tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)",zeroline=False,tickfont=dict(size=10))
    return fig

COLORS = ["#6e56ff","#00f5d4","#ff5f57","#ffbe0b"]
MODEL_NAMES = ["Logistic_Regression","Random_Forest","XGBoost","LightGBM"]
results = load_results()

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div id="overview" style="padding-top:64px"></div>
<div id="wrap">
<div id="hero">
  <div class="h-eye">ML Pipeline · Telecom Intelligence</div>
  <h1>
    <span class="h-t1">Predict.</span>
    <span class="h-t2">Prevent. Retain.</span>
  </h1>
  <p class="h-sub">
    End-to-end churn intelligence — 4 ML models, MLflow experiment
    tracking, SHAP explainability and real-time scoring. Built for
    production.
  </p>
  <div class="cta-row">
    <a class="btn-p" href="#predict">→ Run Prediction</a>
    <a class="btn-g" href="#models">View Leaderboard</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
if results:
    best = max(results, key=lambda k: results[k]["cv_auc_mean"])
    br   = results[best]
    st.markdown(f"""
    <div id="kpi" class="rv">
      <div class="kc">
        <div class="kl">Champion Model</div>
        <div class="kv" style="font-size:22px;letter-spacing:-0.5px">{best.replace('_',' ')}</div>
        <div class="kd">↑ Best by CV-AUC</div>
      </div>
      <div class="kc">
        <div class="kl">AUC-ROC</div>
        <div class="kv"><span data-count="{br['cv_auc_mean']}" data-dec="4">—</span></div>
        <div class="kd">± {br['cv_auc_std']:.4f} · 5-fold stratified</div>
      </div>
      <div class="kc">
        <div class="kl">F1 Score</div>
        <div class="kv"><span data-count="{br['cv_f1_mean']}" data-dec="4">—</span></div>
        <div class="kd">Precision {br['cv_precision_mean']:.3f} · Recall {br['cv_recall_mean']:.3f}</div>
      </div>
      <div class="kc">
        <div class="kl">Customers Analysed</div>
        <div class="kv">7,043</div>
        <div class="kd">IBM Telco · 26.5% churn rate</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="models"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec" style="border-top:1px solid rgba(255,255,255,0.045)">
  <div class="sec-tag rv">Experiment Tracking</div>
  <div class="sec-h  rv">Model Leaderboard</div>
""", unsafe_allow_html=True)

if results:
    order = sorted(results, key=lambda k: results[k]["cv_auc_mean"], reverse=True)
    best_n = order[0]
    html  = '<div class="mgrid">'
    for i,mn in enumerate(order):
        r  = results[mn]
        is_best = mn==best_n
        badge = '<span class="mb ch">🏆 Champion</span>' if is_best else '<span class="mb fn">◆ Finalist</span>'
        c  = COLORS[i % len(COLORS)]
        w  = f"{r['cv_auc_mean']*100:.0f}%"
        html += f"""
        <div class="mc {'best' if is_best else ''} rv">
          {badge}
          <div class="mn">{mn.replace('_',' ')}</div>
          <div class="mm"><span class="mml">AUC-ROC</span><span class="mmv">{r['cv_auc_mean']:.4f}</span></div>
          <div class="mm"><span class="mml">F1</span>    <span class="mmv">{r['cv_f1_mean']:.4f}</span></div>
          <div class="mm"><span class="mml">Acc</span>   <span class="mmv">{r['cv_accuracy_mean']:.4f}</span></div>
          <div class="mm"><span class="mml">Recall</span><span class="mmv">{r['cv_recall_mean']:.4f}</span></div>
          <div class="mbar-w">
            <div class="mbar" data-w="{w}" style="background:linear-gradient(90deg,{c},{c}66)"></div>
          </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Radar + bar side-by-side
    c1, c2 = st.columns([3,2])
    with c1:
        mkeys  = ["cv_auc_mean","cv_f1_mean","cv_accuracy_mean","cv_precision_mean","cv_recall_mean"]
        mlabels= ["AUC","F1","Accuracy","Precision","Recall"]
        fig_r  = go.Figure()
        for i,mn in enumerate(order):
            r = results[mn]
            vals = [r[k] for k in mkeys]
            rgb  = tuple(int(COLORS[i][j:j+2],16) for j in (1,3,5))
            fig_r.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=mlabels+[mlabels[0]],
                fill="toself", name=mn.replace("_"," "),
                line=dict(color=COLORS[i],width=2),
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.07)"
            ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True,range=[0.5,1.0],
                  gridcolor="rgba(255,255,255,0.055)",
                  tickfont=dict(size=9,color="rgba(238,238,245,0.4)"),
                  tickcolor="transparent"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.055)",
                  tickfont=dict(size=11,color="rgba(238,238,245,0.65)"))
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#eeeef5",size=11,family="DM Mono"),
              bgcolor="rgba(7,7,15,0.7)",bordercolor="rgba(255,255,255,0.07)",borderwidth=1),
            height=440, margin=dict(t=10,b=10)
        )
        st.markdown('<div class="gl rv">', unsafe_allow_html=True)
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        df_r = [{"M":mn.replace("_"," "),"AUC":results[mn]["cv_auc_mean"],
                 "Std":results[mn]["cv_auc_std"]} for mn in order]
        fig_b = go.Figure(go.Bar(
            x=[d["AUC"] for d in df_r],
            y=[d["M"]   for d in df_r],
            orientation="h",
            marker_color=COLORS[:len(df_r)],
            error_x=dict(type="data",array=[d["Std"] for d in df_r],
                         color="rgba(255,255,255,0.25)",thickness=1.2,width=4),
            text=[f"{d['AUC']:.4f}" for d in df_r], textposition="outside",
            textfont=dict(size=10,color="rgba(238,238,245,0.55)",family="DM Mono")
        ))
        fig_b = dark_plotly(fig_b, 420)
        fig_b.update_xaxes(range=[0.80,0.87])
        fig_b.update_yaxes(autorange="reversed")
        st.markdown('<div class="gl rv">', unsafe_allow_html=True)
        st.plotly_chart(fig_b, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="predict"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec" style="border-top:1px solid rgba(255,255,255,0.045)">
  <div class="sec-tag rv">Live Inference</div>
  <div class="sec-h  rv">Single Prediction</div>
""", unsafe_allow_html=True)

preprocessor = load_preprocessor()
if not results:
    st.warning("Run `python src/train.py` first.")
else:
    mc_col, _ = st.columns([2,4])
    with mc_col:
        model_choice = st.selectbox("MODEL",
            [m.replace("_"," ") for m in MODEL_NAMES if (MODEL_DIR/f"{m}.joblib").exists()])
    model = load_model(model_choice.replace(" ","_"))

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="gl rv">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--ff-mono);font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">ACCOUNT INFO</div>', unsafe_allow_html=True)
        gender    = st.selectbox("Gender",         ["Male","Female"])
        senior    = st.selectbox("Senior Citizen", ["No","Yes"])
        partner   = st.selectbox("Partner",        ["Yes","No"])
        depend    = st.selectbox("Dependents",     ["No","Yes"])
        tenure    = st.slider("Tenure (months)",   0,72,12)
        contract  = st.selectbox("Contract",       ["Month-to-month","One year","Two year"])
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="gl rv">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--ff-mono);font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">SERVICES</div>', unsafe_allow_html=True)
        phone     = st.selectbox("Phone Service",     ["Yes","No"])
        multi_l   = st.selectbox("Multiple Lines",    ["No","Yes"])
        internet  = st.selectbox("Internet Service",  ["Fiber optic","DSL","No"])
        online_s  = st.selectbox("Online Security",   ["No","Yes"])
        online_b  = st.selectbox("Online Backup",     ["No","Yes"])
        device    = st.selectbox("Device Protection", ["No","Yes"])
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="gl rv">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--ff-mono);font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">BILLING</div>', unsafe_allow_html=True)
        tech_s    = st.selectbox("Tech Support",      ["No","Yes"])
        stv       = st.selectbox("Streaming TV",      ["No","Yes"])
        smov      = st.selectbox("Streaming Movies",  ["No","Yes"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        payment   = st.selectbox("Payment Method",    [
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"])
        monthly_c = st.number_input("Monthly Charges ($)", 18.0,120.0,65.0,0.5)
        total_c   = st.number_input("Total Charges ($)",   0.0,9000.0,float(monthly_c*tenure),1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("→ PREDICT CHURN"):
        raw = pd.DataFrame([{
            "gender":gender,"SeniorCitizen":1 if senior=="Yes" else 0,
            "Partner":partner,"Dependents":depend,"tenure":tenure,
            "PhoneService":phone,"MultipleLines":multi_l,
            "InternetService":internet,"OnlineSecurity":online_s,
            "OnlineBackup":online_b,"DeviceProtection":device,
            "TechSupport":tech_s,"StreamingTV":stv,
            "StreamingMovies":smov,"Contract":contract,
            "PaperlessBilling":paperless,"PaymentMethod":payment,
            "MonthlyCharges":monthly_c,"TotalCharges":total_c,
        }])
        raw["tenure_group"]     = pd.cut(raw["tenure"],bins=[0,12,24,48,60,np.inf],
            labels=["0-1yr","1-2yr","2-4yr","4-5yr","5+yr"]).astype(str)
        raw["revenue_per_month"]= raw["TotalCharges"]/(raw["tenure"]+1)
        raw["high_value"]       = (raw["MonthlyCharges"]>65).astype(int)
        svc = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        raw["num_services"]     = raw[svc].apply(lambda r:(r=="Yes").sum(),axis=1)

        if preprocessor:
            X_in = preprocessor.transform(raw)
            prob = model.predict_proba(X_in)[0][1]
            risk = "HIGH RISK" if prob>=0.7 else "MEDIUM RISK" if prob>=0.4 else "LOW RISK"
            col  = "#ff5f57" if prob>=0.7 else "#ffbe0b" if prob>=0.4 else "#00f5d4"

            st.markdown(f"""
            <div class="pred-box rv" style="border-color:{col}44;background:rgba(0,0,0,0.3)">
              <div>
                <div class="pred-label">Risk Assessment</div>
                <div class="pred-risk" style="color:{col}">{risk}</div>
              </div>
              <div>
                <div class="pred-label">Churn Probability</div>
                <div class="pred-val" style="color:{col}">{prob:.1%}</div>
              </div>
              <div>
                <div class="pred-label">Retention Score</div>
                <div class="pred-val" style="color:var(--a2)">{1-prob:.1%}</div>
              </div>
              <div style="grid-column:1/-1">
                <div class="pbar-w">
                  <div class="pbar" style="width:{prob*100:.1f}%;
                    background:linear-gradient(90deg,{col},{col}66)"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number={"suffix":"%","font":{"size":34,"color":col,"family":"Syne"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"rgba(255,255,255,0.15)",
                            "tickfont":{"size":9,"color":"rgba(255,255,255,0.3)"}},
                    "bar":{"color":col,"thickness":0.45},
                    "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                    "steps":[
                        {"range":[0,40], "color":"rgba(0,245,212,0.07)"},
                        {"range":[40,70],"color":"rgba(255,190,11,0.07)"},
                        {"range":[70,100],"color":"rgba(255,95,87,0.07)"},
                    ],
                    "threshold":{"line":{"color":"rgba(255,255,255,0.25)","width":2},"value":50}
                }
            ))
            fig_g.update_layout(height=250,paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#eeeef5"),margin=dict(t=20,b=10,l=30,r=30))
            st.plotly_chart(fig_g, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SHAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="shap"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec" style="border-top:1px solid rgba(255,255,255,0.045)">
  <div class="sec-tag rv">Explainable AI</div>
  <div class="sec-h  rv">SHAP Analysis</div>
""", unsafe_allow_html=True)

avail = [m for m in MODEL_NAMES if (SHAP_DIR/f"shap_summary_{m}.png").exists()]
if not avail:
    st.warning("Run `python src/shap_report.py` to generate SHAP plots.")
else:
    sh_col,_ = st.columns([2,4])
    with sh_col:
        shap_m = st.selectbox("MODEL", [m.replace("_"," ") for m in avail], key="shap_s")
    mk = shap_m.replace(" ","_")
    t1,t2,t3 = st.tabs(["BEESWARM","IMPORTANCE","WATERFALL"])
    with t1:
        img = SHAP_DIR/f"shap_summary_{mk}.png"
        if img.exists():
            st.markdown('<div class="gl rv">', unsafe_allow_html=True)
            st.image(str(img), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with t2:
        csv_p = SHAP_DIR/f"shap_values_{mk}.csv"
        if csv_p.exists():
            sdf   = pd.read_csv(csv_p)
            top20 = sdf.abs().mean().sort_values(ascending=True).tail(20)
            rgb0  = (110,86,255); rgb1=(0,245,212)
            fig_sh= go.Figure(go.Bar(
                x=top20.values, y=top20.index, orientation="h",
                marker=dict(
                    color=top20.values,
                    colorscale=[[0,f"rgb{rgb0}"],[1,f"rgb{rgb1}"]],
                    line=dict(width=0)
                ),
                text=[f"{v:.4f}" for v in top20.values], textposition="outside",
                textfont=dict(size=9,color="rgba(238,238,245,0.5)",family="DM Mono")
            ))
            fig_sh = dark_plotly(fig_sh,560)
            fig_sh.update_coloraxes(showscale=False)
            st.markdown('<div class="gl rv">', unsafe_allow_html=True)
            st.plotly_chart(fig_sh, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with t3:
        wf = SHAP_DIR/f"shap_waterfall_{mk}.png"
        if wf.exists():
            st.markdown('<div class="gl rv">', unsafe_allow_html=True)
            st.image(str(wf), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Waterfall not available for this model.")

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.045);
  padding:36px 60px;display:flex;justify-content:space-between;align-items:center;
  margin-top:20px">
  <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;
    background:linear-gradient(135deg,#6e56ff,#00f5d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    ⬡ ChurnFlow
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;
    color:rgba(238,238,245,0.28);letter-spacing:1.2px">
    BUILT BY NIKHIL YADAV · BIT DURG · CSE DATA SCIENCE
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;
    color:rgba(238,238,245,0.28);letter-spacing:1px">
    MLflow 2.12.1 · SHAP 0.45 · Streamlit 1.33
  </div>
</div>
</div>
""", unsafe_allow_html=True)