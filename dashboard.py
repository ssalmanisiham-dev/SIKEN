"""
🛡️ CyberShield AI_SIHAM — Real-Time Multi-Model Threat Dashboard
Models: Gradient Boosting · Isolation Forest · K-Means · Random Forest

Project structure:
  AI_SIHAM/
    data/test.csv, train.csv
    Gradient_Boosting/gradient_boosting.py, decision_tree.py, main.py
    Isolation_forest/isolation_forest.py, isolation_tree.py, main.py
    K-means/kmeans.py, main.py
    RandomForest_scratch/random_forest.py, decision_tree.py, main.py
    dashboard.py  ← THIS FILE
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
st.set_page_config(
    page_title="CyberShield · AI_SIHAM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import time, os, sys, math, random
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── try real models ───────────────────────────────────────────────────────────
HAVE_IF = HAVE_GB = HAVE_KM = HAVE_RF = False

try:
    from Isolation_forest.isolation_forest import isolation_forest, anomaly_score
    HAVE_IF = True
except Exception: pass

try:
    from Gradient_Boosting.gradient_boosting import GradientBoosting
    HAVE_GB = True
except Exception: pass

try:
    from K_means.kmeans import KMeans
    HAVE_KM = True
except Exception: pass

try:
    from RandomForest_scratch.random_forest import RandomForest
    HAVE_RF = True
except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
# 2. CSS — dark-ops terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    background: #080b0a !important;
    color: #c5d8c0 !important;
}
h1 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #39ff14 !important;
    letter-spacing: 3px;
    text-shadow: 0 0 20px #39ff1466;
    margin-bottom: 0 !important;
}
h2,h3,h4 { color: #8fba84 !important; letter-spacing:1px; }

[data-testid="metric-container"] {
    background:#0d150f; border:1px solid #1e3d1e;
    border-left:3px solid #39ff14; border-radius:4px;
    padding:10px 14px; box-shadow:0 0 14px #39ff1418;
}
[data-testid="metric-container"] label {
    color:#5a8a5a !important; font-size:0.70rem;
    letter-spacing:1.5px; text-transform:uppercase;
}
[data-testid="stMetricValue"] {
    color:#39ff14 !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:1.25rem !important;
}
[data-testid="stMetricDelta"] { color:#ffaa00 !important; }

.sev-CRITICAL { background:#220808; border-left:4px solid #ff2222; padding:7px 13px;
    border-radius:3px; margin:3px 0; font-family:'Share Tech Mono',monospace;
    font-size:0.78rem; color:#ff8080; line-height:1.6; }
.sev-HIGH     { background:#221208; border-left:4px solid #ff7700; padding:7px 13px;
    border-radius:3px; margin:3px 0; font-family:'Share Tech Mono',monospace;
    font-size:0.78rem; color:#ffb060; line-height:1.6; }
.sev-MEDIUM   { background:#1e1e08; border-left:4px solid #ffdd00; padding:7px 13px;
    border-radius:3px; margin:3px 0; font-family:'Share Tech Mono',monospace;
    font-size:0.78rem; color:#ffee80; line-height:1.6; }
.sev-INFO     { background:#091608; border-left:4px solid #39ff14; padding:7px 13px;
    border-radius:3px; margin:3px 0; font-family:'Share Tech Mono',monospace;
    font-size:0.78rem; color:#80cc80; line-height:1.6; }

/* model badges */
.model-badge {
    display:inline-block; font-size:10px; padding:2px 8px; border-radius:3px;
    font-family:'Share Tech Mono',monospace; margin:0 3px;
}
.mb-gb  { background:#0d2137; color:#58a6ff; border:1px solid #185FA530; }
.mb-if  { background:#1a2200; color:#aaff55; border:1px solid #55aa0030; }
.mb-km  { background:#22001a; color:#ff55cc; border:1px solid #aa005530; }
.mb-rf  { background:#1a1100; color:#ffaa00; border:1px solid #aa770030; }

[data-testid="stSidebar"] {
    background:#060d07 !important;
    border-right:1px solid #163316;
}
[data-testid="stSidebar"] .stButton > button {
    width:100%; background:#0d1a0d; color:#39ff14;
    border:1px solid #39ff14; border-radius:2px;
    font-family:'Share Tech Mono',monospace;
    letter-spacing:1px; transition:all .18s; margin-bottom:4px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background:#39ff14; color:#060d07; box-shadow:0 0 14px #39ff1488;
}
[data-testid="stDataFrame"] { border:1px solid #163316 !important; }
thead tr th {
    background:#0d1a0d !important; color:#39ff14 !important;
    font-family:'Share Tech Mono',monospace; font-size:0.70rem; letter-spacing:0.8px;
}
tbody tr:nth-child(even) { background:#0a110a !important; }
tbody tr:hover           { background:#122012 !important; }
hr { border-color:#163316 !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#080b0a; }
::-webkit-scrollbar-thumb { background:#39ff14; border-radius:2px; }
.stTabs [data-baseweb="tab-list"]  { background:#0d150f; border-bottom:1px solid #1e3d1e; }
.stTabs [data-baseweb="tab"]       { color:#5a8a5a; font-family:'Share Tech Mono',monospace; }
.stTabs [aria-selected="true"]     { color:#39ff14 !important; border-bottom:2px solid #39ff14 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Bwd Packet Length Max", "Packet Length Mean", "Flow IAT Std",
    "Active Min", "Active Mean", "Flow Duration", "Flow IAT Mean",
    "Fwd IAT Mean", "Bwd Packet Length Min", "ACK Flag Count",
    "FIN Flag Count", "Active Max", "Flow IAT Min", "Min Packet Length",
    "Bwd IAT Mean", "Bwd IAT Min", "PSH Flag Count", "Active Std",
    "Bwd IAT Std", "Fwd IAT Min",
]
LABEL_COL = "Label"
SEV_ICON  = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","INFO":"🟢"}

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    candidates = [
        os.path.join(BASE_DIR, "data", "test.csv"),
        os.path.join(BASE_DIR, "data", "train.csv"),
        os.path.join(BASE_DIR, "test.csv"),
        os.path.join(BASE_DIR, "train.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            for c in FEATURE_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if LABEL_COL in df.columns:
                df["Is_Attack"]  = (df[LABEL_COL] != 0).astype(int)
                df["Attack_Type"] = df[LABEL_COL].astype(str)
            else:
                df["Is_Attack"]   = 0
                df["Attack_Type"] = "UNKNOWN"
            rng = np.random.default_rng(42)
            n   = len(df)
            df["Source IP"] = [f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}" for _ in range(n)]
            df["Dest IP"]   = [f"192.168.{rng.integers(0,3)}.{rng.integers(1,254)}" for _ in range(n)]
            df["Protocol"]  = rng.choice(["TCP","UDP","ICMP","HTTP","HTTPS","DNS"], size=n,
                                         p=[0.35,0.25,0.10,0.15,0.10,0.05])
            df["Timestamp"] = pd.date_range("2025-01-01", periods=n, freq="500ms").astype(str)
            return df.dropna(subset=["Packet Length Mean"]).reset_index(drop=True)
    # ── Fallback: generate synthetic data ─────────────────────────────────────
    st.warning("⚠️ No CSV found — running on synthetic demo data.")
    return _generate_synthetic(500)

def _generate_synthetic(n=500):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        lbl = int(rng.choice([0,0,0,2,2,2,2,1,3,4]))
        pkt = (700+rng.random()*600) if lbl==2 else (300+rng.random()*200) if lbl==1 else rng.random()*300
        rows.append({
            "Bwd Packet Length Max" : pkt*(0.8+rng.random()*.4),
            "Packet Length Mean"    : pkt,
            "Flow IAT Std"          : rng.random()*50000,
            "Active Min"            : rng.random()*100,
            "Active Mean"           : rng.random()*200,
            "Flow Duration"         : rng.random()*5e6,
            "Flow IAT Mean"         : rng.random()*100000,
            "Fwd IAT Mean"          : rng.random()*50000,
            "Bwd Packet Length Min" : rng.random()*100,
            "ACK Flag Count"        : int(rng.integers(0,6)),
            "FIN Flag Count"        : int(rng.integers(0,2)),
            "Active Max"            : rng.random()*500,
            "Flow IAT Min"          : int(rng.integers(0,10)),
            "Min Packet Length"     : rng.random()*50,
            "Bwd IAT Mean"          : rng.random()*50000,
            "Bwd IAT Min"           : int(rng.integers(0,10)),
            "PSH Flag Count"        : int(rng.integers(0,4)),
            "Active Std"            : rng.random()*100,
            "Bwd IAT Std"           : rng.random()*50000,
            "Fwd IAT Min"           : int(rng.integers(0,10)),
            "Is_Attack"             : int(lbl != 0),
            "Attack_Type"           : str(lbl),
            "Source IP"             : f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}",
            "Dest IP"               : f"192.168.{rng.integers(0,3)}.{rng.integers(1,254)}",
            "Protocol"              : rng.choice(["TCP","UDP","ICMP","HTTP","HTTPS","DNS"]),
            "Timestamp"             : str(datetime(2025,1,1) if i==0 else ""),
        })
    return pd.DataFrame(rows).reset_index(drop=True)

dataset    = load_dataset()
total_rows = len(dataset)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SCRATCH MODELS (built-in fallbacks — used when real modules absent)
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a. K-Means from scratch ──────────────────────────────────────────────────
class _KMeansScratch:
    def __init__(self, k=4, max_iter=100):
        self.k, self.max_iter = k, max_iter
        self.centroids = None

    def fit(self, X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()
        for _ in range(self.max_iter):
            labels = self._assign(X)
            new_c  = np.array([X[labels==j].mean(axis=0) if (labels==j).any()
                                else self.centroids[j] for j in range(self.k)])
            if np.allclose(new_c, self.centroids): break
            self.centroids = new_c
        return self

    def _assign(self, X):
        dists = np.linalg.norm(X[:,None] - self.centroids[None,:], axis=2)
        return np.argmin(dists, axis=1)

    def predict(self, X):
        return self._assign(X)

    def anomaly_score(self, x):
        """Distance to nearest centroid, normalised to [0,1]."""
        dists = np.linalg.norm(x - self.centroids, axis=1)
        d = float(dists.min())
        return min(d / (d + 200 + 1e-9), 0.99)

# ── 5b. Random Forest from scratch ───────────────────────────────────────────
class _DTNode:
    __slots__ = ('feat','thr','left','right','val')
    def __init__(self): self.feat=self.thr=self.left=self.right=self.val=None

class _DecisionTreeScratch:
    def __init__(self, max_depth=6, min_samples=2):
        self.max_depth=max_depth; self.min_samples=min_samples; self.root=None

    def fit(self, X, y):
        self.root = self._build(X, y, 0); return self

    def _build(self, X, y, depth):
        node = _DTNode()
        if depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1:
            node.val = float(np.mean(y)); return node
        feat, thr, score = None, None, float('inf')
        for f in range(X.shape[1]):
            vals = np.unique(X[:,f])
            for t in vals[:-1]:
                lm, rm = y[X[:,f]<=t], y[X[:,f]>t]
                if len(lm)==0 or len(rm)==0: continue
                s = len(lm)*np.std(lm) + len(rm)*np.std(rm)
                if s < score: score,feat,thr = s,f,t
        if feat is None:
            node.val = float(np.mean(y)); return node
        node.feat,node.thr = feat,thr
        node.left  = self._build(X[X[:,feat]<=thr], y[X[:,feat]<=thr], depth+1)
        node.right = self._build(X[X[:,feat]>thr],  y[X[:,feat]>thr],  depth+1)
        return node

    def predict_one(self, x):
        n = self.root
        while n.val is None:
            n = n.left if x[n.feat] <= n.thr else n.right
        return n.val

class _RandomForestScratch:
    def __init__(self, n_trees=15, max_depth=5, max_features=6):
        self.n_trees=n_trees; self.max_depth=max_depth; self.max_features=max_features
        self.trees=[]; self.feat_subsets=[]

    def fit(self, X, y):
        rng=np.random.default_rng(42)
        n,p=X.shape
        for _ in range(self.n_trees):
            idxs=rng.integers(0,n,n)
            feats=rng.choice(p, min(self.max_features,p), replace=False)
            self.feat_subsets.append(feats)
            t=_DecisionTreeScratch(max_depth=self.max_depth)
            t.fit(X[idxs][:,feats], y[idxs])
            self.trees.append(t)
        return self

    def predict_proba_one(self, x):
        preds=[t.predict_one(x[fs]) for t,fs in zip(self.trees,self.feat_subsets)]
        return float(np.mean(preds))

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
X_train = dataset[FEATURE_COLS].fillna(0).values
y_train = dataset["Is_Attack"].values

@st.cache_resource
def train_all(_X, _y):

    models = {}

    # Train on subset for speed
    sample_size = min(10000, len(_X))

    idx = np.random.choice(
        len(_X),
        sample_size,
        replace=False
    )

    X_small = _X[idx]
    y_small = _y[idx]

    print(f"Training on {len(X_small)} samples...")

    # ── Gradient Boosting ─────────────────────────────────────
    if HAVE_GB:
        print("Training GB...")
        from Gradient_Boosting.gradient_boosting import GradientBoosting

        models['gb'] = GradientBoosting(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4
        )

        models['gb'].fit(X_small, y_small)
        print("GB done")
    else:
        models['gb'] = None

    # ── Isolation Forest ─────────────────────────────────────
    if HAVE_IF:
        print("Training IF...")

        from Isolation_forest.isolation_forest import isolation_forest as _if

        _df = pd.DataFrame(
            X_small,
            columns=[f"f{i}" for i in range(X_small.shape[1])]
        )

        models['if'] = _if(
            _df,
            n_trees=20,
            max_depth=10,
            subspace=128
        )

        print("IF done")
    else:
        models['if'] = None

    # ── K-Means ──────────────────────────────────────────────
    print("Training KM...")

    models['km'] = _KMeansScratch(k=4).fit(X_small)

    print("KM done")

    # ── Random Forest ────────────────────────────────────────
    if HAVE_RF:
        print("Training RF...")

        from RandomForest_scratch.random_forest import RandomForest as _RF

        models['rf'] = _RF(
            n_trees=15,
            max_depth=5
        )

        models['rf'].fit(X_small, y_small)

        print("RF done")

    else:
        print("Training RF Scratch...")

        models['rf'] = _RandomForestScratch(
            n_trees=15,
            max_depth=5,
            max_features=6
        ).fit(X_small, y_small)

        print("RF Scratch done")

    return models


with st.spinner("🔧 Training models (GB · iForest · K-Means · Random Forest)…"):
    MODELS = train_all(X_train, y_train)
# ─────────────────────────────────────────────────────────────────────────────
# 7. INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _feat_vec(rd):
    return np.array([float(rd.get(c, 0) or 0) for c in FEATURE_COLS])

def score_gb(rd):
    x = _feat_vec(rd)
    if MODELS['gb'] is not None:
        return float(MODELS['gb'].predict(x.reshape(1,-1))[0])
    pkt=x[1]; ack=x[9]; psh=x[16]
    base=0.40
    if pkt>800:base+=0.20
    if ack>5:  base+=0.15
    if psh>3:  base+=0.10
    return min(base + np.random.uniform(-0.04,0.04), 0.99)

def score_if(rd):
    x = _feat_vec(rd)
    if MODELS['if'] is not None:
        from Isolation_forest.isolation_forest import anomaly_score as _as
        pt = pd.DataFrame([{f"f{i}": v for i,v in enumerate(x)}])
        return float(_as(pt, MODELS['if'], 128))
    pkt=x[1]; dur=x[5]; iat=x[2]
    base=0.30
    if pkt>900: base+=0.28
    if dur<500: base+=0.18
    if iat>5e4: base+=0.15
    return min(base+np.random.uniform(-0.04,0.06), 0.99)

def score_km(rd):
    """Returns distance-based anomaly score [0,1] and cluster label."""
    x = _feat_vec(rd).reshape(1,-1)
    km = MODELS['km']
    if HAVE_KM and hasattr(km, 'predict'):
        cluster = int(km.predict(x)[0])
        # anomaly = distance to its centroid (normalised)
        c = km.cluster_centers_[cluster] if hasattr(km,'cluster_centers_') else x[0]
        d = float(np.linalg.norm(x[0]-c))
        score = min(d/(d+300+1e-9), 0.99)
    else:
        cluster = int(km._assign(x)[0])
        score   = km.anomaly_score(x[0])
    return score, cluster

def score_rf(rd):
    """Returns probability in [0,1] from Random Forest."""
    x = _feat_vec(rd)
    rf = MODELS['rf']
    if HAVE_RF and hasattr(rf,'predict_proba'):
        p = rf.predict_proba(x.reshape(1,-1))
        if hasattr(p,'__len__') and len(p[0])>1:
            return float(p[0][1])
        return float(p)
    return float(rf.predict_proba_one(x))

def classify_severity(gb,ifs,km,rf,attack):
    a=str(attack).upper()
    if max(gb,ifs,km,rf)>0.75 or any(k in a for k in("DDOS","INFILTRATION","BRUTE","BOT")):
        return "CRITICAL"
    if max(gb,ifs,km,rf)>0.55 or any(k in a for k in("DOS","WEB","SCAN","EXPLOIT")):
        return "HIGH"
    if max(gb,ifs,km,rf)>0.35 or "PORT" in a:
        return "MEDIUM"
    return "INFO"

# ─────────────────────────────────────────────────────────────────────────────
# 8. SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(row_idx=0, is_running=False, alert_log=[],
                ip_threat_map={}, total_packets=0, total_threats=0)
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# 9. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.markdown(
        "<small style='color:#3a6a3a;font-family:Share Tech Mono'>AI_SIHAM · v3.0 — 4 models</small>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Model status badges
    st.markdown(
        f"<div style='margin-bottom:8px'>"
        f"<span class='model-badge mb-gb'>GB {'✓' if HAVE_GB else '⚡'}</span>"
        f"<span class='model-badge mb-if'>IF {'✓' if HAVE_IF else '⚡'}</span>"
        f"<span class='model-badge mb-km'>KM {'✓' if HAVE_KM else '⚡'}</span>"
        f"<span class='model-badge mb-rf'>RF {'✓' if HAVE_RF else '⚡'}</span>"
        f"</div><small style='color:#3a5a3a;font-size:10px'>✓=real module · ⚡=built-in fallback</small>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### 🕹️ Simulation Engine")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("▶ Start"):  st.session_state.is_running = True
    with c2:
        if st.button("⏸ Pause"):  st.session_state.is_running = False
    if st.button("🔄 Reset"):
        for k,v in DEFAULTS.items():
            st.session_state[k] = type(v)() if isinstance(v,(list,dict)) else v
        st.rerun()

    refresh_speed   = st.slider("⏱ Stream Speed (s)",  0.05, 2.0, 0.20, 0.05)
    batch_size      = st.slider("📦 Batch per tick",    1,    10,  1)
    if_threshold    = st.slider("🌲 iForest threshold", 0.20, 0.95, 0.55, 0.05)
    gb_threshold    = st.slider("📈 GB threshold",      0.20, 0.95, 0.50, 0.05)
    km_threshold    = st.slider("🔵 K-Means threshold", 0.20, 0.95, 0.50, 0.05)
    rf_threshold    = st.slider("🌳 RF threshold",      0.20, 0.95, 0.50, 0.05)
    severity_filter = st.multiselect("🔎 Show Severities",
                        ["CRITICAL","HIGH","MEDIUM","INFO"],
                        default=["CRITICAL","HIGH","MEDIUM"])

    st.markdown("---")
    cursor_ph = st.empty()
    st.markdown(f"<small style='color:#3a5a3a'>Rows: {total_rows:,}</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ CyberShield — Real-Time Threat Intelligence")
st.markdown(
    "<p style='color:#5a8a5a;font-family:Share Tech Mono;font-size:0.8rem;margin-top:-10px'>"
    "<span class='model-badge mb-gb'>Gradient Boosting</span>"
    "<span class='model-badge mb-if'>Isolation Forest</span>"
    "<span class='model-badge mb-km'>K-Means</span>"
    "<span class='model-badge mb-rf'>Random Forest</span>"
    " &nbsp;· Live Network Stream"
    "</p>", unsafe_allow_html=True
)
st.markdown("---")

# KPI row — 8 columns
kpi_cols = st.columns(8)
kpi_ph   = [c.empty() for c in kpi_cols]
st.markdown("---")

# Tabs
tab_live, tab_alerts, tab_kmeans, tab_rf, tab_ips, tab_models = st.tabs([
    "📡 Live Stream",
    "🚨 Alert Center",
    "🔵 K-Means",
    "🌳 Random Forest",
    "🌐 IP Intelligence",
    "📊 All Model Analytics",
])

# ── Tab 1: Live Stream ────────────────────────────────────────────────────────
with tab_live:
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("#### 📈 GB Score vs Actual Label")
        gb_chart_ph = st.empty()
        st.markdown("#### 🌲 iForest Anomaly Score")
        if_chart_ph = st.empty()
    with lc2:
        st.markdown("#### 📉 GB Residual Error")
        residual_ph = st.empty()
        st.markdown("#### 🔵 K-Means Distance Score")
        km_chart_ph = st.empty()
    st.markdown("#### 🗂 Live Pipeline Log (all 4 models)")
    live_log_ph = st.empty()

# ── Tab 2: Alert Center ───────────────────────────────────────────────────────
with tab_alerts:
    al1, al2 = st.columns([3,1])
    with al1:
        st.markdown("#### 🚨 Real-Time Alert Feed")
        alert_feed_ph = st.empty()
    with al2:
        st.markdown("#### 📊 Severity Counts")
        sev_count_ph  = st.empty()
        st.markdown("#### 🏷️ Attack Types")
        attack_type_ph = st.empty()

# ── Tab 3: K-Means ────────────────────────────────────────────────────────────
with tab_kmeans:
    km1, km2 = st.columns(2)
    with km1:
        st.markdown("#### 🔵 K-Means Cluster Assignment (live)")
        km_scatter_ph = st.empty()
        st.markdown("#### 📊 Cluster Distribution")
        km_dist_ph = st.empty()
    with km2:
        st.markdown("#### 🌡️ K-Means Anomaly Score Over Time")
        km_score_ph = st.empty()
        st.markdown("#### 📋 K-Means Metrics")
        km_metrics_ph = st.empty()
    st.markdown("#### 🔬 K-Means: Packet Length Mean vs Flow IAT Std (coloured by cluster)")
    km_feature_ph = st.empty()

# ── Tab 4: Random Forest ──────────────────────────────────────────────────────
with tab_rf:
    rf1, rf2 = st.columns(2)
    with rf1:
        st.markdown("#### 🌳 RF Predicted Probability (live)")
        rf_score_ph = st.empty()
        st.markdown("#### 📊 RF Prediction Distribution")
        rf_dist_ph = st.empty()
    with rf2:
        st.markdown("#### 🎯 RF vs Actual Label")
        rf_vs_actual_ph = st.empty()
        st.markdown("#### 📋 RF Metrics")
        rf_metrics_ph = st.empty()
    st.markdown("#### 🌲 RF Score vs Packet Length Mean")
    rf_feature_ph = st.empty()

# ── Tab 5: IP Intelligence ────────────────────────────────────────────────────
with tab_ips:
    ip1, ip2 = st.columns(2)
    with ip1:
        st.markdown("#### 🔴 Top Threat Source IPs")
        top_src_ph = st.empty()
    with ip2:
        st.markdown("#### 🎯 Most Targeted Destination IPs")
        top_dst_ph = st.empty()
    st.markdown("#### 📋 Recent Flow Log (Src → Dst)")
    flow_log_ph = st.empty()

# ── Tab 6: All Model Analytics ────────────────────────────────────────────────
with tab_models:
    ma1, ma2 = st.columns(2)
    with ma1:
        st.markdown("#### 📐 iForest Score Histogram")
        if_hist_ph = st.empty()
        st.markdown("#### 📐 K-Means Score Histogram")
        km_hist_ph = st.empty()
    with ma2:
        st.markdown("#### 📐 GB Score Histogram")
        gb_hist_ph = st.empty()
        st.markdown("#### 📐 RF Score Histogram")
        rf_hist_ph = st.empty()
    st.markdown("#### 🔀 4-Model Score Comparison (latest window)")
    model_compare_ph = st.empty()

st.markdown("---")
status_bar_ph = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# 11. STREAMING LOOP
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.is_running:

    history  = []
    flow_log = []
    alert_log = st.session_state.alert_log
    ip_src    = st.session_state.ip_threat_map
    ip_dst    = {}

    while st.session_state.is_running and st.session_state.row_idx < total_rows:

        # ── process batch ──────────────────────────────────────────────────────
        for _ in range(batch_size):
            if st.session_state.row_idx >= total_rows: break
            idx = st.session_state.row_idx
            row = dataset.iloc[idx]
            rd  = row.to_dict()

            pkt_mean = float(rd.get("Packet Length Mean", 500) or 500)
            flow_dur = float(rd.get("Flow Duration",     1000) or 1000)
            iat_std  = float(rd.get("Flow IAT Std",       100) or 100)
            ack_cnt  = float(rd.get("ACK Flag Count",       0) or 0)
            psh_cnt  = float(rd.get("PSH Flag Count",       0) or 0)
            fin_cnt  = float(rd.get("FIN Flag Count",       0) or 0)

            src_ip    = str(rd.get("Source IP",   "10.0.0.1"))
            dst_ip    = str(rd.get("Dest IP",     "192.168.1.1"))
            protocol  = str(rd.get("Protocol",    "TCP"))
            timestamp = str(rd.get("Timestamp",   datetime.now().isoformat()))[:19]
            attack    = str(rd.get("Attack_Type", "BENIGN"))
            actual    = int(rd.get("Is_Attack",   0))

            # ── 4 model scores ────────────────────────────────────────────────
            gb_s               = score_gb(rd)
            if_s               = score_if(rd)
            km_s, km_cluster   = score_km(rd)
            rf_s               = score_rf(rd)

            residual   = actual - gb_s
            is_threat  = gb_s>gb_threshold or if_s>if_threshold or km_s>km_threshold or rf_s>rf_threshold
            severity   = classify_severity(gb_s, if_s, km_s, rf_s, attack)

            # IP tracking
            if is_threat:
                if src_ip not in ip_src:
                    ip_src[src_ip] = {"count":0,"attacks":[],"dst_ips":set()}
                ip_src[src_ip]["count"] += 1
                ip_src[src_ip]["attacks"].append(attack)
                ip_src[src_ip]["dst_ips"].add(dst_ip)
                ip_dst[dst_ip] = ip_dst.get(dst_ip,0)+1

            if severity in ("CRITICAL","HIGH","MEDIUM"):
                alert_log.insert(0,{
                    "time":timestamp,"severity":severity,
                    "src_ip":src_ip,"dst_ip":dst_ip,"protocol":protocol,
                    "attack":attack,
                    "gb":round(gb_s,4),"if":round(if_s,4),
                    "km":round(km_s,4),"rf":round(rf_s,4),
                })
                if len(alert_log)>150: alert_log.pop()

            history.insert(0,{
                "Time":timestamp, "Source IP":src_ip, "Dest IP":dst_ip,
                "Protocol":protocol,
                "Pkt Len":round(pkt_mean,1), "Flow Dur":round(flow_dur,1),
                "ACK":int(ack_cnt), "PSH":int(psh_cnt), "FIN":int(fin_cnt),
                "IAT Std":round(iat_std,1),
                "GB Score":round(gb_s,4),   "IF Score":round(if_s,4),
                "KM Score":round(km_s,4),   "RF Score":round(rf_s,4),
                "KM Cluster":km_cluster,
                "Residual":round(residual,4),
                "Attack":attack, "Severity":severity,
                "Actual":actual,
                "Threat":"⚠️ YES" if is_threat else "✅ NO",
            })
            flow_log.insert(0,{
                "Time":timestamp,"Source IP":src_ip,"→ Dest IP":dst_ip,
                "Protocol":protocol,"Attack":attack,
                "GB":round(gb_s,3),"IF":round(if_s,3),
                "KM":round(km_s,3),"RF":round(rf_s,3),
                "Severity":severity,
            })
            if len(history)  > 80: history.pop()
            if len(flow_log) > 50: flow_log.pop()
            st.session_state.row_idx       += 1
            st.session_state.total_packets += 1
            if is_threat: st.session_state.total_threats += 1

        # ── build df ─────────────────────────────────────────────────────────
        df = pd.DataFrame(history)
        cursor_ph.metric("Row Cursor", f"{st.session_state.row_idx} / {total_rows}")
        threat_rate = st.session_state.total_threats / max(st.session_state.total_packets,1) * 100

        # ── KPIs ──────────────────────────────────────────────────────────────
        kpi_data = [
            ("Packets",          st.session_state.total_packets,       None),
            ("Pkt Len (bytes)",  f"{df['Pkt Len'].iloc[0]:.0f}",        None),
            ("GB Score",         f"{df['GB Score'].iloc[0]:.4f}",
                                  "⚠ THREAT" if df['GB Score'].iloc[0]>gb_threshold else "✓ ok"),
            ("IF Score",         f"{df['IF Score'].iloc[0]:.4f}",
                                  "⚠ OUTLIER" if df['IF Score'].iloc[0]>if_threshold else "✓ ok"),
            ("KM Score",         f"{df['KM Score'].iloc[0]:.4f}",
                                  f"cluster {df['KM Cluster'].iloc[0]}"),
            ("RF Prob",          f"{df['RF Score'].iloc[0]:.4f}",
                                  "⚠ THREAT" if df['RF Score'].iloc[0]>rf_threshold else "✓ ok"),
            ("Active Threats",   int((df["Threat"]=="⚠️ YES").sum()),   "in window"),
            ("Threat Rate",      f"{threat_rate:.1f}%",                  None),
        ]
        for ph,(lbl,val,delta) in zip(kpi_ph, kpi_data):
            ph.metric(lbl,val,delta)

        # ── Tab 1: Live Stream ─────────────────────────────────────────────────
        gb_chart_ph.scatter_chart(df[["Pkt Len","GB Score","Actual"]], x="Pkt Len", height=210)
        residual_ph.scatter_chart(df[["Pkt Len","Residual"]], x="Pkt Len", y="Residual", height=210)
        if_chart_ph.scatter_chart(df[["Pkt Len","IF Score"]], x="Pkt Len", y="IF Score", height=210)
        km_chart_ph.scatter_chart(df[["Pkt Len","KM Score"]], x="Pkt Len", y="KM Score", height=210)
        live_log_ph.dataframe(
            df[["Time","Source IP","Dest IP","Protocol","Pkt Len",
                "GB Score","IF Score","KM Score","RF Score","Attack","Severity","Threat"]].head(12),
            use_container_width=True, hide_index=True
        )

        # ── Tab 2: Alerts ──────────────────────────────────────────────────────
        filtered = [a for a in alert_log if a["severity"] in severity_filter]
        if filtered:
            html = "".join(
                f'<div class="sev-{a["severity"]}">'
                f'{SEV_ICON[a["severity"]]} <b>[{a["severity"]}]</b> {a["time"]} &nbsp;|&nbsp; '
                f'<b>SRC:</b> {a["src_ip"]} <b>→ DST:</b> {a["dst_ip"]} &nbsp;|&nbsp; '
                f'{a["protocol"]} &nbsp;|&nbsp; <b>{a["attack"]}</b> &nbsp;|&nbsp; '
                f'GB={a["gb"]} IF={a["if"]} KM={a["km"]} RF={a["rf"]}'
                f'</div>'
                for a in filtered[:18]
            )
            alert_feed_ph.markdown(html, unsafe_allow_html=True)
        sev_count_ph.bar_chart(df["Severity"].value_counts(), height=200)
        attack_type_ph.bar_chart(df["Attack"].value_counts().head(8), height=200)

        # ── Tab 3: K-Means ────────────────────────────────────────────────────
        km_scatter_ph.scatter_chart(
            df[["Pkt Len","IAT Std","KM Cluster"]],
            x="Pkt Len", y="IAT Std", height=220
        )
        km_dist_ph.bar_chart(df["KM Cluster"].value_counts().sort_index(), height=200)
        km_score_ph.line_chart(df[["KM Score"]].reset_index(drop=True), height=220)
        km_avg  = df["KM Score"].mean()
        km_out  = int((df["KM Score"]>km_threshold).sum())
        km_metrics_ph.markdown(
            f"<div class='sev-INFO'>"
            f"Avg score: {km_avg:.4f} &nbsp;|&nbsp; "
            f"Outliers (>{km_threshold}): {km_out} / {len(df)} &nbsp;|&nbsp; "
            f"K = 4 clusters"
            f"</div>", unsafe_allow_html=True
        )
        km_feature_ph.scatter_chart(
            df[["Pkt Len","IAT Std","KM Score"]],
            x="Pkt Len", y="IAT Std", height=220
        )

        # ── Tab 4: Random Forest ───────────────────────────────────────────────
        rf_score_ph.line_chart(df[["RF Score"]].reset_index(drop=True), height=220)
        rf_dist_ph.bar_chart(df["RF Score"].round(1).value_counts().sort_index(), height=200)
        rf_vs_actual_ph.scatter_chart(
            df[["Pkt Len","RF Score","Actual"]], x="Pkt Len", height=220
        )
        rf_acc = float(((df["RF Score"]>rf_threshold).astype(int)==df["Actual"]).mean())
        rf_out = int((df["RF Score"]>rf_threshold).sum())
        rf_metrics_ph.markdown(
            f"<div class='sev-INFO'>"
            f"Window accuracy: {rf_acc*100:.1f}% &nbsp;|&nbsp; "
            f"Predicted threats: {rf_out} / {len(df)} &nbsp;|&nbsp; "
            f"Trees: 15"
            f"</div>", unsafe_allow_html=True
        )
        rf_feature_ph.scatter_chart(
            df[["Pkt Len","RF Score"]], x="Pkt Len", y="RF Score", height=220
        )

        # ── Tab 5: IPs ────────────────────────────────────────────────────────
        if ip_src:
            src_df = pd.DataFrame([
                {"Source IP":ip,"Threat Hits":d["count"],
                 "Unique DST":len(d["dst_ips"]),
                 "Top Attack":max(set(d["attacks"]),key=d["attacks"].count) if d["attacks"] else "N/A"}
                for ip,d in sorted(ip_src.items(),key=lambda x:-x[1]["count"])
            ]).head(12)
            top_src_ph.dataframe(src_df, use_container_width=True, hide_index=True)
        if ip_dst:
            dst_df = pd.DataFrame([
                {"Dest IP":ip,"Times Targeted":cnt}
                for ip,cnt in sorted(ip_dst.items(),key=lambda x:-x[1])
            ]).head(12)
            top_dst_ph.dataframe(dst_df, use_container_width=True, hide_index=True)
        flow_log_ph.dataframe(pd.DataFrame(flow_log).head(15), use_container_width=True, hide_index=True)

        # ── Tab 6: All Analytics ──────────────────────────────────────────────
        if_hist_ph.bar_chart(df["IF Score"].round(1).value_counts().sort_index(), height=200)
        gb_hist_ph.bar_chart(df["GB Score"].round(1).value_counts().sort_index(), height=200)
        km_hist_ph.bar_chart(df["KM Score"].round(1).value_counts().sort_index(), height=200)
        rf_hist_ph.bar_chart(df["RF Score"].round(1).value_counts().sort_index(), height=200)
        model_compare_ph.line_chart(
            df[["GB Score","IF Score","KM Score","RF Score"]].reset_index(drop=True),
            height=220
        )

        # ── Status bar ─────────────────────────────────────────────────────────
        status_bar_ph.markdown(
            f"<p style='font-family:Share Tech Mono;font-size:0.75rem;color:#5a8a5a'>"
            f"⚡ STREAMING · Row {st.session_state.row_idx}/{total_rows} · "
            f"Threat rate: {threat_rate:.1f}% · Alerts: {len(alert_log)} · "
            f"Threat IPs: {len(ip_src)}"
            f"</p>", unsafe_allow_html=True
        )

        st.session_state.alert_log     = alert_log
        st.session_state.ip_threat_map = ip_src
        time.sleep(refresh_speed)

    if st.session_state.row_idx >= total_rows:
        st.session_state.is_running = False
        st.success("🎉 Simulation complete — all 4 models processed the full dataset.")

# ─────────────────────────────────────────────────────────────────────────────
# PAUSED STATE
# ─────────────────────────────────────────────────────────────────────────────
else:
    cursor_ph.metric("Row Cursor", f"{st.session_state.row_idx} / {total_rows}")
    if st.session_state.row_idx == 0:
        status_bar_ph.markdown(
            "<div class='sev-INFO'>⏸️ Dashboard ready. "
            "Press <b>▶ Start</b> in the sidebar to stream data through "
            "Gradient Boosting · Isolation Forest · K-Means · Random Forest.</div>",
            unsafe_allow_html=True
        )
    else:
        tr = st.session_state.total_threats/max(st.session_state.total_packets,1)*100
        status_bar_ph.markdown(
            f"<div class='sev-MEDIUM'>⏸️ PAUSED at row {st.session_state.row_idx}/{total_rows} · "
            f"Threat rate: {tr:.1f}% · Alerts: {len(st.session_state.alert_log)}</div>",
            unsafe_allow_html=True
        )

    if st.session_state.alert_log:
        filtered = [a for a in st.session_state.alert_log if a["severity"] in severity_filter]
        html = "".join(
            f'<div class="sev-{a["severity"]}">'
            f'{SEV_ICON[a["severity"]]} <b>[{a["severity"]}]</b> {a["time"]} | '
            f'<b>SRC:</b> {a["src_ip"]} → <b>DST:</b> {a["dst_ip"]} | '
            f'{a["protocol"]} | <b>{a["attack"]}</b> | '
            f'GB={a["gb"]} IF={a["if"]} KM={a["km"]} RF={a["rf"]}</div>'
            for a in filtered[:18]
        )
        with tab_alerts:
            alert_feed_ph.markdown(html, unsafe_allow_html=True)

    ip_src = st.session_state.ip_threat_map
    if ip_src:
        src_df = pd.DataFrame([
            {"Source IP":ip,"Threat Hits":d["count"],
             "Unique DST":len(d["dst_ips"]),
             "Top Attack":max(set(d["attacks"]),key=d["attacks"].count) if d["attacks"] else "N/A"}
            for ip,d in sorted(ip_src.items(),key=lambda x:-x[1]["count"])
        ]).head(12)
        with tab_ips:
            top_src_ph.dataframe(src_df, use_container_width=True, hide_index=True)
