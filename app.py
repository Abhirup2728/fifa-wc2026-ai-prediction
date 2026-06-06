
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import math
import os
from itertools import permutations, combinations

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "FIFA WC2026 AI Predictor",
    page_icon   = "⚽",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 12px; color: white;
        text-align: center; margin: 0.3rem;
    }
    .gold-card   { background: linear-gradient(135deg,#f7971e,#ffd200); color:#1a1a1a; }
    .silver-card { background: linear-gradient(135deg,#bdc3c7,#2c3e50); color:white; }
    .bronze-card { background: linear-gradient(135deg,#cd7f32,#a0522d); color:white; }
    .predict-box {
        background: #f8f9fa; border-radius: 12px;
        padding: 1.5rem; border-left: 4px solid #667eea;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Base paths ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models")
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed")
RAW_PATH    = os.path.join(BASE_DIR, "data", "raw")

# ── Load all artifacts (cached) ───────────────────────────────
@st.cache_resource
def load_models():
    with open(f"{MODELS_PATH}/logistic_regression.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODELS_PATH}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODELS_PATH}/elo_ratings.pkl", "rb") as f:
        elo = pickle.load(f)
    with open(f"{MODELS_PATH}/feature_cols.pkl", "rb") as f:
        feat_cols = pickle.load(f)
    return model, scaler, elo, feat_cols

@st.cache_data
def load_data():
    sim     = pd.read_csv(f"{DATA_PATH}/simulation_results.csv")
    wc2026  = pd.read_csv(f"{DATA_PATH}/wc2026_groups.csv")
    history = pd.read_csv(f"{DATA_PATH}/wc_historical_record.csv")
    results = pd.read_csv(f"{RAW_PATH}/results.csv")
    results["date"] = pd.to_datetime(results["date"])
    features= pd.read_csv(f"{DATA_PATH}/features.csv")
    features["date"] = pd.to_datetime(features["date"])
    return sim, wc2026, history, results, features

model, scaler, elo_ratings, FEATURE_COLS = load_models()
sim_df, wc2026, wc_history, raw_df, features_df = load_data()

WC_TEAMS = sorted(wc2026["team"].tolist())

# ── Helper functions ──────────────────────────────────────────
def sanitize(val, default=0.0):
    if val is None: return default
    try:
        if math.isnan(val) or math.isinf(val): return default
    except: return default
    return float(val)

def get_latest_stats(team):
    D = {"form_5":7.5,"form_10":15.0,"goals_scored":1.2,"goals_conceded":1.2}
    hm = features_df[features_df["home_team"] == team]
    am = features_df[features_df["away_team"] == team]
    if len(hm) == 0 and len(am) == 0: return D
    lh = hm.iloc[-1] if len(hm) > 0 else None
    la = am.iloc[-1] if len(am) > 0 else None
    if lh is not None and la is not None:
        row, prefix = (lh,"home") if lh["date"] >= la["date"] else (la,"away")
    elif lh is not None: row, prefix = lh, "home"
    else: row, prefix = la, "away"
    return {
        "form_5"        : sanitize(row[f"{prefix}_form_5"],         D["form_5"]),
        "form_10"       : sanitize(row[f"{prefix}_form_10"],        D["form_10"]),
        "goals_scored"  : sanitize(row[f"{prefix}_goals_scored_avg"],   D["goals_scored"]),
        "goals_conceded": sanitize(row[f"{prefix}_goals_conceded_avg"], D["goals_conceded"]),
    }

def get_h2h(team_a, team_b):
    m = raw_df[
        ((raw_df["home_team"]==team_a)&(raw_df["away_team"]==team_b))|
        ((raw_df["home_team"]==team_b)&(raw_df["away_team"]==team_a))
    ]
    if len(m)==0: return 0.5, 0
    wins = sum(
        1 for _,r in m.iterrows()
        if (r["home_team"]==team_a and r["home_score"]>r["away_score"]) or
           (r["away_team"]==team_a and r["away_score"]>r["home_score"])
    )
    return wins/len(m), len(m)

def build_features(home, away):
    h_elo = sanitize(elo_ratings.get(home,1500),1500)
    a_elo = sanitize(elo_ratings.get(away,1500),1500)
    elo_d = h_elo - a_elo
    hp = get_latest_stats(home)
    ap = get_latest_stats(away)
    h2h_wr, h2h_p = get_h2h(home, away)
    row = {
        "home_elo":h_elo,"away_elo":a_elo,"elo_diff":elo_d,
        "elo_win_prob_home":sanitize(1/(1+10**(-elo_d/400)),0.5),
        "home_form_5":hp["form_5"],"away_form_5":ap["form_5"],
        "home_form_10":hp["form_10"],"away_form_10":ap["form_10"],
        "form_diff_5":hp["form_5"]-ap["form_5"],
        "form_diff_10":hp["form_10"]-ap["form_10"],
        "home_goals_scored_avg":hp["goals_scored"],
        "home_goals_conceded_avg":hp["goals_conceded"],
        "away_goals_scored_avg":ap["goals_scored"],
        "away_goals_conceded_avg":ap["goals_conceded"],
        "attack_diff":hp["goals_scored"]-ap["goals_scored"],
        "defense_diff":hp["goals_conceded"]-ap["goals_conceded"],
        "h2h_home_win_rate":sanitize(h2h_wr,0.5),
        "h2h_matches_played":sanitize(h2h_p,0),
        "is_neutral":1.0,"home_advantage":0.0,
        "tournament_weight":8.0,"year":2026.0,"month":6.0,
    }
    feat = pd.DataFrame([row])[FEATURE_COLS].fillna(0.0)
    return feat

def predict_match(home, away):
    feat  = build_features(home, away)
    scaled= scaler.transform(feat)
    proba = model.predict_proba(scaled)[0]
    proba = np.clip(proba,0,None); proba = proba/proba.sum()
    return {"home_win":float(proba[0]),"draw":float(proba[1]),"away_win":float(proba[2])}

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ FIFA WC2026")
    st.markdown("### AI Prediction Platform")
    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Accuracy:** 59.64%")
    st.markdown("**Simulations:** 10,000 Monte Carlo")
    st.markdown("**Features:** 23 engineered features")
    st.markdown("**Training data:** 32,260 matches")
    st.markdown("---")
    champion = sim_df.iloc[0]
    st.markdown(f"### 🏆 Predicted Champion")
    st.markdown(f"## 🇪🇸 {champion['team']}")
    st.metric("Win Probability", f"{champion['win_title_%']:.2f}%")
    st.markdown("---")
    st.markdown("Built by **Abhirup Gumtya**")
    st.markdown("B.Tech CSE (AI & ML) | Brainware University")

# ════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">⚽ FIFA World Cup 2026 — AI Prediction Platform</div>',
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:1.1rem;'>"
            "Machine Learning powered predictions using 49,378 historical matches "
            "& 10,000 Monte Carlo simulations</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "⚽ Match Predictor",
    "📊 Team Analytics",
    "🏆 Tournament Bracket",
    "📈 EDA Insights"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 WC2026 Championship Predictions")
    st.markdown("Based on **10,000 Monte Carlo tournament simulations**")

    top3 = sim_df.head(3)
    c1,c2,c3 = st.columns(3)
    for col, card, medal, row in zip(
        [c1,c2,c3],
        ["gold-card","silver-card","bronze-card"],
        ["🥇","🥈","🥉"],
        top3.itertuples()
    ):
        with col:
            st.markdown(
                f'<div class="metric-card {card}">                <div style="font-size:2.5rem">{medal}</div>                <div style="font-size:1.6rem;font-weight:800">{row.team}</div>                <div style="font-size:1.2rem">{row.win_title_pct:.2f}% chance</div>                <div>Elo: {row.elo_rating:.0f}</div></div>'.replace("win_title_pct",
                "win_title_%"),
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Top 16 Teams — Championship Probability")
    top16 = sim_df.head(16)
    fig = px.bar(top16, x="team", y="win_title_%",
                 color="win_title_%", color_continuous_scale="RdYlGn",
                 labels={"win_title_%":"Win Probability (%)","team":"Team"},
                 text=top16["win_title_%"].apply(lambda x: f"{x:.1f}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=450,
                      xaxis_tickangle=-30, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Full Stage Progression Probabilities")
    display_cols = ["team","group","elo_rating","qualify_%",
                    "reach_qf_%","reach_semi_%","reach_final_%","win_title_%"]
    st.dataframe(
        sim_df[display_cols].style
            .background_gradient(subset=["win_title_%","reach_final_%"], cmap="RdYlGn")
            .format({"elo_rating":"{:.0f}",
                     "qualify_%":"{:.1f}%","reach_qf_%":"{:.1f}%",
                     "reach_semi_%":"{:.1f}%","reach_final_%":"{:.1f}%",
                     "win_title_%":"{:.2f}%"}),
        use_container_width=True, height=400
    )

# ════════════════════════════════════════════════════════════
# TAB 2 — MATCH PREDICTOR
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("⚽ AI Match Predictor")
    st.markdown("Select any two WC2026 teams to get AI-powered win probabilities")
    st.markdown("---")

    col1, col_vs, col2 = st.columns([2,1,2])
    with col1:
        st.markdown("### 🏠 Team A")
        home_team = st.selectbox("Select Home Team", WC_TEAMS,
                                  index=WC_TEAMS.index("Spain") if "Spain" in WC_TEAMS else 0,
                                  key="home_sel")
    with col_vs:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;color:#e74c3c'>VS</h2>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("### ✈️ Team B")
        away_options = [t for t in WC_TEAMS if t != home_team]
        away_team = st.selectbox("Select Away Team", away_options,
                                  index=away_options.index("France") if "France" in away_options else 0,
                                  key="away_sel")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
        with st.spinner("Running AI prediction..."):
            pred = predict_match(home_team, away_team)

        winner_key = max(pred, key=pred.get)
        winner_label = {
            "home_win": f"🏠 {home_team} wins",
            "draw"    : "🤝 Draw",
            "away_win": f"✈️ {away_team} wins"
        }[winner_key]

        st.markdown("---")
        st.markdown(f"### 🎯 Prediction: **{winner_label}**")

        c1,c2,c3 = st.columns(3)
        with c1:
            st.metric(f"🏠 {home_team} Win", f"{pred['home_win']*100:.1f}%")
        with c2:
            st.metric("🤝 Draw", f"{pred['draw']*100:.1f}%")
        with c3:
            st.metric(f"✈️ {away_team} Win", f"{pred['away_win']*100:.1f}%")

        # Probability bar chart
        fig = go.Figure(go.Bar(
            x=[pred["home_win"]*100, pred["draw"]*100, pred["away_win"]*100],
            y=[f"{home_team} Win","Draw",f"{away_team} Win"],
            orientation="h",
            marker_color=["#2ecc71","#f39c12","#e74c3c"],
            text=[f"{v*100:.1f}%" for v in [pred["home_win"],pred["draw"],pred["away_win"]]],
            textposition="outside"
        ))
        fig.update_layout(title=f"Match Probability: {home_team} vs {away_team}",
                          height=280, xaxis=dict(range=[0,100]),
                          xaxis_title="Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

        # H2H stats
        h2h_wr, h2h_p = get_h2h(home_team, away_team)
        if h2h_p > 0:
            st.info(f"📊 **Head-to-Head:** {h2h_p} historical meetings | "
                    f"{home_team} won {h2h_wr*100:.1f}% | "
                    f"{away_team} won {(1-h2h_wr)*100:.1f}%")
        else:
            st.info("📊 **Head-to-Head:** No prior meetings in dataset")

        # Elo comparison
        h_elo = elo_ratings.get(home_team, 1500)
        a_elo = elo_ratings.get(away_team, 1500)
        st.info(f"📈 **Elo Ratings:** {home_team}: {h_elo:.0f} | "
                f"{away_team}: {a_elo:.0f} | "
                f"Difference: {abs(h_elo-a_elo):.0f} points")

# ════════════════════════════════════════════════════════════
# TAB 3 — TEAM ANALYTICS
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Team Analytics Dashboard")
    selected_team = st.selectbox("Select a WC2026 Team", WC_TEAMS, key="team_sel")

    team_sim  = sim_df[sim_df["team"] == selected_team].iloc[0]
    team_hist = wc_history[wc_history["team"] == selected_team]
    group     = wc2026[wc2026["team"] == selected_team]["group"].values[0]
    stats     = get_latest_stats(selected_team)
    elo_val   = elo_ratings.get(selected_team, 1500)

    st.markdown(f"### {selected_team} — Group {group[-1]}")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🏆 Title Probability",  f"{team_sim['win_title_%']:.2f}%")
    c2.metric("🎯 Final Probability",  f"{team_sim['reach_final_%']:.2f}%")
    c3.metric("📈 Elo Rating",         f"{elo_val:.0f}")
    c4.metric("✅ Qualify Probability",f"{team_sim['qualify_%']:.2f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Tournament Stage Probabilities")
        stages  = ["Qualify","Quarter-Final","Semi-Final","Final","Win Title"]
        probs   = [team_sim["qualify_%"], team_sim["reach_qf_%"],
                   team_sim["reach_semi_%"], team_sim["reach_final_%"],
                   team_sim["win_title_%"]]
        fig = go.Figure(go.Bar(
            x=stages, y=probs, marker_color="#667eea",
            text=[f"{p:.1f}%" for p in probs], textposition="outside"
        ))
        fig.update_layout(yaxis_title="Probability (%)", height=350,
                          yaxis=dict(range=[0, max(probs)*1.2]))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🏆 WC Historical Record")
        if len(team_hist) > 0:
            hist = team_hist.iloc[0]
            fig = go.Figure(go.Pie(
                labels=["Wins","Draws","Losses"],
                values=[hist["wins"], hist["draws"], hist["losses"]],
                marker_colors=["#2ecc71","#f39c12","#e74c3c"],
                hole=0.4
            ))
            fig.update_layout(height=350,
                               annotations=[dict(text=f"{hist['played']}\nplayed",
                                                 x=0.5,y=0.5,showarrow=False,font_size=14)])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"{selected_team} has no World Cup history in our dataset.")

    st.subheader("⚽ Group Stage Opponents")
    group_teams = wc2026[wc2026["group"] == group]["team"].tolist()
    opponents   = [t for t in group_teams if t != selected_team]

    cols = st.columns(len(opponents))
    for col, opp in zip(cols, opponents):
        with col:
            pred = predict_match(selected_team, opp)
            winner_pct = pred["home_win"]*100
            st.markdown(f"**vs {opp}**")
            st.metric("Win", f"{winner_pct:.1f}%")
            st.metric("Draw", f"{pred['draw']*100:.1f}%")
            st.metric("Loss", f"{pred['away_win']*100:.1f}%")

# ════════════════════════════════════════════════════════════
# TAB 4 — TOURNAMENT BRACKET
# ════════════════════════════════════════════════════════════
with tab4:
    st.header("🏆 Predicted Tournament Bracket")
    st.markdown("Showing top-probability progression based on 10,000 simulations")
    st.markdown("---")

    # Group stage results
    st.subheader("🔵 Predicted Group Stage Qualifiers")
    cols = st.columns(4)
    for idx, group_name in enumerate(sorted(wc2026["group"].unique())):
        group_teams = wc2026[wc2026["group"] == group_name]["team"].tolist()
        group_sim   = sim_df[sim_df["team"].isin(group_teams)].sort_values(
                       "qualify_%", ascending=False)

        col = cols[idx % 4]
        with col:
            st.markdown(f"**Group {group_name[-1]}**")
            for _, row in group_sim.iterrows():
                bar_w = int(row["qualify_%"])
                emoji = "🟢" if row["qualify_%"] > 60 else "🟡" if row["qualify_%"] > 40 else "🔴"
                st.markdown(f"{emoji} {row['team']} ({row['qualify_%']:.0f}%)")
            st.markdown("---")

    # Stage-by-stage
    st.subheader("🏆 Predicted Knockout Progression")
    stages_show = ["reach_qf_%","reach_semi_%","reach_final_%","win_title_%"]
    stage_names = ["Quarter-Final","Semi-Final","Final","Champion"]

    for stage_col, stage_name in zip(stages_show, stage_names):
        n = {"Quarter-Final":8,"Semi-Final":4,"Final":2,"Champion":1}[stage_name]
        top_n = sim_df.nlargest(n, stage_col)[["team", stage_col]]
        st.markdown(f"#### {stage_name}")
        cols = st.columns(n)
        for col, (_, row) in zip(cols, top_n.iterrows()):
            with col:
                pct = row[stage_col]
                color = "#2ecc71" if pct > 20 else "#f39c12" if pct > 10 else "#3498db"
                st.markdown(
                    f'<div style="background:{color};padding:0.6rem;border-radius:8px;text-align:center;color:white;margin:0.2rem"><b>{row["team"]}</b><br>{pct:.1f}%</div>',
                    unsafe_allow_html=True
                )
        st.markdown("")

# ════════════════════════════════════════════════════════════
# TAB 5 — EDA INSIGHTS
# ════════════════════════════════════════════════════════════
with tab5:
    st.header("📈 Key EDA Insights from 49,378 Matches")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches",   "49,378")
    c2.metric("Home Win Rate",   "48.9%")
    c3.metric("Draw Rate",       "22.8%")
    c4.metric("Avg Goals/Match", "2.94")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚽ Outcome Distribution")
        fig = px.pie(values=[48.9,22.8,28.2],
                     names=["Home Win","Draw","Away Win"],
                     color_discrete_sequence=["#2ecc71","#f39c12","#e74c3c"],
                     hole=0.4)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏆 Top 10 Teams by WC Wins")
        top_wc = wc_history.nlargest(10,"wins")[["team","wins","draws","losses"]]
        fig = px.bar(top_wc, x="team", y=["wins","draws","losses"],
                     color_discrete_sequence=["#2ecc71","#f39c12","#e74c3c"],
                     barmode="stack", labels={"value":"Matches","variable":"Result"})
        fig.update_layout(height=320, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 WC2026 Group Difficulty")
    group_avg = []
    for g in sorted(wc2026["group"].unique()):
        teams = wc2026[wc2026["group"]==g]["team"].tolist()
        avg_elo = np.mean([elo_ratings.get(t,1500) for t in teams])
        group_avg.append({"group":g[-1],"avg_elo":avg_elo})
    gdf = pd.DataFrame(group_avg).sort_values("avg_elo",ascending=False)
    fig = px.bar(gdf, x="group", y="avg_elo", color="avg_elo",
                 color_continuous_scale="RdYlGn_r",
                 labels={"avg_elo":"Average Elo Rating","group":"Group"},
                 title="Average Elo Rating per Group (higher = tougher)")
    fig.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Model Performance")
    model_data = {
        "Model":["Logistic Regression","Random Forest","XGBoost","Neural Network"],
        "Accuracy":[59.64,57.49,55.89,54.44],
        "Log Loss":[0.8692,0.8879,0.9016,0.8983]
    }
    mdf = pd.DataFrame(model_data)
    fig = px.bar(mdf, x="Model", y="Accuracy", color="Accuracy",
                 color_continuous_scale="RdYlGn", text="Accuracy",
                 labels={"Accuracy":"Test Accuracy (%)"})
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.85rem'>"
    "Built by <b>Abhirup Gumtya</b> | B.Tech CSE (AI & ML) | Brainware University | "
    "Powered by Scikit-learn, Plotly & Streamlit</p>",
    unsafe_allow_html=True
)
