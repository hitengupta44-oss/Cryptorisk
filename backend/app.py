from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from collections import deque
from typing import Optional
import os, time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REAL_DATA:   dict  = {}
PRED_DATA:   dict  = {}
RISK_STORE:  dict  = {}
ONCHAIN:     dict  = {}
SOLANA:      dict  = {}
CRASH_SIM:   dict  = {}
VAR_STORE:   dict  = {}
CORR_STORE:  dict  = {}
PRICES:      dict  = {}
VOICE_ALERT: dict  = {"text":"","audio_ready":False,"time":""}
RISK_HIST:   deque = deque(maxlen=200)
last_real: Optional[str] = None

def _trim(d, n):
    while len(d) > n: d.pop(next(iter(d)))

@app.get("/")
def home(): return {"status": "CryptoRisk Backend ✅"}

@app.post("/update")
def update(payload: dict):
    global last_real
    t   = payload.get("time", str(time.time()))
    typ = payload.get("type")

    if typ == "real":
        REAL_DATA[t] = payload
        if last_real is None or t > last_real:
            last_real = t; PRED_DATA.clear()
        _trim(REAL_DATA, 60)

    elif typ == "prediction":
        PRED_DATA[t] = payload
        _trim(PRED_DATA, 10)

    elif typ == "risk_update":
        if "risk_score"     in payload:
            RISK_STORE.update({**payload["risk_score"], "timestamp": t})
            RISK_HIST.append({"time":t,"score":payload["risk_score"].get("score"),
                               "level":payload["risk_score"].get("level")})
        if "fear_greed"     in payload:
            ONCHAIN.update({"fear_greed":payload.get("fear_greed",{}),
                            "funding_rate":payload.get("funding_rate",{}),
                            "open_interest":payload.get("open_interest",{}),"timestamp":t})
        if "solana_onchain" in payload: SOLANA.update({**payload["solana_onchain"],"timestamp":t})
        if "crash_sim"      in payload: CRASH_SIM.update({**payload["crash_sim"],"timestamp":t})
        if "var_stats"      in payload: VAR_STORE.update({**payload["var_stats"],"timestamp":t})
        if "correlation"    in payload: CORR_STORE.update({**payload["correlation"],"timestamp":t})
        if "current_prices" in payload: PRICES.update({**payload["current_prices"],"timestamp":t})

    elif typ == "voice_alert":
        VOICE_ALERT.update(payload)

    return {"status": "ok"}

@app.get("/data")
def get_data():
    real = [REAL_DATA[k] for k in sorted(REAL_DATA)]
    pred = [PRED_DATA[k] for k in sorted(PRED_DATA)]
    return real + pred

@app.get("/dashboard")
def dashboard():
    return {
        "risk_score":     RISK_STORE,
        "onchain":        ONCHAIN,
        "solana":         SOLANA,
        "portfolio":      CRASH_SIM,
        "var":            VAR_STORE,
        "correlation":    CORR_STORE,
        "current_prices": PRICES,
        "risk_history":   list(RISK_HIST),
        "voice_alert":    VOICE_ALERT,
        "candles": {
            "real": [REAL_DATA[k] for k in sorted(REAL_DATA)],
            "pred": [PRED_DATA[k] for k in sorted(PRED_DATA)],
        }
    }

@app.get("/risk")
def get_risk(): return {**RISK_STORE,"history":list(RISK_HIST)} if RISK_STORE else {"message":"warming up"}

@app.get("/onchain")
def get_onchain(): return ONCHAIN or {"message":"no data yet"}

@app.get("/solana")
def get_solana(): return SOLANA or {"message":"no data yet"}

@app.get("/portfolio")
def get_portfolio(): return {"portfolio":CRASH_SIM,"prices":PRICES} if CRASH_SIM else {"message":"no data yet"}

@app.get("/var")
def get_var(): return VAR_STORE or {"message":"no data yet"}

@app.get("/scenarios")
def get_scenarios(): return CRASH_SIM.get("scenarios") or {"message":"no data yet"}

@app.get("/alert/status")
def alert_status(): return VOICE_ALERT

@app.get("/alert/audio")
def alert_audio():
    if not os.path.exists("/tmp/alert.mp3"): raise HTTPException(404,"No audio yet")
    return FileResponse("/tmp/alert.mp3", media_type="audio/mpeg")
