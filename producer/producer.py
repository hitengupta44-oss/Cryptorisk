import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import ta
from binance.client import Client

# ===== Flask keep-alive (Render Web Service) =====
from flask import Flask
import threading

app = Flask(__name__)

@app.route("/")
def home():
    return "Producer running"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask).start()

# ================= CONFIG =================

BACKEND_URL  = "https://cryptorisk-production.up.railway.app/update"
BACKEND_HOME = "https://cryptorisk-production.up.railway.app/"
SERVICE_URL  = "https://producer-production-2e4b.up.railway.app/"

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
LOOKBACK = 60
PRED_MINUTES = 10
RETRAIN_INTERVAL = 1800  # 30 min

client = Client()

print("Producer started — Stable MT5 Model")

model = None
scaler = MinMaxScaler()
last_candle_time = None
last_train_time = None

# Keep-alive timers
last_self_ping = 0
last_backend_ping = 0

# ================= MODEL =================

def build_model(n_features):
    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ================= FETCH DATA =================

def fetch_data():
    klines = client.get_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=500
    )

    df = pd.DataFrame(klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','taker_base','taker_quote','ignore'
    ])

    df['time'] = pd.to_datetime(df['open_time'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)

    return df[['time','open','high','low','close','volume']]

# ===================================================================
#  NEW — Risk engine config  (✏️ edit these 3 lines only)
# ===================================================================
ELEVENLABS_API_KEY  = "your_elevenlabs_api_key_here"   # ✏️ edit
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
PORTFOLIO = {"BTCUSDT": 0.5, "ETHUSDT": 3.0, "SOLUSDT": 20.0, "BNBUSDT": 5.0}

# ===== Risk engine state =====
ASSETS       = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
SOLANA_RPC   = "https://api.mainnet-beta.solana.com"
last_voice_t = 0
last_level   = "LOW"
ret_hist     = {a: [] for a in ASSETS}

CRASH_SCENARIOS = {
    "COVID_2020": {"drop": -0.63, "duration_days": 2,   "recovery_days": 60},
    "LUNA_2022":  {"drop": -0.99, "duration_days": 7,   "recovery_days": None},
    "FTX_2022":   {"drop": -0.25, "duration_days": 3,   "recovery_days": 180},
    "BEAR_2018":  {"drop": -0.84, "duration_days": 365, "recovery_days": 730},
    "FLASH_2021": {"drop": -0.30, "duration_days": 1,   "recovery_days": 14},
}

# ===== Risk engine helpers =====
def _fetch_sym(symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval=INTERVAL, limit=300)
        df = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','qav','trades','taker_base','taker_quote','ignore'])
        df['time'] = pd.to_datetime(df['open_time'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df[['time','open','high','low','close','volume']]
    except Exception as e:
        print(f"[{symbol}] {e}"); return None

def _add_indicators(df):
    df = df.copy()
    df["EMA20"]    = df["close"].ewm(span=20, adjust=False).mean()
    df["SMA50"]    = df["close"].rolling(50).mean()
    df["RSI"]      = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["VWAP"]     = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["RET"]      = df["close"].pct_change()
    bb             = ta.volatility.BollingerBands(df["close"], window=20)
    df["BB_WIDTH"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["ATR"]      = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14).average_true_range()
    return df.dropna()

def _fear_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()["data"][0]
        v = int(d["value"])
        return {"value": v, "label": d["value_classification"],
                "risk_signal": "HIGH" if v > 75 else "MEDIUM" if v > 55 else "LOW"}
    except: return {"value": 50, "label": "Neutral", "risk_signal": "LOW"}

def _funding():
    try:
        r = float(requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1",
            timeout=5).json()[0]["fundingRate"])
        return {"rate": r, "rate_pct": round(r*100, 4),
                "risk_signal": "HIGH" if abs(r)>0.001 else "MEDIUM" if abs(r)>0.0005 else "LOW"}
    except: return {"rate": 0, "rate_pct": 0, "risk_signal": "LOW"}

def _open_interest():
    try:
        oi = float(requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
            timeout=5).json()["openInterest"])
        return {"open_interest": oi}
    except: return {"open_interest": 0}

def _solana():
    r = {"source": "Solana Blockchain", "current_tps": 0,
         "network_load_pct": 0, "slot": 0, "epoch": 0,
         "epoch_progress_pct": 0, "risk_signal": "LOW"}
    try:
        s = requests.post(SOLANA_RPC, json={"jsonrpc":"2.0","id":1,
            "method":"getRecentPerformanceSamples","params":[3]}, timeout=8).json().get("result",[])
        if s:
            tps = sum(x["numTransactions"]/x["samplePeriodSecs"] for x in s)/len(s)
            r["current_tps"]      = round(tps, 1)
            r["network_load_pct"] = round(min(tps/65000*100, 100), 2)
        sl = requests.post(SOLANA_RPC, json={"jsonrpc":"2.0","id":2,"method":"getSlot"}, timeout=8).json()
        r["slot"] = sl.get("result", 0)
        ei = requests.post(SOLANA_RPC, json={"jsonrpc":"2.0","id":3,"method":"getEpochInfo"}, timeout=8).json().get("result",{})
        r["epoch"]              = ei.get("epoch", 0)
        r["epoch_progress_pct"] = round(ei.get("slotIndex",0)/max(ei.get("slotsInEpoch",1),1)*100, 1)
        r["risk_signal"] = "HIGH" if r["current_tps"]>45000 else "MEDIUM" if r["current_tps"]>25000 else "LOW"
    except Exception as e: print(f"[Solana] {e}")
    return r

def _risk_score(fg, fr, vol, rsi, bb, sol):
    score = round(
        fg["value"]*0.23 +
        min(abs(fr["rate"])/0.002*100, 100)*0.22 +
        min(vol/0.03*100, 100)*0.20 +
        min(abs(rsi-50)/50*100, 100)*0.15 +
        min(bb/0.1*100, 100)*0.10 +
        sol.get("network_load_pct", 0)*0.10, 2)
    return {
        "score": score,
        "level": "CRITICAL" if score>75 else "HIGH" if score>55 else "MEDIUM" if score>35 else "LOW",
        "components": {
            "fear_greed":     round(fg["value"], 2),
            "funding":        round(min(abs(fr["rate"])/0.002*100, 100), 2),
            "volatility":     round(min(vol/0.03*100, 100), 2),
            "rsi":            round(min(abs(rsi-50)/50*100, 100), 2),
            "bb_width":       round(min(bb/0.1*100, 100), 2),
            "solana_network": round(sol.get("network_load_pct", 0), 2),
        }
    }

def _crashes(prices):
    total = sum(PORTFOLIO.get(a,0)*prices.get(a,0) for a in ASSETS)
    sc = {}
    for n, p in CRASH_SCENARIOS.items():
        cr = total*(1+p["drop"])
        sc[n] = {"portfolio_value_now": round(total,2),
                 "portfolio_value_crashed": round(cr,2),
                 "loss_usd": round(total-cr,2),
                 "loss_pct": round(p["drop"]*100,2),
                 "duration_days": p["duration_days"],
                 "recovery_days": p["recovery_days"]}
    return {"scenarios": sc, "total_portfolio_usd": round(total,2)}

def _var(vol, usd):
    r = np.prod(1+np.random.default_rng().normal(0,vol,(5000,10)), axis=1)-1
    v = float(np.percentile(r,5)); cv = float(np.mean(r[r<=v]))
    return {"VaR_pct":round(v*100,2), "CVaR_pct":round(cv*100,2),
            "VaR_usd":round(v*usd,2), "CVaR_usd":round(cv*usd,2),
            "worst_pct":round(float(r.min())*100,2)}

def _corr(rd):
    tickers = list(rd.keys())
    if len(tickers) < 2: return {}
    mat = pd.DataFrame(rd).corr().round(3)
    n   = len(tickers)
    avg = float(mat.values[np.triu_indices(n,k=1)].mean())
    return {"normal_avg_correlation":round(avg,3), "crash_correlation":0.95,
            "amplification_factor":round(0.95/max(abs(avg),0.01),2),
            "correlation_matrix":mat.to_dict(),
            "warning":f"Correlations surge {round(avg,2)} → 0.95 in crashes"}

def _voice(rs, var, fg, sol):
    global last_voice_t, last_level
    level = rs["level"]
    if time.time()-last_voice_t < 600: return
    if level=="CRITICAL" and last_level!="CRITICAL":
        txt=(f"Critical warning. Crash risk score is {int(rs['score'])} out of 100. "
             f"Fear greed is {fg['value']}, {fg['label']}. "
             f"Portfolio VaR is {abs(var['VaR_pct'])} percent. Review immediately.")
    elif level=="HIGH" and last_level not in("HIGH","CRITICAL"):
        txt=f"Risk alert. Score risen to {int(rs['score'])}. Review your positions."
    else: last_level=level; return
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY=="your_elevenlabs_api_key_here":
        last_level=level; return
    try:
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={"xi-api-key":ELEVENLABS_API_KEY,"Content-Type":"application/json"},
            json={"text":txt,"model_id":"eleven_monolingual_v1",
                  "voice_settings":{"stability":0.55,"similarity_boost":0.80}}, timeout=15)
        if resp.status_code==200:
            open("/tmp/alert.mp3","wb").write(resp.content)
            requests.post(BACKEND_URL, json={"type":"voice_alert","text":txt,
                "audio_ready":True,"time":pd.Timestamp.now().isoformat()}, timeout=5)
            print(f"🔊 {txt[:60]}…")
            last_voice_t = time.time()
    except Exception as e: print(f"[ElevenLabs] {e}")
    last_level = level

# ================= MAIN LOOP =================

while True:
    try:
        # ===== Render Keep-Alive Fix =====
        # Ping self every 5 minutes
        if time.time() - last_self_ping > 300:
            try:
                requests.get(SERVICE_URL, timeout=5)
                print("Self ping successful")
            except:
                print("Self ping failed")
            last_self_ping = time.time()

        # Ping backend home every 5 minutes
        if time.time() - last_backend_ping > 300:
            try:
                requests.get(BACKEND_HOME, timeout=5)
                print("Backend keep-alive ping")
            except:
                print("Backend ping failed")
            last_backend_ping = time.time()

        df = fetch_data()

        # ===== Safety check =====
        if df is None or len(df) < 100:
            print("Data fetch issue, retrying...")
            time.sleep(5)
            continue

        # Use LAST CLOSED candle
        current_time = df.iloc[-2]["time"]

        if last_candle_time == current_time:
            print("Waiting for new candle...")
            time.sleep(5)
            continue

        last_candle_time = current_time
        print("New closed candle:", current_time)

        # ===== Indicators =====
        df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["RET"] = df["close"].pct_change()

        df = df.dropna()

        features = ["RET","EMA20","SMA50","RSI","VWAP"]
        scaled = scaler.fit_transform(df[features])

        # ===== Sequences =====
        X, y = [], []
        for i in range(LOOKBACK, len(saled := scaled)):
            X.append(saled[i-LOOKBACK:i])
            y.append(saled[i,0])

        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            print("Not enough data for training")
            time.sleep(5)
            continue

        # ===== Train =====
        if model is None:
            model = build_model(len(features))
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last_train_time = time.time()

        if time.time() - last_train_time > RETRAIN_INTERVAL:
            model.fit(X, y, epochs=1, batch_size=32, verbose=0)
            last_train_time = time.time()

        # ===== Send last 60 real =====
        real60 = df.tail(60).reset_index(drop=True)

        for i, row in real60.iterrows():
            signal = None
            if i > 0:
                prev = real60.iloc[i-1]
                if row["EMA20"] > row["SMA50"] and prev["EMA20"] <= prev["SMA50"]:
                    signal = "BUY"
                elif row["EMA20"] < row["SMA50"] and prev["EMA20"] >= prev["SMA50"]:
                    signal = "SELL"

            requests.post(BACKEND_URL, json={
                "time": row["time"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "ema20": float(row["EMA20"]),
                "sma50": float(row["SMA50"]),
                "vwap": float(row["VWAP"]),
                "rsi": float(row["RSI"]),
                "signal": signal,
                "type": "real"
            }, timeout=5)

        # ===== Predictions =====
        volatility = df["RET"].std()
        last_price = df.iloc[-1]["close"]

        last_seq = scaled[-LOOKBACK:]
        temp_df = df.copy()

        for _ in range(PRED_MINUTES):

            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(features)),
                verbose=0
            )[0][0]

            pred_ret = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1,-1)
            )[0][0]

            pred_ret = np.clip(pred_ret, -2.5*volatility, 2.5*volatility)
            pred_price = last_price * (1 + pred_ret)

            ema = temp_df.iloc[-1]["EMA20"]
            pred_price = 0.9 * pred_price + 0.1 * ema

            pred_price *= (1 + np.random.normal(0, volatility/2))

            body = abs(pred_price - last_price)
            wick = max(body*0.5, last_price*volatility*0.2)

            high_p = max(last_price, pred_price) + wick
            low_p = min(last_price, pred_price) - wick

            future_time = temp_df.iloc[-1]["time"] + timedelta(minutes=1)

            new_row = {
                "time": future_time,
                "open": last_price,
                "high": high_p,
                "low": low_p,
                "close": pred_price,
                "volume": temp_df.iloc[-1]["volume"]
            }

            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

            temp_df["EMA20"] = temp_df["close"].ewm(span=20).mean()
            temp_df["SMA50"] = temp_df["close"].rolling(50).mean()
            temp_df["RSI"] = ta.momentum.RSIIndicator(temp_df["close"]).rsi()
            temp_df["VWAP"] = (temp_df["close"] * temp_df["volume"]).cumsum() / temp_df["volume"].cumsum()
            temp_df["RET"] = temp_df["close"].pct_change()

            latest = temp_df.iloc[-1]
            prev = temp_df.iloc[-2]

            signal = None
            if latest["EMA20"] > latest["SMA50"] and prev["EMA20"] <= prev["SMA50"]:
                signal = "BUY"
            elif latest["EMA20"] < latest["SMA50"] and prev["EMA20"] >= prev["SMA50"]:
                signal = "SELL"

            requests.post(BACKEND_URL, json={
                "time": latest["time"].isoformat(),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "ema20": float(latest["EMA20"]),
                "sma50": float(latest["SMA50"]),
                "vwap": float(latest["VWAP"]),
                "rsi": float(latest["RSI"]),
                "signal": signal,
                "type": "prediction"
            }, timeout=5)

            last_price = pred_price
            last_seq = scaler.transform(temp_df[features].tail(LOOKBACK))

        print("Sent 60 real + 10 prediction")

        # =================================================================
        #  RISK ENGINE — runs after original block, completely isolated
        #  Does not touch model / scaler / predictions above in any way
        # =================================================================
        try:
            prices, rframes = {}, {}
            for sym in ASSETS:
                raw = _fetch_sym(sym)
                if raw is not None and len(raw) >= 100:
                    rframes[sym] = _add_indicators(raw)
                    prices[sym]  = float(rframes[sym].iloc[-2]["close"])

            if len(rframes) >= 2:
                fg  = _fear_greed()
                fr  = _funding()
                oi  = _open_interest()
                sol = _solana()

                btc_e   = rframes.get("BTCUSDT", _add_indicators(df))
                vol_now = float(btc_e["RET"].std())
                rsi_now = float(btc_e.iloc[-2]["RSI"])
                bb_now  = float(btc_e.iloc[-2]["BB_WIDTH"]) if "BB_WIDTH" in btc_e.columns else 0.05

                rs  = _risk_score(fg, fr, vol_now, rsi_now, bb_now, sol)
                cs  = _crashes(prices)
                var = _var(vol_now, cs["total_portfolio_usd"])

                for a in ASSETS:
                    if a in rframes:
                        ret_hist[a] = rframes[a]["RET"].tail(60).tolist()
                corr = _corr({s: ret_hist[s] for s in ASSETS if len(ret_hist[s])>=10})

                _voice(rs, var, fg, sol)

                requests.post(BACKEND_URL, json={
                    "type":           "risk_update",
                    "time":           current_time.isoformat(),
                    "risk_score":     rs,
                    "fear_greed":     fg,
                    "funding_rate":   fr,
                    "open_interest":  oi,
                    "solana_onchain": sol,
                    "crash_sim":      cs,
                    "var_stats":      var,
                    "correlation":    corr,
                    "current_prices": prices,
                }, timeout=10)

                print(f"✓ Risk {rs['score']} [{rs['level']}] | SOL TPS {sol.get('current_tps','?')}")

        except Exception as re:
            print(f"[Risk Engine] {re}")
        # =================================================================
        #  END RISK ENGINE
        # =================================================================

    except Exception as e:
        print("Error:", e)

    time.sleep(5)

