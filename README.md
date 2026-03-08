# Cryptorisk
CryptoRISK is a real-time crypto risk intelligence platform that combines AI prediction, market analytics, and blockchain infrastructure data to help traders and analysts understand market risk and make safer trading decisions.

The system integrates live crypto market data, AI-generated price predictions, on-chain metrics, and portfolio risk simulations into a unified dashboard.

Features
Real-Time Market Monitoring

Live BTC market candles

Technical indicators (RSI, EMA20, SMA50, VWAP)

Buy/Sell signal detection

AI Price Prediction

Predicts future BTC price movements

Forecasts next market candles

Combines historical and predicted data in charts

Crypto Risk Scoring

Composite risk model based on:

Fear & Greed sentiment

Funding rates

Market volatility

RSI momentum

Bollinger Band width

Blockchain network activity

Blockchain Integration

Uses Solana network data to monitor infrastructure health:

TPS (Transactions per Second)

Network load

Slot and epoch progress

These signals contribute to the overall crypto risk score.

Portfolio Risk Analysis

Value at Risk (VaR)

Conditional VaR (CVaR)

Worst case loss estimation

Historical crash simulations

Simulated scenarios include:

COVID Crash 2020

LUNA Collapse 2022

FTX Collapse 2022

2018 Crypto Bear Market

Flash Crash events

Correlation Risk Monitoring

Multi-asset correlation matrix

Crash-time correlation amplification detection

Portfolio diversification risk analysis

Interactive Trading Dashboard

Real-time candlestick charts

AI prediction overlay

Risk score visualization

Portfolio stress simulation panels

System Architecture

The system consists of three core components:

Market Data Sources
        │
        ▼
Prediction + Risk Engine
        │
        ▼
FastAPI Backend API
        │
        ▼
Live Dashboard (HTML/CSS + Plotly)
Backend

FastAPI server responsible for:

collecting live market data

storing prediction results

computing risk metrics

serving dashboard API

AI Prediction Engine

Machine learning model generates:

future BTC candles

technical indicators

trading signals

Frontend Dashboard

A real-time terminal interface showing:

charts

risk scores

portfolio analytics

blockchain metrics

Tech Stack
Backend

Python

FastAPI

REST APIs

AI / ML

LSTM price prediction model

Technical indicator analysis

Risk aggregation model

Blockchain Data

Solana RPC metrics

Frontend

HTML

CSS

Plotly.js

JavaScript
<img width="1918" height="870" alt="quantpro" src="https://github.com/user-attachments/assets/f379875a-345a-453e-8e73-735c87469bc2" />
<img width="1917" height="851" alt="rsi alert" src="https://github.com/user-attachments/assets/153eb9ed-ae40-497e-87a2-d4dd5a2c7a9e" />

