@echo off
title Axon Research Engine
echo Starting Pro Research Suite...
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run streamlit run frontend.py
) else (
    python -m streamlit run frontend.py
)
pause
