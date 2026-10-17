@echo off
cd /d "%~dp0"
python -m streamlit run ecg_interface_streamlit.py
pause
