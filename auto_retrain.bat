@echo off
cd /d "C:\loan default mlops"

echo ========================================
echo AUTOMATIC LOAN MODEL RETRAINING
echo ========================================
echo.

python src\retrain.py

echo.
echo ========================================
echo RETRAINING CHECK FINISHED
echo ========================================