@echo off
setlocal

cd /d C:\Users\arals\Desktop\CADQuantityCalcAutonom
if not exist logs mkdir logs

set OUT_LOG=logs\rtdetr_scheduled_1024_20260606.out.log
set ERR_LOG=logs\rtdetr_scheduled_1024_20260606.err.log

echo [%DATE% %TIME%] Starting RT-DETR resume from epoch 16.>> "%OUT_LOG%"
python -u src\training\train_rtdetr.py --model runs\detect\models\rtdetr\runs\door_window_1024\weights\last.pt --resume --imgsz 1024 --device 0 --batch 1 --workers 0 --no-plots >> "%OUT_LOG%" 2>> "%ERR_LOG%"
set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] RT-DETR resume exited with code %EXIT_CODE%.>> "%OUT_LOG%"
exit /b %EXIT_CODE%
