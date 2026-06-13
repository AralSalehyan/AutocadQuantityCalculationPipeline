@echo off
setlocal

cd /d C:\Users\arals\Desktop\CADQuantityCalcAutonom
if not exist logs mkdir logs

set OUT_LOG=logs\segformer_room_wall.out.log
set ERR_LOG=logs\segformer_room_wall.err.log

echo [%DATE% %TIME%] Resuming SegFormer room/wall training.>> "%OUT_LOG%"
C:\Users\arals\anaconda3\python.exe -u src\training\train_segformer.py --data datasets\room_wall\data.yaml --output models\segformer\room_wall --resume models\segformer\room_wall --epochs 20 --batch 2 --imgsz 512 --device cuda --save-every 1 --num-workers 0 >> "%OUT_LOG%" 2>> "%ERR_LOG%"
set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] SegFormer resume exited with code %EXIT_CODE%.>> "%OUT_LOG%"
exit /b %EXIT_CODE%
