@echo off
call "%~dp013_run_food_fight.cmd" --resources ample --output "runs\shared-meal\live" --video "runs\shared-meal\demo.mp4" %*
exit /b %errorlevel%
