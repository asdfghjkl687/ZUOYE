@echo off
chcp 65001
echo ==============================================
echo 运动损伤风险预测项目 - GitHub 推送脚本
echo ==============================================
echo.

:: 配置信息
set "GIT_USERNAME=asdfghjkl687"
set "GIT_EMAIL=your_email@example.com"
set "REPO_URL=https://github.com/asdfghjkl687/ZUOYE.git"
set "COMMIT_MESSAGE=feat: 添加完整项目文件"

echo 1. 配置 Git 用户信息...
git config --global user.name "%GIT_USERNAME%"
git config --global user.email "%GIT_EMAIL%"
echo    配置完成！
echo.

echo 2. 检查当前目录...
cd /d "d:\wenjian\Trae\SJWJDZY"
echo    当前目录: %cd%
echo.

echo 3. 初始化 Git 仓库（如果需要）...
if not exist ".git" (
    git init
    echo    仓库初始化完成！
) else (
    echo    仓库已存在，跳过初始化
)
echo.

echo 4. 添加远程仓库...
git remote add origin "%REPO_URL%" 2>NUL || echo    远程仓库已存在，跳过添加
echo    远程仓库配置完成！
echo.

echo 5. 添加所有文件...
git add .
echo    文件添加完成！
echo.

echo 6. 提交到本地仓库...
git commit -m "%COMMIT_MESSAGE%"
echo    提交完成！
echo.

echo 7. 推送到 GitHub...
git push -u origin main
echo    推送完成！
echo.

echo ==============================================
echo 操作完成！请检查 GitHub 仓库确认文件已上传
echo ==============================================
pause
