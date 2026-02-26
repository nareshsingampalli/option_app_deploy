# Local Deployment Script for PowerShell
git add .
git commit -m "Auto-deploy: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
git push origin crudeoil

ssh -i C:\Users\vgnsi\Downloads\test\myvm.key -o StrictHostKeyChecking=no ubuntu@140.245.8.242 "/home/ubuntu/option_app/deploy.sh"
