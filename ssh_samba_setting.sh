echo "[正在執行遠端主機共用資料夾設定bash檔]\n"

# 1. 安裝ssh連線套件
echo "\e[43m1.安裝ssh直接輸入密碼套件\e[0m"
sudo apt-get install sshpass

# 2. 透過ssh連線遠端修改samba設定
echo "\n正透過SSH連線遠端主機並更改samba設定..."
echo "\e[43m2. 遠端主機連線並寫入samba設定\e[0m"
echo -n "請輸入欲連線的遠端主機IP: "
read remote_device_ip
echo -n "請輸入遠端主機帳號(預設為ubuntu，按下enter套用預設值): "
read remote_device_account
remote_device_account=${remote_device_account:-"ubuntu"}
echo -n "請輸入遠端主機密碼(預設為875184991qaz2wsx，按下enter套用預設值): "
read remote_device_password
remote_device_password=${remote_device_password:-"875184991qaz2wsx"}
# sshpass -p '875184991qaz2wsx' ssh ubuntu@54.238.24.207 "

# 3. 修改samba設定並創件samba使用者
sshpass -p "$remote_device_password" ssh "$remote_device_account@$remote_device_ip" "
api_directory=\$(find /home/ubuntu -type d -name '*pvms*api*')
file_path=\"/etc/samba/smb.conf\"
samba_setting=\"
[FileSystem]
path=\$api_directory/media
browseable = yes
read only = no
writable = yes
valid users = $remote_device_account
guest ok = no
\" 
echo -e \"\n\e[33mSamba setting:\$samba_setting\e[0m\"
echo \"Remote samba setting file path = \$file_path\"
if ! grep -qF \"\$samba_setting\" \"\$file_path\"; then
    echo -e \"\$samba_setting\" | sudo tee -a \"\$file_path\"
    echo ' >> 寫入完成\n'
else
    echo -e ' >> 此設定檔已被寫入過遠端主機smb.conf中 。 \n(samba command already exist in smb.conf)\n'
fi  

echo -e '\e[43m3.新增samba User，共用資料夾透過這組帳密連線\e[0m'
echo '(需要輸入帳號及密碼，通常會使用ubuntu/875184991qaz2wsx)'
while true; do
    echo -e -n \"請輸入samba user帳號： \"
    user_account=\"\"
    read user_account
    # 檢查變數是否為空
    if [ -z \"\$user_account\" ]; then
        echo \"帳號不能為空，請重新輸入。\"
    else
        break
    fi
done
sudo smbpasswd -a \$user_account
echo '列出新增的samba user profile'
sudo pdbedit -L -v
echo '設定完成.'
"

