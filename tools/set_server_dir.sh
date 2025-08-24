#!/bin/bash

# 檢查參數數量
if [ "$#" -ne 4 ]; then
  echo "❗ 使用方式: $0 [user@host] [password] [target_folder_path] [shortcut_name]"
  echo "👉 範例: $0 user@192.168.1.100 mypassword /home/user/project shortcut_to_project"
  exit 1
fi

REMOTE="$1"
PASSWORD="$2"
TARGET_FOLDER="$3"
LINK_NAME="$4"

echo "🔗 使用 sshpass 自動連線到 $REMOTE..."

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$REMOTE" <<EOF
  mkdir -p "$LINK_NAME"
  ln -sf "$TARGET_FOLDER" "\$HOME/$LINK_NAME"
  echo "✅ 已建立捷徑：\$HOME/$LINK_NAME -> $TARGET_FOLDER"
EOF
