LOGFILE="/home/$LOGNAME/open_main.log"
MAXSIZE=10485760  # 10MB

if [ -f "$LOGFILE" ]; then
    FILESIZE=$(stat -c%s "$LOGFILE")

    if [ "$FILESIZE" -gt "$MAXSIZE" ]; then
        echo "日誌文件過大，正在進行截斷。" >> $LOGFILE
        cp /dev/null $LOGFILE  # 清空日誌文件
    fi
fi

LOGFILE_="/home/set_.log"
MAXSIZE=10485760  # 10MB

if [ -f "$LOGFILE_" ]; then
    FILESIZE=$(stat -c%s "$LOGFILE_")

    if [ "$FILESIZE" -gt "$MAXSIZE" ]; then
        echo "日誌文件過大，正在進行截斷。" >> $LOGFILE_
        cp /dev/null $LOGFILE_  # 清空日誌文件
    fi
fi