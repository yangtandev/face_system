import paramiko
import os
import json
import subprocess
import shlex
from scp import SCPClient

config_ = {}
ssh = paramiko.SSHClient()

try:
    with open(os.path.join(os.path.dirname(__file__), "../config.json"), "r", encoding="utf-8") as json_file:
        config_ = json.load(json_file)
except Exception as e:
    print("ssh-載入失敗", e)

def sync_with_rsync(ssh_config, source_dir, dest_dir):
    """
    使用 rsync 透過 SSH 高效地同步遠端資料夾到本地。

    Args:
        ssh_config (dict): 包含 'ip', 'username', 'password' 的字典。
        source_dir (str): 遠端來源資料夾路徑。
        dest_dir (str): 本地目標資料夾路徑。

    Returns:
        bool: True 表示成功，False 表示失敗。
    """
    # 確保本地目標資料夾存在
    os.makedirs(dest_dir, exist_ok=True)

    # 建立 rsync 指令
    # -a: 歸檔模式，保留權限、時間等資訊
    # -v: 顯示詳細過程
    # -z: 壓縮傳輸
    # --delete: 刪除本地多出來的檔案，保持與遠端完全一致
    # -e: 指定使用的 ssh 指令
    command = [
        "rsync",
        "-avz",
        "--delete",
        "-e", f"ssh -p {ssh_config.get('port', 22)}",
        f"{ssh_config['username']}@{ssh_config['ip']}:{source_dir}/", # 注意來源路徑結尾的斜線
        dest_dir
    ]

    print(f"Executing rsync command: {' '.join(command)}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 即時讀取輸出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        stderr = process.communicate()[1]
        if process.returncode != 0:
            print(f"Rsync failed with return code {process.returncode}")
            print(f"Stderr: {stderr}")
            return False

        print("Rsync completed successfully.")
        return True

    except FileNotFoundError:
        print("Error: 'rsync' command not found. Is it installed and in your system's PATH?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during rsync: {e}")
        return False



def connect_server():
    """
    建立 SSH 連線，使用 config.json 中設定的伺服器資訊。
    連線失敗會設定 ssh 為 None。
    """
    global ssh
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname=config_["Server"]["ip"], port=22,
                    username=config_["Server"]["username"], password=config_["Server"]["password"])
        print("SSH連線成功")
    except Exception as e:
        print("SSH連線失敗", e)
        ssh = None

def check_connection():
    """
    檢查目前 SSH 是否已連線（含傳輸層），若無效則嘗試重新連線。
    """
    if ssh is None or not ssh.get_transport() or not ssh.get_transport().is_active():
        print("SSH連線中斷，嘗試重連...")
        connect_server()

def check_file(main_dir, copy_dir, updata=0):
    """
    比對本地與伺服器中的圖檔（.png/.jpg/.jpeg），決定是否下載/上傳/刪除。

    Parameters:
    main_dir (str): 遠端資料夾路徑（如 /home/ubuntu/media/pic_bak）
    copy_dir (str): 本地資料夾路徑
    updata (int): 同步方向，0 表示從伺服器下載、1 表示上傳至伺服器

    Returns:
    new (list): 新增的檔案名稱列表（下載或上傳成功的）
    del_ (list): 被刪除的檔案名稱列表
    """
    check_connection()
    if ssh is None:
        print("無法執行操作，SSH連線失敗")
        return [], []

    print(f"{main_dir} 資料更新到 {copy_dir}")
    stdin, stdout, stderr = ssh.exec_command(f"ls {main_dir}")  #server
    server_file_list = []
    for line in stdout:
        if line.strip('\n').lower().endswith(('.png', '.jpg', '.jpeg')):
            server_file_list.append(line.strip('\n'))

    local_file_list = os.listdir(copy_dir)  #local
    features_name = {}
    new = []
    del_ = []

    if updata == 0:
        for i in local_file_list:
            features_name[i] = 0
        # 遍歷資料夾中的所有圖片
        for filename in server_file_list:
            features_name[filename] = 1
            if not filename in local_file_list:
                new.append(filename)
                with SCPClient(ssh.get_transport()) as scp:
                    scp.get(f'{main_dir}/{filename}', f'{copy_dir}/{filename}')

        for i in features_name.keys():
            if features_name[i] == 0:
                del_.append(i)
                os.remove(os.path.join(copy_dir, i))

    elif updata == 1:  # 上傳
        for i in server_file_list:
            features_name[i] = 0
        for filename in local_file_list:
            features_name[filename] = 1
            if not filename in server_file_list:
                new.append(filename)
                with SCPClient(ssh.get_transport()) as scp:
                    scp.put(f'{copy_dir}/{filename}', f'{main_dir}/{filename}')
        del_str = "rm "
        for i in features_name.keys():
            if features_name[i] == 0:
                del_.append(i)
                del_str += f"{main_dir}/{i} "
        stdin, stdout, stderr = ssh.exec_command(del_str)

    return new, del_

def move_data(main_dir):
    """
    將符合特定檔名規則（含多個底線）的圖檔移動到 `pic_bak` 子資料夾。

    Parameters:
    main_dir (str): 遠端的主資料夾(應包含 pic_bak)
    """
    check_connection()
    if ssh is None:
        print("無法移動資料,SSH連線失敗")
        return

    stdin, stdout, stderr = ssh.exec_command("ls " + main_dir + " | grep -E '^([^_]+_){3,}[^_]+'")  #server
    for line in stdout:
        if line.strip('\n').lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = line.strip('\n')
            stdin, stdout, stderr = ssh.exec_command(f"mv {main_dir}/{filename} {main_dir}/pic_bak")
            print(f"move {filename}")

if __name__ == "__main__":
    # 載入設定並連接到伺服器
    connect_server()
