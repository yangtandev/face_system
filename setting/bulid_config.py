import json
import os

# 預設的設定資料
default_config = {
    "ip_set": {
        "ip_address": "192.168.31.130",
        "ip_gateway": "192.168.31.254",
        "ip_mask": "255.255.255.0"
    },
    "cameraIP": {
        "in_camera": "rtsp://admin:!QAZ87518499@192.168.31.101:554",
        "out_camera": "rtsp://admin:!QAZ87518499@192.168.31.59:554"
    },
    "Server": {
        "ip": "54.92.8.82",
        "username": "ubuntu",
        "password": "875184991qaz2wsx",
        "face_data_dir": "/home/ubuntu/pvms-api/media",
        "API_url": "https://demosite.api.ginibio.com/api/v1",
        "location_ID": 1
    },
    "say": {
        "in": "\u7c3d\u5230",
        "out": "\u7c3d\u96e2",
        "clothes": "\u8acb\u6b63\u78ba\u8457\u88dd"
    },
    "inCamera":{
        "close":False
    },
    "outCamera":{
        "close":False
    },
    "door": "0",
    "Clothes_detection": False,
    "Clothes_show": False,
    "min_face": 300,
    "test_mod": False,
    "auto_open": False,
    "full_screen": False
}

def merge_config(default, current):
    """合併 default 與 current，回傳更新後的 config"""
    updated = {}
    for key, value in default.items():
        if key in current:
            if isinstance(value, dict):
                updated[key] = merge_config(value, current[key])
            else:
                updated[key] = current[key]
        else:
            updated[key] = value  # 新增缺少的 key

    return updated  # 移除多餘的 key（不包含於 default）

def update_config_file(config_path='config.json'):
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"已建立預設設定檔: {config_path}")
        return

    # 讀取現有 config 並與 default 比對
    with open(config_path, 'r', encoding='utf-8') as f:
        current_config = json.load(f)

    updated_config = merge_config(default_config, current_config)

    # 檢查是否有多餘的 key 並移除
    def prune_extra_keys(default, current):
        if isinstance(default, dict):
            return {k: prune_extra_keys(default[k], current[k]) for k in default if k in current}
        else:
            return current

    final_config = prune_extra_keys(default_config, updated_config)

    # 寫回更新後的 config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, ensure_ascii=False, indent=2)
    print(f"已更新設定檔: {config_path}")

if __name__ == "__main__":
    # 使用範例
    update_config_file("config.json")
