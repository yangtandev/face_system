import paramiko, os
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
import numpy as np
from scp import SCPClient

face_img_dir = "/home/ubuntu/.pvms-api-release/media/crop_face/pic_bak"
local_img_dir = "E:/eric/face_d/download_img_test"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname="18.178.116.237", port=22, username="ubuntu", password="875184991qaz2wsx")

stdin, stdout, stderr = ssh.exec_command(f"ls {face_img_dir}")
img_list = []
for line in stdout:
    #print(line.strip('\n'))
    if line.strip('\n').lower().endswith(('.png', '.jpg', '.jpeg')):
        img_list.append(line.strip('\n'))

feature_list = os.listdir(local_img_dir)
features_name = {}
new = 0
del_ = 0
for i in os.listdir(local_img_dir):
    features_name[i] = 0

# 遍歷資料夾中的所有圖片
for filename in img_list:
    features_name[filename] = 1
    if not filename in feature_list:
        new += 1
        with SCPClient(ssh.get_transport()) as scp:
            scp.get(f'{face_img_dir}/{filename}', f'{local_img_dir}/{filename}')
        """ image_path = os.path.join(local_img_dir, filename)
        image = Image.open(image_path).convert('RGB')
        img_cropped = mtcnn(image, save_path=f"E:/eric/face_d/face/just/{filename}") #, save_path=<optional save path>
        if img_cropped != None:
            new += 1
            img_embedding1 = resnet(img_cropped.unsqueeze(0))
            np.save(f'./feature/{filename.split(".")[0]}.npy', img_embedding1[0].detach().numpy())
            features_name[feature_name] = 1 """

for i in features_name.keys():
    if features_name[i] == 0:
        del_ += 1
        os.remove(os.path.join(local_img_dir, i))


print(f"人臉資料更新, 新增{new}, 刪除{del_}")

ssh.close()
