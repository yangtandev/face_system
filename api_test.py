from PVMS_Library import config
import json
api=config.API("https://demosite.api.ginibio.com/api/v1", 1)

re = api.face_recognition_in("G15")
#re = api.face_recognition_out("G04")
_, al_staff = api.Get_all_staff_data()
print(al_staff.keys())

#https://demosite.api.ginibio.com/api/v1/media