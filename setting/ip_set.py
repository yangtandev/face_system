
import yaml,os
path = "/etc/netplan/01-network-manager-all.yaml"
password = "87518499"

def sudoCMD(command,password):
    str = os.system('echo %s | sudo -S %s' % (password,command))
    print("CMD:",str)
    return str
 
 
sudoCMD('chmod 557 '+ path, password)
 
def add_dict(name, ip, gateway):
    data = {'network':{}}
    data['network']['version'] = 2
    data['network']['renderer'] = "networkd"
    data['network']['ethernets'] = {}
    data['network']['ethernets'][name] = {'addresses':[ip],'dhcp4':'no','optional':"true",
                     'routes':[{"to":"default", "via":gateway}],'nameservers':{'addresses':['8.8.8.8']}}

    file = open(path, 'w', encoding='utf-8')
    yaml.dump(data, file)
    file.close()
 
def set_ip(ip, netmask, gateway):
    result = os.popen("ip -o link show | awk -F': ' '/^[0-9]+: en/ {print $2}'",'r')
    res = result.read()
    for line in res.splitlines():
        name = line
    result.close()
    print(name)
    result = ""
    for num in netmask.split('.'):
        temp = str(bin(int(num)))[2:]
        result = result + temp
    mask_len = len("".join(str(result).split('0')[0:1]))
    ip = f"{ip}/{mask_len}"
    add_dict(name, ip, gateway)
    return sudoCMD('sudo netplan apply', password)