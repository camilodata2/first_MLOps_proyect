import os
from base64 import b64encode

def archivo():
    key=os.environ.get("SERVICE_ACCOUNT_KEY")
    with open('path.json','w',encoding='utf8') as json_file:
        json_file.write(b64encode(key).decode())
        print(os.path.realpath('path.json'))
        if __name__== '__main__':
            archivo()