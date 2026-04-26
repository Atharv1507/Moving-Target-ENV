import urllib.request
import json
import urllib.error
try:
    req = urllib.request.urlopen("https://huggingface.co/spaces/Atharv0707/Moving_Target")
    print(req.read().decode('utf-8'))
except urllib.error.URLError as e:
    print(e)
    try:
        print(e.read().decode('utf-8'))
    except:
        pass
