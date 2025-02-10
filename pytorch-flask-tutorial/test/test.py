import requests

resp = requests.post("http://localhost:5000/predict", files={'file': open('8.png', 'rb')})

print(resp.text)