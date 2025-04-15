import pandas
import requests

d = []
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0"
}
df = pandas.read_excel('data.xlsx')
path = 'data'
for i in df.index[:]:
    item = dict(df.loc[i, :])
    print(item['main_image_url'])
    while True:
        try:
            response = requests.get(item['main_image_url'], headers=headers, timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
    with open(f'{path}/{i}_1.jpg', 'wb') as f:
        f.write(response.content)
    item['main_image_url'] = f'{path}/{i}_1.jpg'
    print(item['suggestion_1_image'])
    while True:
        try:
            response = requests.get(item['suggestion_1_image'], headers=headers, timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
    with open(f'{path}/{i}_2.jpg', 'wb') as f:
        f.write(response.content)
    print(item['suggestion_2_image'])
    item['suggestion_1_image'] = f'{path}/{i}_2.jpg'
    if pandas.isna(item['suggestion_2_image']):
        continue
    while True:
        try:
            response = requests.get(item['suggestion_2_image'], headers=headers, timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
    with open(f'{path}/{i}_3.jpg', 'wb') as f:
        f.write(response.content)
    item['suggestion_2_image'] = f'{path}/{i}_3.jpg'
    print(item)
    d.append(item)
pandas.DataFrame(d).to_excel('data2.xlsx', index=False)
