import requests, json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta


def get_image_ids(date, api_key, env="q1bdc3of"):
    headers = {"apikey": api_key}
    url = f"https://{env}.ouest-france.fr/media/photos"
    query = {"date": date}
    payload = {"page": "tout"}
    payload["q"] = json.dumps(query, separators=(",", ":"))  # compact encoding
    print(payload)
    req = requests.get(url, params=payload, headers=headers)
    images = [img["id"] for img in req.json()["hits"]]
    return images


def download_images(date, api_key, outpath, env="q1bdc3of"):
    headers = {"apikey": api_key}
    p = Path(outpath)
    p.mkdir(parents=True, exist_ok=True)
    img_ids = get_image_ids(date, api_key, env)
    for img_id in tqdm(img_ids):
        url = f"https://{env}.ouest-france.fr/media/photo/{img_id}"
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            outfile = p / f"{img_id}.jpg"
            with outfile.open("wb") as f:
                f.write(r.content)


def dates_between(d1, d2):
    fmt = "%Y-%m-%d"
    d1 = datetime.strptime(d1, fmt)
    d2 = datetime.strptime(d2, fmt)
    delta = d2 - d1
    return [str((d1 + timedelta(days=i)).date()) for i in range(delta.days + 1)]


def download_images_dates(start_day, end_day, api_key, outpath, env="q1bdc3of"):
    for day in dates_between(start_day, end_day):
        download_images(day, api_key, outpath, env="q1bdc3of")


if __name__ == "__main__":
    import fire

    fire.Fire()


