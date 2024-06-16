import argparse
import os
import datetime
import zipfile
import requests
from tqdm import tqdm
from decouple import config
from datetime import datetime, timedelta
import os
import subprocess


def unzip_file(zip_file_path, extract_to_dir):
    """
    Unzips a file into a specific directory.

    :param zip_file_path: Path to the zip file.
    :param extract_to_dir: Directory to extract the contents to.
    """
    # Ensure the extraction directory exists
    os.makedirs(extract_to_dir, exist_ok=True)

    # Open the zip file and extract all contents
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
    print(f"Unzipped {zip_file_path} to {extract_to_dir}")


def download_sentinel_product(id, copernicus_dataspace_login, copernicus_dataspace_passwd, image_path):
    access_token = get_access_token(copernicus_dataspace_login, copernicus_dataspace_passwd)

    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value"

    headers = {"Authorization": f"Bearer {access_token}"}

    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, headers=headers, stream=True)

    total_size = int(response.headers.get('content-length', 0))

    with open(image_path, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def get_available_sentinel_products(aoi, start_date, end_date):
    data_collection = "SENTINEL-2"
    json = requests.get(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 1.00)"
    )
    return json.json()


def geojson_to_wkt_bbox(geojson):
    if geojson['type'] != 'Polygon':
        raise ValueError("O GeoJSON fornecido não é um 'Polygon'")

    coordinates = geojson['coordinates'][0]  # Assumindo que não há buracos no polígono

    # Encontrando os limites do bounding box
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for x, y in coordinates:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    # Criando a representação WKT do bounding box
    wkt = f"POLYGON(({min_x} {min_y}, {min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"

    return wkt



def car_to_geojson(car, api_root, api_token):
    print('Getting geometry from CAR')
    url = f'{api_root}/api/data/environmental/car/imovel/codigo-imovel/{car}'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token ' + api_token
    }
    response = requests.get(url, headers=headers)
    return response.json()['data']['geometry']


def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]



def run_canopy_prediction(image_path):
    DEPLOY_IMAGE_PATH = image_path
    GCHM_DEPLOY_DIR = "./deploy_example/predictions/"
    GCHM_MODEL_DIR = "./trained_models/GLOBAL_GEDI_2019_2020"
    GCHM_NUM_MODELS = "5"
    filepath_failed_image_paths = "./deploy_example/log_failed.txt"
    GCHM_DOWNLOAD_FROM_AWS = "False"
    GCHM_DEPLOY_SENTINEL2_AWS_DIR = "./deploy_example/sentinel2_aws"

    os.makedirs(GCHM_DEPLOY_DIR, exist_ok=True)
    os.makedirs(GCHM_DEPLOY_SENTINEL2_AWS_DIR, exist_ok=True)

    command = [
        "python3", "gchm/deploy.py",
        "--model_dir", GCHM_MODEL_DIR,
        "--deploy_image_path", DEPLOY_IMAGE_PATH,
        "--deploy_dir", GCHM_DEPLOY_DIR,
        "--deploy_patch_size", "512",
        "--num_workers_deploy", "4",
        "--num_models", GCHM_NUM_MODELS,
        "--finetune_strategy", "FT_Lm_SRCB",
        "--filepath_failed_image_paths", filepath_failed_image_paths,
        "--download_from_aws", GCHM_DOWNLOAD_FROM_AWS,
        "--sentinel2_dir", GCHM_DEPLOY_SENTINEL2_AWS_DIR,
        "--remove_image_after_pred", "False"
    ]

    subprocess.run(command)


def run_merge_predictions(tile_name):
    GCHM_DEPLOY_DIR = "./deploy_example/predictions/"
    reduction = "inv_var_mean"
    out_dir = f"{GCHM_DEPLOY_DIR}_merge/preds_{reduction}/"
    out_type = "uint8"
    nodata_value = 255
    from_aws = False
    finetune_strategy = "sua_estrategia_de_finetune"  # Substitua com o valor apropriado
    os.makedirs(out_dir, exist_ok=True)

    print("*************************************")
    print(f"merging... {reduction}")
    print("reduction:", reduction)
    print("out_dir:", out_dir)

    # Executando o script Python
    command = [
        "python3",
        "gchm/merge_predictions_tile.py",
        f"--tile_name={tile_name}",
        f"--out_dir={out_dir}",
        f"--deploy_dir={GCHM_DEPLOY_DIR}",
        f"--out_type={out_type}",
        f"--nodata_value={nodata_value}",
        f"--from_aws={from_aws}",
        f"--reduction={reduction}",
        f"--finetune_strategy={finetune_strategy}"
    ]

    subprocess.run(command)


def main(car):
    api_root = 'https://api.scicrop.com.br'
    api_token = config('TOKEN')
    copernicus_dataspace_login = config('COPERNICUS_DATASPACE_LOGIN')
    copernicus_dataspace_passwd = config('COPERNICUS_DATASPACE_PASSWD')
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=6 * 30)
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")
    print("start_date:", start_date_str)
    print("end_date:", end_date_str)
    output_path = './deploy_example/sentinel2/'

    json = car_to_geojson(car, api_root, api_token)
    wkt = geojson_to_wkt_bbox(json)
    json = get_available_sentinel_products(wkt, start_date_str, end_date_str)

    if json is not None and len(json['value']) > 0:
        for item in json['value']:
            print(item['Name'], item['OriginDate'])
        print(f"Starting to download file {json['value'][0]['Name']}.")
        image_path = output_path + json['value'][0]['Name'] + ".zip"
        tile = json['value'][0]['Name'].split('_T')[1].split('_')[0]
        print(tile)
        #download_sentinel_product(json['value'][0]['Id'], copernicus_dataspace_login, copernicus_dataspace_passwd, image_path)
        #unzip_file(image_path, output_path)
        #run_canopy_prediction(image_path)
    else:
        print("No sentinel products found for this period/aoi/cloud cover.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Sentinel-2 images and predict Canopy height.')
    parser.add_argument('--car', required=True, help='CAR code of the property')
    args = parser.parse_args()
    main(args.car)