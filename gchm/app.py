import argparse
import json
import os
import datetime
import venv
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from pyproj import Proj
from tqdm import tqdm
from decouple import config
from datetime import datetime, timedelta
import os
import subprocess
import geopandas as gpd
from shapely.geometry import shape


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
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 1.00) "
        f"and contains(Name%2C%27L2A%27)"
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
    GCHM_DEPLOY_DIR = "../deploy_example/predictions/"
    GCHM_MODEL_DIR = "../trained_models/GLOBAL_GEDI_2019_2020"
    GCHM_NUM_MODELS = "5"
    filepath_failed_image_paths = "../deploy_example/log_failed.txt"
    GCHM_DOWNLOAD_FROM_AWS = "False"
    GCHM_DEPLOY_SENTINEL2_AWS_DIR = "../deploy_example/sentinel2_aws"

    os.makedirs(GCHM_DEPLOY_DIR, exist_ok=True)
    os.makedirs(GCHM_DEPLOY_SENTINEL2_AWS_DIR, exist_ok=True)

    home_dir = os.path.expanduser("~")
    venv_python = os.path.join(home_dir, "venvs/brchm/bin/python")


    command = [
        venv_python, "deploy.py",
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
    GCHM_DEPLOY_DIR = "../deploy_example/predictions"
    reduction = "inv_var_mean"
    out_dir = f"{GCHM_DEPLOY_DIR}_merge/preds_{reduction}/"
    out_type = "uint8"
    nodata_value = 255
    from_aws = False
    finetune_strategy = "FT_Lm_SRCB"
    os.makedirs(out_dir, exist_ok=True)

    print("*************************************")
    print(f"merging... {reduction}")
    print("reduction:", reduction)
    print("out_dir:", out_dir)

    # Executando o script Python
    command = [
        "python3",
        "merge_predictions_tile.py",
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


def plot_pixel_distribution(file_path):
    # Abra o arquivo GeoTIFF
    with rasterio.open(file_path) as src:
        # Leia a primeira banda
        band1 = src.read(1)

        # Exclua os valores 255
        valid_pixels = band1[band1 != 255]

        # Plote o histograma dos valores dos pixels
        plt.figure(figsize=(10, 6))
        plt.hist(valid_pixels, bins=50, color='blue', edgecolor='black')
        plt.title('Distribuição dos Valores dos Pixels (excluindo 255)')
        plt.xlabel('Valor do Pixel')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.show()


def plot_tiff(file_path, gdf):
    with rasterio.open(file_path) as src:
        num_bands = src.count
        print(f"O arquivo TIFF possui {num_bands} bandas.")

        # Leia a primeira banda
        band1 = src.read(1)

        # Obtenha o CRS do GeoTIFF
        tiff_crs = src.crs
        print(f"CRS do TIFF: {tiff_crs}")

        # Verifique se o GeoDataFrame está no mesmo CRS do TIFF (EPSG:32722)
        if gdf.crs != tiff_crs:
            print("Reprojecionando GeoDataFrame para CRS do TIFF...")
            gdf = gdf.to_crs(tiff_crs)

        # Mascarar valores iguais a 255
        masked_band1 = np.ma.masked_equal(band1, 255)

        # Configura a figura e o eixo
        fig, ax = plt.subplots(figsize=(10, 10))

        # Obter a extensão do raster
        raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        # Exiba a imagem com uma paleta de cores diferente e valores 255 mascarados
        im = ax.imshow(masked_band1, cmap='viridis', extent=raster_extent)  # Usar extent para alinhar corretamente

        # Converter as coordenadas do GeoDataFrame para o sistema de coordenadas do raster
        if gdf.crs != tiff_crs:
            proj_in = Proj(gdf.crs)
            proj_out = Proj(tiff_crs)
            gdf['geometry'] = gdf['geometry'].to_crs(proj_out)

        # Plote o GeoDataFrame no eixo
        gdf.plot(ax=ax, facecolor='none', edgecolor='red')

        # Adicione uma barra de cores
        cbar = plt.colorbar(im, ax=ax, label='Valores de pixel')

        # Adicione título e rótulos
        plt.title('Imagem GeoTIFF com Valores 255 Mascarados')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Exiba a plotagem
        plt.show()


def normalize_band(band, max_value):
    # Excluir valores 255
    mask = band != 255
    valid_pixels = band[mask]

    # Normalizar os valores dos pixels
    max_pixel_value = valid_pixels.max()

    normalized_pixels = (valid_pixels / max_pixel_value) * max_value

    # Criar uma cópia da banda original
    new_band = band.copy()

    # Aplicar os valores normalizados aos pixels válidos
    new_band[mask] = normalized_pixels

    return new_band


def create_normalized_tiff(input_file, output_file, max_value=20):
    # Abra o arquivo GeoTIFF original
    with rasterio.open(input_file) as src:
        # Leia a primeira banda
        band1 = src.read(1)

        # Normalize os valores da banda
        normalized_band = normalize_band(band1, max_value=max_value)

        # Crie um novo arquivo GeoTIFF com os valores normalizados
        profile = src.profile
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(normalized_band, 1)



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
    output_path = '../deploy_example/sentinel2/'

    json_data = car_to_geojson(car, api_root, api_token)
    geometry = shape(json_data)
    with open('/tmp/aoi.json', 'w') as f:
        json.dump(json_data, f, indent=4)
    wkt = geojson_to_wkt_bbox(json_data)
    json_data = get_available_sentinel_products(wkt, start_date_str, end_date_str)

    if json_data is not None and len(json_data['value']) > 0:
        tile = json_data['value'][0]['Name'].split('_T')[1].split('_')[0]
        print(tile)
        for item in json_data['value']:
            image_path = output_path + json_data['value'][0]['Name'] + ".zip"
            print(f"Starting to download file {json_data['value'][0]['Name']} at {image_path}.")
            file_path = Path(image_path)
            dir_path = Path(output_path+json_data['value'][0]['Name'])
            '''
            if not dir_path.is_dir() or not file_path.is_file():
                download_sentinel_product(json_data['value'][0]['Id'], copernicus_dataspace_login, copernicus_dataspace_passwd, image_path)
                unzip_file(image_path, output_path)
            else:
                print('Product already downloaded.')
            run_canopy_prediction(image_path)
        run_merge_predictions(tile)
        '''
        merged_predictions_file_path = '../deploy_example/predictions_merge/preds_inv_var_mean/' + tile + '_pred.tif'

        gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs="EPSG:4326")

        plot_tiff(merged_predictions_file_path, gdf)
        plot_pixel_distribution(merged_predictions_file_path)
        output_file = '/tmp/arquivo_normalizado.tif'
        create_normalized_tiff(merged_predictions_file_path, output_file)
        plot_tiff(output_file, gdf)
        plot_pixel_distribution(output_file)


    else:
        print("No sentinel products found for this period/aoi/cloud cover.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Sentinel-2 images and predict Canopy height.')
    parser.add_argument('--car', required=True, help='CAR code of the property')
    args = parser.parse_args()
    main(args.car)