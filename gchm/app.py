import argparse
import json
import datetime
import sys
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from pyproj import Proj
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, BBox, CRS
from tqdm import tqdm
from decouple import config
from datetime import datetime, timedelta
import os
import subprocess
import geopandas as gpd
from shapely.geometry import shape
from jsonschema import validate, ValidationError
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from gchm.utils.prepare import prepare, check_env


def unzip_file(zip_file_path, extract_to_dir):
    os.makedirs(extract_to_dir, exist_ok=True)
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


def geojson_to_bbox(geojson):
    if geojson['type'] != 'Polygon':
        raise ValueError("O GeoJSON fornecido não é um 'Polygon'")

    coordinates = geojson['coordinates'][0]

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

    bbox = f"POLYGON(({min_x} {min_y}, {min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"

    return bbox


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
    with rasterio.open(file_path) as src:
        band1 = src.read(1)
        valid_pixels = band1[band1 != 255]
        plt.figure(figsize=(10, 6))
        plt.hist(valid_pixels, bins=50, color='blue', edgecolor='black')
        plt.title('Canopy Height by Pixel')
        plt.xlabel('Canopy Height')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def plot_tiff_rgb(file_path, gdf):
    with rasterio.open(file_path) as src:
        tiff_crs = src.crs
        if gdf.crs != tiff_crs:
            gdf = gdf.to_crs(tiff_crs)

        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
        rgb = np.dstack((red, green, blue))
        def adjust_contrast(image, lower_percentile=2, upper_percentile=98, gamma=1.0):
            lower = np.percentile(image, lower_percentile)
            upper = np.percentile(image, upper_percentile)
            image = np.clip(image, lower, upper)
            image = (image - lower) / (upper - lower)
            image = np.power(image, gamma)
            return image

        red_adjusted = adjust_contrast(red, gamma=0.8)
        green_adjusted = adjust_contrast(green, gamma=0.8)
        blue_adjusted = adjust_contrast(blue, gamma=0.8)

        rgb_adjusted = np.dstack((red_adjusted, green_adjusted, blue_adjusted))

        fig, ax = plt.subplots(figsize=(10, 10))
        raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        im = ax.imshow(rgb_adjusted, extent=raster_extent)
        if gdf.crs != tiff_crs:
            proj_out = Proj(tiff_crs)
            gdf['geometry'] = gdf['geometry'].to_crs(proj_out)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')


        plt.title('Sentinel 2 Image (RGB)')
        plt.axis('off')
        plt.show()


def plot_tiff(file_path, gdf):
    with rasterio.open(file_path) as src:
        band1 = src.read(1)
        tiff_crs = src.crs

        if gdf.crs != tiff_crs:
            gdf = gdf.to_crs(tiff_crs)

        masked_band1 = np.ma.masked_equal(band1, 255)
        fig, ax = plt.subplots(figsize=(10, 10))
        raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        im = ax.imshow(masked_band1, cmap='viridis', extent=raster_extent)
        if gdf.crs != tiff_crs:
            proj_out = Proj(tiff_crs)
            gdf['geometry'] = gdf['geometry'].to_crs(proj_out)

        gdf.plot(ax=ax, facecolor='none', edgecolor='red')
        cbar = plt.colorbar(im, ax=ax, label='Canopy Height Prediction')
        plt.title('Canopy Height')
        plt.axis('off')
        plt.show()


def normalize_band(band, max_value):
    mask = band != 255
    valid_pixels = band[mask]
    max_pixel_value = valid_pixels.max()
    normalized_pixels = (valid_pixels / max_pixel_value) * max_value
    new_band = band.copy()
    new_band[mask] = normalized_pixels
    return new_band


def create_normalized_tiff(input_file, output_file, max_value=20):
    with rasterio.open(input_file) as src:
        band1 = src.read(1)
        normalized_band = normalize_band(band1, max_value=max_value)
        profile = src.profile
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(normalized_band, 1)


def plot_rgb(directory, output_file):
    granule_dir = os.path.join(directory, 'GRANULE')
    subdir = next(os.walk(granule_dir))[1][0]
    subdir_path = os.path.join(granule_dir, subdir)
    img_data_dir = os.path.join(subdir_path, 'IMG_DATA', 'R10m')
    files = os.listdir(img_data_dir)
    b2_file = next(f for f in files if 'B02_10m.jp2' in f)
    b3_file = next(f for f in files if 'B03_10m.jp2' in f)
    b4_file = next(f for f in files if 'B04_10m.jp2' in f)
    b2_path = os.path.join(img_data_dir, b2_file)
    b3_path = os.path.join(img_data_dir, b3_file)
    b4_path = os.path.join(img_data_dir, b4_file)
    with rasterio.open(b2_path) as b2, rasterio.open(b3_path) as b3, rasterio.open(b4_path) as b4:
        b2_data = b2.read(1)
        b3_data = b3.read(1)
        b4_data = b4.read(1)

        rgb = np.stack((b4_data, b3_data, b2_data), axis=-1)

        profile = b2.profile
        profile.update(count=3)

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(rgb[:, :, 0], 1)
            dst.write(rgb[:, :, 1], 2)
            dst.write(rgb[:, :, 2], 3)


def main(car, aoi, only_prepare):
    check_env()
    if only_prepare:
        prepare(False)
    else:
        prepare(False)
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
        json_data = None
        if car:
            json_data = car_to_geojson(car, api_root, api_token)
        else:
            with open(aoi, 'r') as file:
                json_data = json.load(file)
        geometry = shape(json_data)
        wkt = geojson_to_bbox(json_data)
        json_data = get_available_sentinel_products(wkt, start_date_str, end_date_str)
        dir_path = ''
        if json_data is not None and len(json_data['value']) > 0:
            tile = json_data['value'][0]['Name'].split('_T')[1].split('_')[0]
            print(tile)
            for item in json_data['value']:
                image_path = output_path + json_data['value'][0]['Name'] + ".zip"
                print(f"Starting to download file {json_data['value'][0]['Name']} at {image_path}.")
                file_path = Path(image_path)
                dir_path = Path(output_path+json_data['value'][0]['Name'])

                if not dir_path.is_dir() or not file_path.is_file():
                    download_sentinel_product(json_data['value'][0]['Id'], copernicus_dataspace_login, copernicus_dataspace_passwd, image_path)
                    unzip_file(image_path, output_path)
                else:
                    print('Product already downloaded.')

                run_canopy_prediction(image_path)
            run_merge_predictions(tile)
            rgb_file_name = f'{tile}_rgb.tif'
            merged_predictions_file_name = f'{tile}_pred.tif'
            normalized_predictions_file_name = f'{tile}_norm.tif'
            merged_predictions_file_path = f'../deploy_example/predictions_merge/preds_inv_var_mean/{merged_predictions_file_name}'
            rgb_file_path = merged_predictions_file_path+rgb_file_name
            plot_rgb(dir_path, rgb_file_path)
            gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs="EPSG:4326")
            plot_tiff_rgb(rgb_file_path, gdf)
            #plot_tiff(merged_predictions_file_path, gdf)
            #plot_pixel_distribution(merged_predictions_file_patobh)
            normalized_predictions_file_path = merged_predictions_file_path + normalized_predictions_file_name
            create_normalized_tiff(merged_predictions_file_path, normalized_predictions_file_path)
            plot_tiff(normalized_predictions_file_path, gdf)
            plot_pixel_distribution(normalized_predictions_file_path)

        else:
            print("No sentinel products found for this period/aoi/cloud cover.")


def is_valid_geojson(file_path):
    geojson_schema = {
        "type": "object",
        "required": ["type", "features"],
        "properties": {
            "type": {"type": "string"},
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "geometry", "properties"],
                    "properties": {
                        "type": {"type": "string"},
                        "geometry": {"type": "object"},
                        "properties": {"type": "object"}
                    }
                }
            }
        }
    }

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            validate(instance=data, schema=geojson_schema)
    except (json.JSONDecodeError, ValidationError):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Sentinel-2 images and predict Canopy height.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--car', help='Brazilian CAR code of the property')
    group.add_argument('--aoi', help='Path to the AOI geojson file')
    group.add_argument('--prepare', help='Only get model and prepare folders')

    args = parser.parse_args()

    if args.aoi:
        if not os.path.isfile(args.aoi):
            print("Error: AOI file does not exists.")
            sys.exit(1)
        if not is_valid_geojson(args.aoi):
            print("Error: AOI is an invalid geojson file.")
            sys.exit(1)
    if args.aoi or args.car or args.prepare:
        main(args.car, args.aoi, args.prepare)
    else:
        print("No --car or --aoi was informed.")
        sys.exit(1)
