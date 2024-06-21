import os
import requests
import zipfile
import shutil
from tqdm import tqdm


def check_env():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    required_keys = ['TOKEN', 'COPERNICUS_DATASPACE_LOGIN', 'COPERNICUS_DATASPACE_PASSWD']

    # Check if .env file exists
    if os.path.exists(env_path):
        # Read the existing .env file
        with open(env_path, 'r') as file:
            lines = file.readlines()

        # Dictionary to hold existing keys and their values
        env_dict = {}
        for line in lines:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                env_dict[key] = value

        # Check for required keys and their values
        with open(env_path, 'a') as file:
            for key in required_keys:
                if key in env_dict:
                    if not env_dict[key]:
                        print(f"The key '{key}' is missing a value and needs to be filled for the program to function.")
                else:
                    print(
                        f"The key '{key}' does not exist and has been created. Please provide an appropriate value for the program to function.")
                    file.write(f"{key}=\n")
    else:
        # Create .env file and write required keys
        print(".env file does not exist. Creating .env file with required keys.")
        with open(env_path, 'w') as file:
            for key in required_keys:
                file.write(f"{key}=\n")
                print(
                    f"The key '{key}' has been created. Please provide an appropriate value for the program to function.")


# Example usage:
# check_env()


def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong")


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, unit='file') as t:
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_to)
                t.update(1)


def prepare(force=False):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    trained_models_dir = os.path.join(base_path, 'trained_models')
    deploy_example_dir = os.path.join(base_path, 'deploy_example')

    if force:
        if os.path.exists(trained_models_dir):
            shutil.rmtree(trained_models_dir)
        if os.path.exists(deploy_example_dir):
            shutil.rmtree(deploy_example_dir)

    if not os.path.exists(trained_models_dir) or not os.path.exists(deploy_example_dir):
        trained_models_url = 'https://public-scicrop.s3.amazonaws.com/academy/trained_models.zip'
        deploy_example_url = 'https://public-scicrop.s3.amazonaws.com/academy/deploy_example.zip'

        trained_models_zip = os.path.join(base_path, 'trained_models.zip')
        deploy_example_zip = os.path.join(base_path, 'deploy_example.zip')

        print("Downloading trained_models.zip")
        download_file(trained_models_url, trained_models_zip)

        print("Downloading deploy_example.zip")
        download_file(deploy_example_url, deploy_example_zip)

        print("Extracting trained_models.zip")
        unzip_file(trained_models_zip, base_path)

        print("Extracting deploy_example.zip")
        unzip_file(deploy_example_zip, base_path)

        os.remove(trained_models_zip)
        os.remove(deploy_example_zip)
    else:
        print("Directories already exist. Use force=True to re-download and extract.")

# Example usage:
# prepare(force=True)
