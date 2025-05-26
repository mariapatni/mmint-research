#Copyright 2015 Yale University - Grablab
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import json
import urllib
import urllib.request
import urllib.error
import requests
from tqdm import tqdm

output_directory = "./ycb"

# Set this to "all" to download all objects
objects_to_download = "all"
#objects_to_download = ["002_master_chef_can", "003_cracker_box"]  # Comment this out
#objects_to_download = ["002_master_chef_can"]  # Comment this out

# Choose which file types you want
files_to_download = ["berkeley_processed", "berkeley_rgbd"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    response = urllib.request.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    try:
        # Make the request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Get file size from headers
        file_size = int(response.headers.get('content-length', 0))

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Open file and write in chunks with progress bar
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=file_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        return False

def tgz_url(object, type):
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url):
    try:
        request = urllib.request.Request(url)
        request.get_method = lambda : 'HEAD'
        response = urllib.request.urlopen(request)
        return True
    except Exception as e:
        return False


if __name__ == "__main__":
    # Fetch the full list of objects if "all" is specified
    if objects_to_download == "all":
        objects = fetch_objects(objects_url)
    else:
        objects = objects_to_download

    for object in objects:
        if objects_to_download == "all" or object in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(object, file_type)
                if not check_url(url):
                    print(f"Skipping {url} - not available")
                    continue
                filename = "{path}/{object}_{file_type}.tgz".format(path=output_directory,
                                                                    object=object,
                                                                    file_type=file_type)
                print(f"Downloading {object} - {file_type}")
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, output_directory)