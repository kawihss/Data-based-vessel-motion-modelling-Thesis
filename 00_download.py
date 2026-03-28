# 00_download.py: Downloading AIS data from BSH for Kiel, Bremerhaven, and Wedel
# Input:  URLs
# Output: Data/Kiel/, Data/Bremerhaven/, Data/Wedel/

import os
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

def download_dataset(feed_url, output_dir):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    response = requests.get(feed_url)
    response.raise_for_status()

    # Parse XML
    root = ET.fromstring(response.content)
    
    # Find all links in the XML that contain '.log' (the actual data files)
    links = []
    for elem in root.iter():
        if 'href' in elem.attrib:
            link = elem.attrib['href']
            if '.log' in link:
                links.append(link)
                
    # Remove duplicates
    links = list(set(links))
    links = sorted(links)
    print(f"{len(links)} files to download.\n")
    
    # Download each file, skip if already exists
    for i, link in enumerate(links):
        filename = link.split('/')[-1]
        filepath = os.path.join(output_dir, filename)
        
        # Skip if file exists AND is not empty
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"  [{i+1}/{len(links)}] skipping {filename} (already exists)")
            continue
            
        print(f"  [{i+1}/{len(links)}] Loading {filename} ...")
        
        try:
            # Use stream=True to download large files in chunks safely
            with requests.get(link, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"  -> ERROR downloading {filename}: {e}")
            # Delete the incomplete file so it doesn't get skipped on the next run
            if os.path.exists(filepath):
                os.remove(filepath)
                
        # Wait a bit between downloads to not overload the server
        time.sleep(0.5)

if __name__ == "__main__":
    
    # 2025 URLS
    datasets = [
        ("https://gdi.bsh.de/de/feed/AIS-data-for-ship-emission-measurement-on-the-mesurementsite-Kiel-2025-series.xml", "Data/Kiel"),
        ("https://gdi.bsh.de/de/feed/AIS-data-for-ship-emission-measurement-on-the-mesurementsite-Bremerhaven-2025-series.xml", "Data/Bremerhaven"),
        ("https://gdi.bsh.de/de/feed/AIS-data-for-ship-emission-measurement-on-the-mesurementsite-Wedel-2025-series.xml", "Data/Wedel")
    ]
    
    for url, folder in datasets:
        download_dataset(url, folder)
        
    print(f"\n{'='*60}")
    print("download done")
    print(f"{'='*60}")