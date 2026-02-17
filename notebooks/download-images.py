import os
from bing_image_downloader import downloader

def get_eggplants():
    keywords = ["eggplant", "aubergine", "brinjal", "defective eggplant", "bad eggplant", "good eggplant","high quality eggplant", "rotten eggplant"]
    
    for query in keywords:
        downloader.download(
            query, 
            limit=50, 
            output_dir='eggplant_dataset', 
            adult_filter_off=True, 
            force_replace=False, 
            timeout=60,
            verbose=True
        )

if __name__ == "__main__":
    get_eggplants()