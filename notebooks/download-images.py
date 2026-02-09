#!/usr/bin/env python3
"""
Download eggplant images from Google using simple_image_download
"""

import os
import sys
import time
from pathlib import Path

def check_and_install():
    """Check and install required packages."""
    print("Checking dependencies...")
    
    try:
        from simple_image_download import simple_image_download as sim
        print("✓ simple_image_download already installed")
        return True
    except ImportError:
        print("Installing simple_image_download...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "simple_image_download"])
            print("✓ simple_image_download installed successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to install: {e}")
            print("\nYou can manually install with: pip install simple_image_download")
            return False

def download_eggplant_images(
    output_dir="downloaded_eggplant_images",
    keywords=None,
    limit_per_keyword=50,
    timeout=10
):
    """
    Download eggplant images from Google.
    
    Args:
        output_dir: Where to save downloaded images
        keywords: List of search keywords
        limit_per_keyword: Max images per keyword
        timeout: Timeout for each download (seconds)
    """
    
    if keywords is None:
        keywords = [
            "eggplant",
            "purple eggplant",
            "aubergine",
            "eggplant vegetable",
            "fresh eggplant",
            "eggplant farm",
            "eggplant plant",
            "eggplant harvest",
            "brinjal",  # Indian name
            "eggplant closeup"
        ]
    
    print(f"\n{'='*60}")
    print("EGGPLANT IMAGE DOWNLOADER")
    print(f"{'='*60}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # Initialize downloader
    try:
        from simple_image_download import simple_image_download as sim
        downloader = sim.simple_image_download
    except Exception as e:
        print(f"✗ Error initializing downloader: {e}")
        return
    
    # Download images for each keyword
    total_downloaded = 0
    
    for keyword in keywords:
        print(f"\n📥 Downloading: '{keyword}' (max: {limit_per_keyword} images)")
        
        try:
            # Set timeout for this download
            start_time = time.time()
            
            # Download images
            downloader().download(
                keyword, 
                limit_per_keyword
            )
            
            download_time = time.time() - start_time
            print(f"  Download completed in {download_time:.1f}s")
            
            # Count downloaded images
            keyword_safe = keyword.replace(" ", "_")
            download_folder = Path("simple_images") / keyword_safe
            if download_folder.exists():
                images = list(download_folder.glob("*.*"))
                print(f"  Found {len(images)} images")
                
                # Move to our output folder
                keyword_output = output_path / keyword_safe
                keyword_output.mkdir(exist_ok=True)
                
                moved_count = 0
                for img in images:
                    if img.is_file():
                        dest = keyword_output / img.name
                        try:
                            img.rename(dest)
                            moved_count += 1
                        except Exception as e:
                            print(f"    Warning: Could not move {img.name}: {e}")
                
                total_downloaded += moved_count
                print(f"  Moved {moved_count} images to: {keyword_output.name}")
            
            # Small delay between downloads
            time.sleep(2)
            
        except Exception as e:
            print(f"  ✗ Error downloading '{keyword}': {e}")
            continue
    
    # Clean up simple_images folder if empty
    simple_images_dir = Path("simple_images")
    if simple_images_dir.exists():
        try:
            # Check if empty
            has_files = any(simple_images_dir.iterdir())
            if not has_files:
                simple_images_dir.rmdir()
        except:
            pass
    
    # Print summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Keywords searched: {len(keywords)}")
    print(f"Total images downloaded: {total_downloaded}")
    print(f"Saved in: {output_path}")
    
    # Show folder structure
    print(f"\nFolder structure:")
    for item in output_path.iterdir():
        if item.is_dir():
            num_images = len(list(item.glob("*.*")))
            print(f"  📁 {item.name}/ - {num_images} images")
    
    print(f"\nReady for annotation!")
    print(f"Run the auto-annotation script on this folder.")

def quick_download_eggplant():
    """Quick download with default settings."""
    print("Starting quick download of eggplant images...")
    
    # Check/install dependencies
    if not check_and_install():
        return
    
    # Download images
    download_eggplant_images(
        output_dir="eggplant_downloads",
        limit_per_keyword=30,  # Smaller for quick test
        timeout=15
    )

def custom_download():
    """Custom download with user input."""
    print("\nCustom Eggplant Image Download")
    print("-" * 40)
    
    # Get output directory
    output_dir = input("Output directory name [eggplant_images]: ").strip()
    if not output_dir:
        output_dir = "eggplant_images"
    
    # Get keywords
    print("\nEnter search keywords (comma-separated, press Enter for default):")
    keywords_input = input().strip()
    
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(",")]
    else:
        keywords = [
            "eggplant",
            "purple eggplant", 
            "aubergine",
            "eggplant farm"
        ]
        print(f"Using default keywords: {', '.join(keywords)}")
    
    # Get limit
    try:
        limit = int(input(f"Images per keyword [50]: ").strip() or "50")
    except:
        limit = 50
    
    # Check/install dependencies
    if not check_and_install():
        return
    
    # Download
    download_eggplant_images(
        output_dir=output_dir,
        keywords=keywords,
        limit_per_keyword=limit,
        timeout=20
    )

def main():
    """Main menu."""
    print("🌱 EGGPLANT IMAGE DOWNLOADER")
    print("=" * 40)
    print("1. Quick download (default settings)")
    print("2. Custom download")
    print("3. Download for auto-annotation")
    print("4. Exit")
    
    choice = input("\nSelect option [1]: ").strip()
    
    if choice == "2":
        custom_download()
    elif choice == "3":
        # Optimal settings for auto-annotation
        print("\nDownloading images optimized for auto-annotation...")
        if check_and_install():
            download_eggplant_images(
                output_dir="farm_images",  # Matches auto-annotation script
                keywords=[
                    "eggplant farm",
                    "eggplant plant",
                    "eggplant field",
                    "harvesting eggplant",
                    "purple eggplant farm",
                    "eggplant agriculture",
                    "fresh eggplant",
                    "organic eggplant"
                ],
                limit_per_keyword=40,
                timeout=15
            )
    elif choice == "4":
        print("Goodbye!")
        return
    else:  # Default to quick download
        quick_download_eggplant()

if __name__ == "__main__":
    main()