import os
import subprocess
import sys
import re
from pathlib import Path

# GIẢI PHÁP 1: Sử dụng yt-dlp (Khuyến nghị nhất)
def download_with_ytdlp(links, output_folder="downloads"):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("🚀 SỬ DỤNG YT-DLP")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(links, 1):
        try:
            print(f"[{i}/{len(links)}] Đang tải: {url}")
            
            # Cấu hình yt-dlp
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '192K',
                '--output', f'{output_folder}/%(title)s.%(ext)s',
                '--no-playlist',  # Chỉ tải video đơn lẻ, không tải playlist
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Thành công!")
                successful += 1
            else:
                print(f"❌ Lỗi: {result.stderr}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            failed += 1
        
        print("-" * 30)
    
    print(f"\n📊 Kết quả: {successful} thành công, {failed} thất bại")

# GIẢI PHÁP 2: Sử dụng youtube-dl
def download_with_youtubedl(links, output_folder="downloads2"):
    """
    Sử dụng youtube-dl classic
    Cài đặt: pip install youtube-dl
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("🚀 SỬ DỤNG YOUTUBE-DL")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(links, 1):
        try:
            print(f"[{i}/{len(links)}] Đang tải: {url}")
            
            cmd = [
                'youtube-dl',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '192K',
                '--output', f'{output_folder}/%(title)s.%(ext)s',
                '--no-playlist',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Thành công!")
                successful += 1
            else:
                print(f"❌ Lỗi: {result.stderr}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            failed += 1
        
        print("-" * 30)
    
    print(f"\n📊 Kết quả: {successful} thành công, {failed} thất bại")

# GIẢI PHÁP 3: Sử dụng pytubefix (fork cải tiến của pytube)
def download_with_pytubefix(links, output_folder="downloads"):
    """
    Sử dụng pytubefix - fork được maintain tốt hơn pytube
    Cài đặt: pip install pytubefix
    """
    try:
        from pytubefix import YouTube
        from moviepy.editor import AudioFileClip
    except ImportError:
        print("❌ Cần cài đặt: pip install pytubefix moviepy")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("🚀 SỬ DỤNG PYTUBEFIX")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(links, 1):
        try:
            print(f"[{i}/{len(links)}] Đang xử lý: {url}")
            
            # Clean URL - loại bỏ playlist parameters
            clean_url = url.split('&list=')[0] if '&list=' in url else url
            
            yt = YouTube(clean_url)
            title = re.sub(r'[<>:"/\\|?*]', '_', yt.title)[:100]
            
            print(f"Tiêu đề: {title}")
            
            # Tải audio stream
            audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                print("❌ Không tìm thấy audio stream")
                failed += 1
                continue
            
            # Download
            temp_path = audio_stream.download(output_path=output_folder, filename=f"temp_{title}")
            
            # Convert to MP3
            mp3_path = os.path.join(output_folder, f"{title}.mp3")
            audio_clip = AudioFileClip(temp_path)
            audio_clip.write_audiofile(mp3_path, verbose=False, logger=None)
            audio_clip.close()
            
            # Cleanup
            os.remove(temp_path)
            
            print(f"✅ Thành công: {title}.mp3")
            successful += 1
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            failed += 1
        
        print("-" * 30)
    
    print(f"\n📊 Kết quả: {successful} thành công, {failed} thất bại")

# GIẢI PHÁP 4: Batch script cho yt-dlp
def create_batch_script(links, output_folder="downloads"):
    """
    Tạo batch script để chạy yt-dlp
    """
    script_content = f'''@echo off
echo Starting YouTube MP3 Downloads...
mkdir "{output_folder}" 2>nul

'''
    
    for i, url in enumerate(links, 1):
        clean_url = url.split('&list=')[0] if '&list=' in url else url
        script_content += f'''echo [{i}/{len(links)}] Downloading: {clean_url}
yt-dlp --extract-audio --audio-format mp3 --audio-quality 192K --output "{output_folder}/%(title)s.%(ext)s" --no-playlist "{clean_url}"
echo.

'''
    
    script_content += '''echo All downloads completed!
pause
'''
    
    with open('download_youtube.bat', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Đã tạo file download_youtube.bat")
    print("Chạy file này để tải xuống tất cả video")

# Danh sách URL đã được làm sạch
youtube_links = [
    "https://www.youtube.com/watch?v=zoipEJ2_Nw8",
    "https://www.youtube.com/watch?v=wVsZ3SNmD74",
    "https://www.youtube.com/watch?v=UHNIU7a84NU",
    "https://www.youtube.com/watch?v=hqpagaYiD9k",
    "https://www.youtube.com/watch?v=J7BwDfoaLnA",
    "https://www.youtube.com/watch?v=7AeqlnTkrYk",
    "https://www.youtube.com/watch?v=nfnsCjz1S34",
    "https://www.youtube.com/watch?v=pqI425WLINg",
    "https://www.youtube.com/watch?v=ecAra80IppM",
    "https://www.youtube.com/watch?v=b6eac3TEnXU",
    "https://www.youtube.com/watch?v=xnu7qIIPJJM"
    "https://www.youtube.com/watch?v=9bFhgoRD8zE&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=54",
    "https://www.youtube.com/watch?v=O0Qhxw6jJYI&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=55",
    "https://www.youtube.com/watch?v=Trp5XskJygw&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=57",
    "https://www.youtube.com/watch?v=hecw9EgfMKE&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=59",
    "https://www.youtube.com/watch?v=SnfMFa4KJEE&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=61",
    "https://www.youtube.com/watch?v=UfxlPYDjFQk&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=63"
    "https://www.youtube.com/watch?v=sCnA_wMfIyI&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=65",
    "https://www.youtube.com/watch?v=5RzWUR--qL8&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=67",
    "https://www.youtube.com/watch?v=E54RnSNc5z0&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=69",
    "https://www.youtube.com/watch?v=o3jzqyIYU6E&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=71",
    "https://www.youtube.com/watch?v=yxoJSMs3HnM&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=73",
    "https://www.youtube.com/watch?v=sP6BEtu672A&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=75",
    "https://www.youtube.com/watch?v=9OLcIGlCZbw&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=77",
    "https://www.youtube.com/watch?v=j--PHVf9aSw&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=79",
    "https://www.youtube.com/watch?v=9rjE81nbhdk&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=81",
    "https://www.youtube.com/watch?v=Gy40Aqr-IR8&list=PLbkz2ORJaK6jCdQwZvlBv04ZXn8k1PEfl&index=83",
]

def main():
    print("CHỌN PHƯƠNG PHÁP TẢI XUỐNG:")
    print("1. yt-dlp (Khuyến nghị)")
    print("2. youtube-dl")
    print("3. pytubefix")
    print("4. Tạo batch script")
    
    choice = input("\nNhập lựa chọn (1-4): ").strip()
    
    if choice == "1":
        download_with_ytdlp(youtube_links)
    elif choice == "2":
        download_with_youtubedl(youtube_links)
    elif choice == "3":
        download_with_pytubefix(youtube_links)
    elif choice == "4":
        create_batch_script(youtube_links)
    else:
        print("Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main()