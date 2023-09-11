import os


def count_JPG_files_in_directory(directory):
    jpg_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_count += 1

    return jpg_count


# ディレクトリのパスを指定して.JPGファイルの数をカウント
directory_path = "/home/sakamoto/dx/coins_data/data4_all_cases/train"
jpg_count = count_JPG_files_in_directory(directory_path)

print(f"{directory_path} 以下にある.JPGファイルの数: {jpg_count}")
