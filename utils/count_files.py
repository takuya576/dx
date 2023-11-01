import os


def count_JPG_files(directory):
    jpg_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_count += 1

    return jpg_count


jpg_count = count_JPG_files("/home/sakamoto/dx/data/data5")
print(jpg_count)
