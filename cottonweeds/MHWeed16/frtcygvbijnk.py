import os

def get_all_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            rel_path = os.path.relpath(os.path.join(root, f), folder)
            files.append(rel_path)
    return set(files)

folder1 = r"E:\UAV_split"
folder2 = r"E:\MH-Weed16\UAV_split"

files1 = get_all_files(folder1)
files2 = get_all_files(folder2)

if files1 == files2:
    print("两个文件夹完全相同，可以删除任意一个。")
else:
    print("文件夹内容不完全相同，请检查差异。")
    only_in_1 = files1 - files2
    only_in_2 = files2 - files1
    print("仅在 E:\\UAV_split 的文件:", only_in_1)
    print("仅在 E:\\MH-Weed16\\UAV_split 的文件:", only_in_2)

