# import os
# from tqdm import tqdm
# from zipfile import ZipFile
#
#
# def get_zipfiles(directory):
#     list_files = []
#     for filename in os.listdir(home):
#         if filename.endswith(".zip"):
#             list_files.append(os.path.join(home, filename))
#     return list_files
#
# zipfiles = get_zipfiles('/home/carlossouza')
# zipfiles.sort()
#
# for zipfile in zipfiles:
#     print(f'Extracting {zipfile}...')
#     with ZipFile(file=zipfile) as zip_file:
#         for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
#             zip_file.extract(member=file)
#
#     os.remove(zipfile)