import csv
import pathlib

from src import project_paths


def max_char_length(path_string):
    """Generate dictionaries to access the class labels corresponding
    to each part of a grapheme.
    """

    label_path = pathlib.Path(path_string)

    max_gr_len = 0
    max_vd_len = 0
    max_cd_len = 0

    with label_path.open(mode='r') as label_csv:

        csv_reader = csv.reader(label_csv)
        next(csv_reader)
        for row in csv_reader:

            if row[0] == 'grapheme_root':
                print(len(row[2]), "yo")
                max_gr_len = max(max_gr_len, len(row[2]))

            if row[0] == 'vowel_diacritic':
                print("here")
                max_vd_len = max(max_vd_len, len(row[2]))

            if row[0] == 'consonant_diacritic':
                max_cd_len = max(max_cd_len, len(row[2]))

    return max_gr_len, max_vd_len, max_cd_len


if __name__ == '__main__':
    data = project_paths()["data"] / 'raw' / 'class_map.csv'
    print(max_char_length(
        '/home/scott/Projects'
        '/kaggle__bengaliai_handwritten_grapheme_classification/data/raw'
        '/class_map'
        '.csv'))
