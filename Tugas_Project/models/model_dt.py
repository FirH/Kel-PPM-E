import numpy as np


def hitung_gini(kolom_kelas):
    elemen, banyak = np.unique(kolom_kelas, return_counts=True)
    nilai_gini = 1 - np.sum(
        [(banyak[i] / np.sum(banyak)) ** 2 for i in range(len(elemen))]
    )
    return nilai_gini


def gini_split(data, nama_fitur_split, nama_fitur_kelas):
    nilai, banyak = np.unique(data[nama_fitur_split], return_counts=True)
    gini_split = np.sum(
        [
            (banyak[i] / np.sum(banyak))
            * hitung_gini(
                data.where(data[nama_fitur_split] == nilai[i]).dropna()[
                    nama_fitur_kelas
                ]
            )
            for i in range(len(nilai))
        ]
    )
    return gini_split


def buat_tree(data, data_awal, daftar_fitur, nama_fitur_kelas, kelas_parent_node=None):
    if len(np.unique(data[nama_fitur_kelas])) <= 1:
        return np.unique(data[nama_fitur_kelas])[0]
    elif len(data) == 0:
        return np.unique(data[nama_fitur_kelas])[
            np.argmax(np.unique(data_awal[nama_fitur_kelas], return_counts=True)[1])
        ]
    elif len(daftar_fitur) == 0:
        return kelas_parent_node
    else:
        kelas_parent_node = np.unique(data[nama_fitur_kelas])[
            np.argmax(np.unique(data[nama_fitur_kelas], return_counts=True)[1])
        ]
        nilai_split = [
            gini_split(data, fitur, nama_fitur_kelas) for fitur in daftar_fitur
        ]
        index_fitur_terbaik = np.argmin(nilai_split)
        fitur_terbaik = daftar_fitur[index_fitur_terbaik]
        tree = {fitur_terbaik: {}}
        daftar_fitur = [i for i in daftar_fitur if i != fitur_terbaik]
        for nilai in np.unique(data[fitur_terbaik]):
            sub_data = data.where(data[fitur_terbaik] == nilai).dropna()
            sub_tree = buat_tree(
                sub_data, data_awal, daftar_fitur, nama_fitur_kelas, kelas_parent_node
            )
            tree[fitur_terbaik][nilai] = sub_tree
        return tree
