import math
import numpy as np

# GINI INDEX


class DTGI:
    def __init__(self):
        self.tree = None

    def hitung_gini(self, kolom_kelas):
        elemen, banyak = np.unique(kolom_kelas, return_counts=True)
        nilai_gini = 1 - np.sum(
            [(banyak[i] / np.sum(banyak)) ** 2 for i in range(len(elemen))]
        )
        return nilai_gini

    def gini_split(self, data, nama_fitur_split, nama_fitur_kelas):
        nilai, banyak = np.unique(data[nama_fitur_split], return_counts=True)
        gini_split = np.sum(
            [
                (banyak[i] / np.sum(banyak))
                * self.hitung_gini(
                    data.where(data[nama_fitur_split] == nilai[i]).dropna()[
                        nama_fitur_kelas
                    ]
                )
                for i in range(len(nilai))
            ]
        )
        return gini_split

    def buat_tree(
        self, data, data_awal, daftar_fitur, nama_fitur_kelas, kelas_parent_node=None
    ):
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
                self.gini_split(data, fitur, nama_fitur_kelas) for fitur in daftar_fitur
            ]
            index_fitur_terbaik = np.argmin(nilai_split)
            fitur_terbaik = daftar_fitur[index_fitur_terbaik]
            tree = {fitur_terbaik: {}}
            daftar_fitur = [i for i in daftar_fitur if i != fitur_terbaik]
            for nilai in np.unique(data[fitur_terbaik]):
                sub_data = data.where(data[fitur_terbaik] == nilai).dropna()
                sub_tree = self.buat_tree(
                    sub_data,
                    data_awal,
                    daftar_fitur,
                    nama_fitur_kelas,
                    kelas_parent_node,
                )
                tree[fitur_terbaik][nilai] = sub_tree
        return tree

    def prediksiGini(self, data_uji, tree):
        for key in list(data_uji.keys()):
            if key in list(tree.keys()):
                try:
                    hasil = tree[key][data_uji[key]]
                except:
                    return 1
                hasil = tree[key][data_uji[key]]
                if isinstance(hasil, dict):
                    return self.prediksiGini(data_uji, hasil)
                else:
                    return hasil

    def fit(self, data_latih, class_name_column):
        self.tree = self.buat_tree(
            data_latih, data_latih, data_latih.columns[:-1], class_name_column
        )


# INFORMATION GAIN


class DTIF:
    def __init__(self) -> None:
        self.tree = None

    def hitung_entropy(self, kolom_kelas):
        elemen, banyak = np.unique(kolom_kelas, return_counts=True)
        entropy = -1 * (
            np.sum(
                [
                    (banyak[i] / np.sum(banyak)) * math.log2(banyak[i] / np.sum(banyak))
                    for i in range(len(elemen))
                ]
            )
        )
        return entropy

    def information_gain(self, data, nama_fitur_split, nama_fitur_kelas):
        # tuliskan kode Anda di sini
        root_entropy = self.itung_entropy(data[nama_fitur_kelas])
        nilai, banyak = np.unique(data[nama_fitur_split], return_counts=True)
        entropy_split = np.sum(
            [
                (banyak[i] / np.sum(banyak))
                * self.hitung_entropy(
                    data.where(data[nama_fitur_split] == nilai[i]).dropna()[
                        nama_fitur_kelas
                    ]
                )
                for i in range(len(nilai))
            ]
        )
        information_gain = root_entropy - entropy_split
        return information_gain

    def buat_tree_ig(
        self, data, data_awal, daftar_fitur, nama_fitur_kelas, kelas_parent_node=None
    ):
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
                self.information_gain(data, fitur, nama_fitur_kelas)
                for fitur in daftar_fitur
            ]
            index_fitur_terbaik = np.argmax(nilai_split)
            fitur_terbaik = daftar_fitur[index_fitur_terbaik]
            tree = {fitur_terbaik: {}}
            daftar_fitur = [i for i in daftar_fitur if i != fitur_terbaik]
            for nilai in np.unique(data[fitur_terbaik]):
                sub_data = data.where(data[fitur_terbaik] == nilai).dropna()
                sub_tree = self.buat_tree_ig(
                    sub_data,
                    data_awal,
                    daftar_fitur,
                    nama_fitur_kelas,
                    kelas_parent_node,
                )
                tree[fitur_terbaik][nilai] = sub_tree
            return tree

    def fit(self, data_latih, class_name_column):
        self.tree = self.buat_tree(
            data_latih, data_latih, data_latih.columns[:-1], class_name_column
        )
