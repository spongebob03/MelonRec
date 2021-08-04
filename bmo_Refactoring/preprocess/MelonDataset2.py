import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from dataLoad.arena_util2 import load_json
from preprocess.data_util2 import genre_gn_all_preprocessing, genre_DicGenerator
import torch

def genre_gn_all_preprocessing(genre_gn_all):
    ## 대분류 장르코드
    # 장르코드 뒷자리 두 자리가 00인 코드를 필터링
    gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] == '00']

    ## 상세 장르코드
    # 장르코드 뒷자리 두 자리가 00이 아닌 코드를 필터링
    dtl_gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] != '00'].copy()
    dtl_gnr_code.rename(columns={'gnr_code': 'dtl_gnr_code', 'gnr_name': 'dtl_gnr_name'}, inplace=True)

    return gnr_code, dtl_gnr_code


def genre_DicGenerator(gnr_code, dtl_gnr_code, song_meta):
    ## gnr_dic (key: 대분류 장르 / value: 대분류 장르 id)
    gnr_dic = {}
    i = 0
    for gnr in gnr_code['gnr_code']:
        gnr_dic[gnr] = i
        i += 1

    ## dtl_dic (key: 상세 장르 / value: 상세 장르 id)
    dtl_dic = {}
    j = 0
    for dtl in dtl_gnr_code['dtl_gnr_code']:
        dtl_dic[dtl] = j
        j += 1

    ## song_gnr_dic (key: 곡 id / value: 해당 곡의 대분류 장르)
    ## song_dtl_dic (key: 곡 id / value: 해당 곡의 상세 장르)
    song_gnr_dic = {}
    song_dtl_dic = {}

    for s in song_meta:
        song_gnr_dic[s['id']] = s['song_gn_gnr_basket']
        song_dtl_dic[s['id']] = s['song_gn_dtl_gnr_basket']

    return gnr_dic, dtl_dic, song_gnr_dic, song_dtl_dic

# SongTagDataset, SongTagGenreDataset 클래스의 공통된 메서드
# 공통된 메서드를 가지는 부모 클래스를 만들어 이를 상속
class ParentDataset(Dataset):
    def __init__(self, json_dataset, tag2id_file_path, prep_song2id_file_path):
        self.train = json_dataset
        self.tag2id = dict(np.load(tag2id_file_path, allow_pickle=True).item())
        self.prep_song2id = dict(np.load(prep_song2id_file_path, allow_pickle=True).item())
        self.num_songs = len(self.prep_song2id)
        self.num_tags = len(self.tag2id)

    def __len__(self):
        return len(self.train)
    
    def _song_ids2vec(self, songs):
        songs = [self.prep_song2id[song] for song in songs if song in self.prep_song2id.keys()]

        songs = np.asarray(songs, dtype=np.int)
        bin_vec = np.zeros(self.num_songs)
        if len(songs) > 0:
            bin_vec[songs] = 1
        return np.array(bin_vec)

    def _tag_ids2vec(self, tags):
        tags = [self.tag2id[tag] for tag in tags if tag in self.tag2id.keys()]
        tags = np.asarray(tags, dtype=np.int)
        bin_vec = np.zeros(self.num_tags)
        bin_vec[tags] = 1
        return np.array(bin_vec)

class SongTagDataset(ParentDataset):
    def __init__(self, json_dataset, tag2id_file_path, prep_song2id_file_path):
        super().__init__(json_dataset, tag2id_file_path, prep_song2id_file_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _id = self.train[idx]['id']
        song_vector = self._song_ids2vec(self.train[idx]['songs'])
        tag_vector = self._tag_ids2vec(self.train[idx]['tags'])
        _input = torch.from_numpy(
            np.concatenate([song_vector, tag_vector]).astype(np.float32))

        return _id, _input


class SongTagGenreDataset(ParentDataset):
    def __init__(self, json_dataset, tag2id_file_path, prep_song2id_file_path):
        self.train = json_dataset
        self.tag2id = dict(np.load(tag2id_file_path, allow_pickle=True).item())
        self.prep_song2id = dict(np.load(prep_song2id_file_path, allow_pickle=True).item())
        self.num_songs = len(self.prep_song2id)
        self.num_tags = len(self.tag2id)
        self._init_song_meta()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _id = self.train[idx]['id']
        song_vector = self._song_ids2vec(self.train[idx]['songs'])
        tag_vector = self._tag_ids2vec(self.train[idx]['tags'])
        gnr_vector = self._get_vector(self.train[idx]['songs'], self.gnr_code, self.gnr_dic, self.song_gnr_dic)
        dtl_gnr_vector = self._get_vector(self.train[idx]['songs'], self.dtl_gnr_code, self.dtl_dic, self.song_dtl_dic)
        _input = torch.from_numpy(np.concatenate([song_vector, tag_vector]).astype(np.float32))

        return _id, _input, gnr_vector, dtl_gnr_vector

    def _init_song_meta(self):
        song_meta = load_json('res/song_meta.json')

        genre_gn_all = pd.read_json('res/genre_gn_all.json', encoding='utf8', typ='series')
        genre_gn_all = pd.DataFrame(genre_gn_all, columns=['gnr_name']).reset_index().rename(
            columns={'index': 'gnr_code'})

        self.gnr_code, self.dtl_gnr_code = genre_gn_all_preprocessing(genre_gn_all)
        self.num_gnr = len(self.gnr_code)
        self.num_dtl_gnr = len(self.dtl_gnr_code)
        self.gnr_dic, self.dtl_dic, self.song_gnr_dic, self.song_dtl_dic = genre_DicGenerator(
            self.gnr_code, self.dtl_gnr_code, song_meta)

    def _get_vector(self, songs, gnr_code, gnr_dic, song_gnr_dic):
        # v_gnr (각 플레이리스트의 수록곡 장르 비율을 담은 30차원 vector)
        v_gnr = np.zeros(len(gnr_code))
        for t_s in songs:
            for g in song_gnr_dic[t_s]:
                if g in gnr_code['gnr_code'].values:
                    v_gnr[gnr_dic[g]] += 1
        if v_gnr.sum() > 0:
            v_gnr = v_gnr / v_gnr.sum()
        return v_gnr