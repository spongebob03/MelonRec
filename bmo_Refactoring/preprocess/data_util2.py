# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import collections
import torch


def tags_ids_convert(json_data, tag2id_filepath, id2tag_filepath):
    playlist_df = pd.DataFrame(json_data)
    tags_list = playlist_df['tags'].to_list()
    _id = 0
    tags_dict = dict()
    ids_dict = dict()
    tags_set = set()
    for tags in tags_list:
        for tag in tags:
            if tag not in tags_set:
                tags_set.add(tag)
                tags_dict[tag] = _id
                ids_dict[_id] = tag
                _id += 1
    with open(tag2id_filepath, 'wb') as f:
        np.save(f, tags_dict)
        print('{} is created'.format(tag2id_filepath))
    with open(id2tag_filepath, 'wb') as f:
        np.save(f, ids_dict)
        print('{} is created'.format(id2tag_filepath))
    return True

# merge binary_songs2ids, binary_tags2ids
def binary_data2ids(_input, output, outsize, data_dict, istrain=False):
    if torch.cuda.is_available():
        _input = _input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
    else:
        _input = _input.detach().numpy()
        output = output.detach().numpy()

    to_song_id = lambda x: [data_dict[_x] for _x in x]

    if not istrain:
        output -= _input

    songs_idxes = output.argsort(axis=1)[:, ::-1][:, :outsize]

    return list(map(to_song_id, songs_idxes))

def save_freq_song_id_dict(train, thr, default_file_path, model_postfix):
    song_counter = collections.Counter()
    for plylst in train:
        song_counter.update(plylst['songs'])

    selected_songs = []
    song_counter = list(song_counter.items())
    for k, v in song_counter:
        if v > thr:
            selected_songs.append(k)

    print(f'{len(song_counter)} songs to {len(selected_songs)} songs')

    freq_song2id = {song: _id for _id, song in enumerate(selected_songs)}
    np.save(f'{default_file_path}/freq_song2id_thr{thr}_{model_postfix}', freq_song2id)
    print(f'{default_file_path}/freq_song2id_thr{thr}_{model_postfix} is created')
    id2freq_song = {v: k for k, v in freq_song2id.items()}
    np.save(f'{default_file_path}/id2freq_song_thr{thr}_{model_postfix}', id2freq_song)
    print(f'{default_file_path}/id2freq_song_thr{thr}_{model_postfix} is created')
