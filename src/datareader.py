import numpy as np
import logging
logger = logging.getLogger()

slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 
'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 
'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description',
'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name',
'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 
'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city',
'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number',
'condition_description', 'condition_temperature']


# father_son_slot={
#     'person':['artist', 'party_size_description','playlist_owner'],
#     'location':['state','city','geographic_poi','country','poi'],
#     'special_name':['album','service','entity_name','playlist','music_item',
#                     'track','movie_name','object_name',
#                     'served_dish','restaurant_name','cuisine'],
#     'common_name':['object_type', 'object_part_of_series_type','movie_type',
#                     'restaurant_type','genre','facility',
#                 'condition_description','condition_temperature',
#                 'object_location_type','location_name'],
#     'number':['rating_value','best_rating','year','party_size_number','timeRange'],
#     'direction':['spatial_relation','current_location','object_select'],
#     'others':['rating_unit', 'sort']
# }

father_keys = ['person', 'location', 'special_name', 'common_name', 'number', 'direction', 'others']

y0_set = ['O','B','I']
y1_set = ['O','B-person','I-person','B-location','I-location','B-special_name','I-special_name','B-common_name','I-common_name',
'B-number','I-number','B-direction','I-direction','B-others','I-others']
y2_set = ['O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 
'I-geographic_poi', 'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name', 
'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type', 
'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service',
 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish', 
 'B-genre',  'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 
 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort',
  'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation',
   'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type', 
   'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value',
    'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description', 'B-condition_temperature']
domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]


father_son_slot={
    'person':['artist','playlist_owner','party_size_description'],
    'location':['state','city','geographic_poi','object_location_type','location_name','country','poi'],
    'special_name':['album','service','entity_name','playlist','music_item','track','movie_name','object_name',
                    'served_dish','restaurant_name','cuisine'],
    'common_name':['object_type', 'object_part_of_series_type','movie_type','restaurant_type','genre','facility',
                'condition_description','condition_temperature'],
    'number':['rating_value','best_rating','year','party_size_number','timeRange'],
    'direction':['spatial_relation','current_location','object_select'],
    'others':['rating_unit', 'sort']
}
domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],

    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 
    'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],

    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

SLOT_PAD = 0
PAD_INDEX = 0
UNK_INDEX = 1


class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX, "UNK":UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2
    def index_words(self, sentence):
        for word in sentence:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1

def get_father_slot():
    res_dict = {}
    for k,v in father_son_slot.items():
        for s in v:
            res_dict[s] = k
    return res_dict

def read_file(filepath, vocab, son_to_fa_slot, use_label_encoder, domain=None):
    utter_list, y0_list, y1_list, y2_list, template_list = [], [], [], [], []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            splits = line.split("\t")
            tokens = splits[0].split()
            l2_list = splits[1].split()

            utter_list.append(tokens)
            y2_list.append(l2_list)

            vocab.index_words(tokens)

            l1_list = []
            l0_list = []
            for l in l2_list:
                if "B" in l:
                    l1_list.append("B-"+son_to_fa_slot[l[2:]])
                    l0_list.append("B")
                elif "I" in l:
                    l1_list.append("I-"+son_to_fa_slot[l[2:]])
                    l0_list.append("I")
                else:
                    l1_list.append("O")
                    l0_list.append("O")
            y1_list.append(l1_list)
            y0_list.append(l0_list)

            if use_label_encoder:
                template_each_sample = [[], [], []]
                assert len(tokens) == len(l2_list)
                for token, l2 in zip(tokens, l2_list):
                    if "I" in l2: continue
                    if l2 == "O":
                        template_each_sample[0].append(token)
                        template_each_sample[1].append(token)
                        template_each_sample[2].append(token) 
                    else:
                        slot_name = l2.split("-")[1]
                        template_each_sample[0].append(slot_name)
                        np.random.shuffle(slot_list)
                        idx = 0
                        for j in range(1, 3):  # j from 1 to 2
                            if slot_list[idx] != slot_name:
                                template_each_sample[j].append(slot_list[idx])
                            else:
                                idx = idx + 1
                                template_each_sample[j].append(slot_list[idx])
                            idx = idx + 1
                        
                assert len(template_each_sample[0]) == len(template_each_sample[1]) == len(template_each_sample[2])

                template_list.append(template_each_sample)

    if use_label_encoder:
        data_dict = {"utter": utter_list,"y0":y0_list, "y1": y1_list, "y2": y2_list, "template_list": template_list}
    else:
        data_dict = {"utter": utter_list,"y0":y0_list, "y1": y1_list, "y2": y2_list}
    
    return data_dict, vocab
        
def binarize_data(data, vocab, dm, use_label_encoder):
    if use_label_encoder:
        data_bin = {"utter": [],"y0":[], "y1": [], "y2": [], "domains": [], "template_list": []}
    else:
        data_bin = {"utter": [],"y0":[], "y1": [], "y2": [], "domains": []}
    assert len(data_bin["utter"]) == len(data_bin["y1"]) == len(data_bin["y2"]) == len(data_bin["y0"])
    dm_idx = domain_set.index(dm)
    for utter_tokens, y0_list, y1_list, y2_list in zip(data['utter'], data['y0'], data['y1'], data['y2']):
        utter_bin, y0_bin, y1_bin, y2_bin = [], [], [], []
        for token in utter_tokens:
            utter_bin.append(vocab.word2index[token])
        data_bin['utter'].append(utter_bin)

        for y0 in y0_list:
            y0_bin.append(y0_set.index(y0))
        data_bin['y0'].append(y0_bin)

        for y1 in y1_list:
            y1_bin.append(y1_set.index(y1))
        data_bin['y1'].append(y1_bin)

        for y2 in y2_list:
            y2_bin.append(y2_set.index(y2))
        data_bin['y2'].append(y2_bin)

        assert len(utter_bin) == len(y1_bin) == len(y2_bin)== len(y0_bin)

        data_bin['domains'].append(dm_idx)

    if use_label_encoder:
        for template_each_sample in data['template_list']:
            template_each_sample_bin = [[],[],[]]
            for tok1, tok2, tok3 in zip(template_each_sample[0], template_each_sample[1], template_each_sample[2]):
                template_each_sample_bin[0].append(vocab.word2index[tok1])
                template_each_sample_bin[1].append(vocab.word2index[tok2])
                template_each_sample_bin[2].append(vocab.word2index[tok3])
            data_bin['template_list'].append(template_each_sample_bin)
    
    return data_bin

def datareader(use_label_encoder=False, prefix_path='/home/sh/data/coachdata/'):
    logger.info("Loading and processing data ...")

    data = {"AddToPlaylist": {}, "BookRestaurant": {}, "GetWeather": {}, "PlayMusic": {}, "RateBook": {}, "SearchCreativeWork": {}, "SearchScreeningEvent": {}}

    # load data and build vocab
    vocab = Vocab()
    
    son_to_fa_slot = get_father_slot()


    AddToPlaylistData, vocab = read_file(prefix_path+"snips/AddToPlaylist/AddToPlaylist.txt", vocab, son_to_fa_slot, use_label_encoder, domain="AddToPlaylist")
    BookRestaurantData, vocab = read_file(prefix_path+"snips/BookRestaurant/BookRestaurant.txt", vocab, son_to_fa_slot, use_label_encoder, domain="BookRestaurant")
    GetWeatherData, vocab = read_file(prefix_path+"snips/GetWeather/GetWeather.txt", vocab, son_to_fa_slot, use_label_encoder, domain="GetWeather")
    PlayMusicData, vocab = read_file(prefix_path+"snips/PlayMusic/PlayMusic.txt", vocab, son_to_fa_slot, use_label_encoder, domain="PlayMusic")
    RateBookData, vocab = read_file(prefix_path+"snips/RateBook/RateBook.txt", vocab, son_to_fa_slot, use_label_encoder, domain="RateBook")
    SearchCreativeWorkData, vocab = read_file(prefix_path+"snips/SearchCreativeWork/SearchCreativeWork.txt", vocab, son_to_fa_slot, use_label_encoder, domain="SearchCreativeWork")
    SearchScreeningEventData, vocab = read_file(prefix_path+"snips/SearchScreeningEvent/SearchScreeningEvent.txt", vocab, son_to_fa_slot, use_label_encoder, domain="SearchScreeningEvent")

    if use_label_encoder:
        # update slot names into vocabulary
        vocab.index_words(slot_list)

    # binarize data
    data["AddToPlaylist"] = binarize_data(AddToPlaylistData, vocab, "AddToPlaylist", use_label_encoder)
    data["BookRestaurant"] = binarize_data(BookRestaurantData, vocab, "BookRestaurant", use_label_encoder)
    data["GetWeather"] = binarize_data(GetWeatherData, vocab, "GetWeather", use_label_encoder)
    data["PlayMusic"] = binarize_data(PlayMusicData, vocab, "PlayMusic", use_label_encoder)
    data["RateBook"] = binarize_data(RateBookData, vocab, "RateBook", use_label_encoder)
    data["SearchCreativeWork"] = binarize_data(SearchCreativeWorkData, vocab, "SearchCreativeWork", use_label_encoder)
    data["SearchScreeningEvent"] = binarize_data(SearchScreeningEventData, vocab, "SearchScreeningEvent", use_label_encoder)

    return data, vocab





def queryexample(domain, slot, num):
    vocab = Vocab()
    son_to_fa_slot = get_father_slot()
    prefix_path = '../coachdata/'
    Data, vocab = read_file(prefix_path+"snips/"+domain+"/"+domain+".txt", vocab, son_to_fa_slot, use_label_encoder = False, domain=domain)
    res_data = []
    for i in range(len(Data['utter'])):
        for k in Data['y2'][i]:
            if slot in k:
                res_data.append([Data['utter'][i], Data['y2'][i]])
                break
    return res_data[:num]

# res = queryexample('RateBook', 'object_part_of_series_type', 20)
# for i in res:
#     print(i)
    



