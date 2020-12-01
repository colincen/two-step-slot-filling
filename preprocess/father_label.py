slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 
'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 
'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description',
'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name',
'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 
'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city',
'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number',
'condition_description', 'condition_temperature']

father_son_slot={
    'person':['artist', 'party_size_description','playlist_owner'],
    'location':['state','city','geographic_poi','object_location_type','location_name','country','poi'],
    'special_name':['album','service','entity_name','playlist','music_item','track','movie_name','object_name',
                    'served_dish','restaurant_name','cuisine'],
    'common_name':['object_type', 'object_part_of_series_type','movie_type','restaurant_type','genre','facility',
                'condition_description','condition_temperature'],
    'number':['rating_value','best_rating','year','party_size_number','timeRange'],
    'direction':['spatial_relation','current_location','object_select'],
    'others':['rating_unit', 'sort']
}
def get_father_slot():
    res_dict = {}
    for k,v in father_son_slot.items():
        for s in v:
            res_dict[s] = k
    return res_dict

