### Load ground trugh

import numpy as np

# Fixed parameters
N_SECTOR    = 9
N_MAXPSN    = 8

def load(filepath, NumTestSamples, max_person_sector):
    
    with open(filepath) as file:
        data_i = 0
        location_array      = np.zeros((NumTestSamples, (max_person_sector + 1) * N_SECTOR))
        location_array_ev   = np.zeros((NumTestSamples, N_SECTOR))
        
        count_array = np.zeros((NumTestSamples, N_MAXPSN))
        for item in file:
            location_array_local = np.zeros(N_SECTOR)
            for item_i in item:
                sec_index = ord(item_i) - ord('A')
                if sec_index >= 0:
                    location_array_local[sec_index] += 1
                            
            for sector_i in range(N_SECTOR):
                if location_array_local[sector_i] > max_person_sector:
                    count_person = max_person_sector
                else:
                    count_person = location_array_local[sector_i]
                location_array[data_i, int(count_person + sector_i * (max_person_sector + 1))] = 1
            count_array[data_i, int(location_array_local.sum()-1)] = 1
            location_array_ev[data_i] = location_array_local
            
            data_i += 1
            if data_i >= NumTestSamples:
                break
            
    return count_array, location_array, location_array_ev
