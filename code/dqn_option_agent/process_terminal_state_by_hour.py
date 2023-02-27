from collections import defaultdict
from config.setting import NUM_REACHABLE_HEX,TERMINAL_STATE_SAVE_PATH
import numpy as np
import os
def load_h_networks():
    all_f_by_hour = np.zeros((24, NUM_REACHABLE_HEX))
    with open('../'+'logs/hex_p_value.csv', 'r') as f:
        next(f)
        for lines in f:
            line = lines.strip().split(',')
            hr, hid, f_value = line  # hour, oridin, dest, trip_time/num of trip
            all_f_by_hour[int(hr)][int(hid)] = f_value
    all_f_values = all_f_by_hour
    f_means_per_hour = np.median(all_f_by_hour, axis=1)

    f_threshold_dict = defaultdict()
    middle_mask = defaultdict()
    term_percentile = 0.05

    for hr in range(24):
        f_middle_sorted = np.sort(all_f_values[hr])
        f_lower_threshold = f_middle_sorted[int(len(f_middle_sorted) * (0.5 - term_percentile))]
        f_higher_threshold = f_middle_sorted[int(len(f_middle_sorted) * (0.5 + term_percentile))]

        f_threshold_dict[hr] = [f_lower_threshold, f_higher_threshold]

        middle_mask[hr] = [np.sign(f_value - f_threshold_dict[hr][0]) * np.sign(f_threshold_dict[hr][1] - f_value)
                           for f_value in all_f_values[hr]]
    middle_mask = middle_mask  # 24 by 1347
    if not os.path.isdir('../'+TERMINAL_STATE_SAVE_PATH):
        os.mkdir('../'+TERMINAL_STATE_SAVE_PATH)
    with open('../'+TERMINAL_STATE_SAVE_PATH + 'term_states_%d.csv' % (0), 'w') as ts:
        for hr in range(24):
            for hex_id, term_flag in enumerate(middle_mask[hr]):
                if term_flag == 1:
                    ts.writelines('{},{}\n'.format(hr, hex_id))
        print('finished record terminal state!!!')

if __name__ == '__main__':
    load_h_networks()
