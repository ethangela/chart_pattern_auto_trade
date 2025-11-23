import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from tqdm import tqdm
import pickle
from scipy.signal import argrelextrema
from collections import defaultdict
import itertools
import os
from itertools import combinations
import shutil
import math



'''helper functions'''
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices
    # return negative val if invalid

    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it is not valid
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line
    err = (diffs ** 2.0).sum()
    return err

def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    # Amount to change slope by. Multiplied by opt_step
    slope_unit = (y.max() - y.min()) / len(y)

    # Optimization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step  # current step

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert (best_err >= 0.0)  # Shouldn’t ever fail with initial slope

    get_derivative = True
    derivative = None

    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases.
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            # If increasing by a small amount fails, try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:  # Derivative failed, give up
                raise Exception("Derivative failed. Check your data.")

            get_derivative = False

        if derivative > 0.0:  # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else:  # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            # slope failed/didn’t reduce error
            curr_step *= 0.5  # Reduce step size
        else:  # test slope reduced error
            best_err = test_err
            best_slope = test_slope
            get_derivative = True  # Recompute derivative

    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])

def fit_trendlines_single(data: np.array):
    # find line of best fit (least squares)
    # coefs[0] = slope, coefs[1] = intercept
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)

def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope, coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]

    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)

def bound_plot(df_filtered_clip, ticker_name):
    #slope
    candles = df_filtered_clip[ticker_name]
    support_coefs_c, resist_coefs_c = fit_trendlines_single(candles.to_numpy())
    support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
    resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]

    def get_line_points(candles, line_points):
        idx = candles.index
        line_i = len(candles) - len(line_points)
        assert (line_i >= 0)
        points = []
        for i in range(line_i, len(candles)):
            points.append((idx[i], line_points[i - line_i]))
        return points

    s_seq2 = get_line_points(candles, support_line_c)
    r_seq2 = get_line_points(candles, resist_line_c)

    # plots
    plt.figure(figsize=(10, 8))
    plt.plot(df_filtered_clip['Date'].to_list(), candles.to_numpy(), label='Close Price')
    plt.plot(df_filtered_clip['Date'].to_list(), [item[1] for item in r_seq2], label='Upper Bound')
    plt.plot(df_filtered_clip['Date'].to_list(), [item[1] for item in s_seq2], label='Lower Bound')
    plt.title(f'Ticker {ticker_name}')
    plt.xlabel('Date')
    plt.ylabel('Close price')
    plt.legend()
    plt.savefig(f'./res_sup/{ticker_name}.png')
    plt.close()

def auto_labeling(data_list, timestamp_list, w):
    labels = np.zeros(len(data_list))
    FP = data_list[0]
    x_H = data_list[0]
    HT = timestamp_list[0]
    x_L = data_list[0]
    LT = timestamp_list[0]
    FP_N = 0
    Cid = 0

    for i in range(len(data_list)):
        if data_list[i] > FP + data_list[0] * w:
            x_H = data_list[i]
            HT = timestamp_list[i]
            FP_N = i
            Cid = 1
            break
        if data_list[i] < FP - data_list[0] * w:
            x_L = data_list[i]
            LT = timestamp_list[i]
            FP_N = i
            Cid = -1
            break

    for i in range(FP_N, len(data_list)):
        if Cid > 0:
            if data_list[i] > x_H:
                x_H = data_list[i]
                HT = timestamp_list[i]
            if data_list[i] < x_H - x_H * w and LT < HT:
                for j in range(len(data_list)):
                    if timestamp_list[j] > LT and timestamp_list[j] <= HT:
                        labels[j] = 1
                x_L = data_list[i]
                LT = timestamp_list[i]
                Cid = -1

        if Cid < 0:
            if data_list[i] < x_L:
                x_L = data_list[i]
                LT = timestamp_list[i]
            if data_list[i] > x_L + x_L * w and HT <= LT:
                for j in range(len(data_list)):
                    if timestamp_list[j] > HT and timestamp_list[j] <= LT:
                        labels[j] = -1
                x_H = data_list[i]
                HT = timestamp_list[i]
                Cid = 1

    # Post-processing
    labels[0] = labels[1]
    labels = np.where(labels == 0, Cid, labels)
    assert len(labels) == len(timestamp_list)
    timestamp2label_dict = {
        timestamp_list[i].strftime("%Y-%m-%d"): (labels[i], i)
        for i in range(len(timestamp_list) - 1)
        if (labels[i] != labels[i + 1]) or (i == 0)
    }

    return labels, timestamp2label_dict





'''generate signal'''
def price_ticker_scanner(ticker_name, close_df, date_list, thd, pattern_tuple,
                         df, price_type='daily', back_window=-252):

    if price_type == 'daily':
        scan_dates = date_list[back_window:]
    elif price_type == 'weekly':
        scan_dates = [date for date in date_list
                      if date.weekday() == date_list[-1].weekday()][back_window:]

    ticker = close_df[['Date', ticker_name]]
    ticker = ticker[ticker['Date'].isin(scan_dates)]

    first_valid_idx = ticker[ticker_name].first_valid_index()
    ticker = ticker.loc[first_valid_idx:].reset_index(drop=True)

    ticker.set_index('Date', inplace=True)
    ticker.reset_index(inplace=True)

    def get_max_min(prices, ticker_name, threshold, include_final=False):
        Date_list = prices['Date'].tolist()
        Cls_list = prices[ticker_name].values.tolist()

        _, timestamp2label_dict = auto_labeling(Cls_list, Date_list, w=threshold)
        price_local_min_dt = []
        price_local_max_dt = []

        for key, values in timestamp2label_dict.items():
            if values[0] == -1:
                price_local_min_dt.append(values[1])
            elif values[0] == 1:
                price_local_max_dt.append(values[1])

        maxima = pd.DataFrame(prices.loc[price_local_max_dt])
        minima = pd.DataFrame(prices.loc[price_local_min_dt])
        max_min = pd.concat([maxima, minima]).sort_index()
        max_min['Date_idx'] = max_min.index
        max_min = max_min.reset_index(drop=True)
        max_min = max_min[~max_min.Date_idx.duplicated()]
        max_min = max_min.reset_index(drop=True)

        if include_final:
            max_min.loc[len(max_min)] = {
                'Date_idx': prices.index[-1],
                'Date': prices.iloc[-1]['Date'],
                ticker_name: prices.iloc[-1][ticker_name]
            }

        max_min.set_index('Date_idx', inplace=True)
        return max_min[['Date', ticker_name]]

    def plot_min_max(prices, max_min, ticker_name, w):
        dst_dir = f'./df_pattern/{price_type}/granularity'
        os.makedirs(dst_dir, exist_ok=True)

        # plot
        plt.figure(figsize=(10, 8))
        plt.plot(ticker['Date'].to_list(), ticker[ticker_name].to_list(),
                 alpha=0.5, color='orange', label='Close Price')
        plt.plot(max_min['Date'].to_list(), max_min[ticker_name].to_list(),
                 alpha=1, linewidth=1, linestyle='dashed', label='Sketch')
        plt.title(f'Ticker {ticker_name} with granularity w={w}%')
        plt.xlabel('Date')
        plt.ylabel('Close price')
        plt.legend()
        plt.savefig(f'{dst_dir}/{ticker_name}_sketch_g{int(w*100)}.png')
        plt.close()

    def support_resistance(window, ticker_name):
        candles = window[ticker_name]
        df_reindexed = candles.reindex(range(candles.index.min(),
                                             candles.index.max() + 1))
        candles = df_reindexed.interpolate(method='linear')

        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles.to_numpy())
        support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
        resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]

        def get_line_points(candles, line_points):
            idx = candles.index
            line_i = len(candles) - len(line_points)
            assert line_i >= 0
            points = [(idx[i], line_points[i - line_i]) for i in range(line_i, len(candles))]
            return points

        s_seq2 = get_line_points(candles, support_line_c)
        r_seq2 = get_line_points(candles, resist_line_c)
        return s_seq2, r_seq2

    def wedge_support_resistance(window, owindow, ticker_name):
        candles = owindow[ticker_name]
        df_reindexed = candles.reindex(range(candles.index.min(),
                                             candles.index.max() + 1))
        candles_og = df_reindexed.interpolate(method='linear')
        oidx = candles_og.index.values

        candles = window[ticker_name]
        df_reindexed = candles.reindex(range(candles.index.min(),
                                             candles.index.max() + 1))
        candles = df_reindexed.interpolate(method='linear')
        nidx = candles.index.values

        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles.to_numpy())
        support_line_c = support_coefs_c[0] * np.arange(-(nidx[0]) + len(candles) +
                                                        (oidx[-1] - nidx[-1])) + support_coefs_c[1]
        resist_line_c = resist_coefs_c[0] * np.arange(-(nidx[0]) + len(candles) +
                                                      (oidx[-1] - nidx[-1])) + resist_coefs_c[1]
        def get_line_points(candles, line_points):
            idx = candles.index
            line_i = len(candles) - len(line_points)
            assert line_i >= 0
            points = [(idx[i], line_points[i - line_i]) for i in range(line_i, len(candles))]
            return points

        s_seq2 = get_line_points(candles_og, support_line_c)
        r_seq2 = get_line_points(candles_og, resist_line_c)

        return s_seq2, r_seq2

    def hs_neckline(window, ticker_name, neck_tp=None):
        candles = window[ticker_name]
        df_reindexed = candles.reindex(range(candles.index.min(),
                                             candles.index.max() + 1))
        candles_og = df_reindexed.interpolate(method='linear')
        oidx = candles_og.index.values

        candles = window.iloc[2:5][ticker_name]
        df_reindexed = candles.reindex(range(candles.index.min(),
                                             candles.index.max() + 1))
        candles = df_reindexed.interpolate(method='linear')
        nidx = candles.index.values

        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles.to_numpy())
        support_line_c = support_coefs_c[0] * np.arange(-(nidx[0]) + len(candles) +
                                                        (oidx[-1] - nidx[-1])) + support_coefs_c[1]
        resist_line_c = resist_coefs_c[0] * np.arange(-(nidx[0]) + len(candles) +
                                                      (oidx[-1] - nidx[-1])) + resist_coefs_c[1]

        def get_line_points(candles, line_points):
            idx = candles.index
            line_i = len(candles) - len(line_points)
            assert line_i >= 0
            points = [(idx[i], line_points[i - line_i]) for i in range(line_i, len(candles))]
            return points

        s_seq2 = get_line_points(candles_og, support_line_c)
        r_seq2 = get_line_points(candles_og, resist_line_c)
        
        return s_seq2, r_seq2

    def neckline(window, ticker_name, neckline_idx=2):
        candles = window[ticker_name]
        df_reindexed = candles.reindex(range(candles.index.min(),
                                             candles.index.max() + 1))
        df_reindexed = df_reindexed.interpolate(method='linear')
        candles = df_reindexed.iloc[neckline_idx:]
        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles.to_numpy())
        support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]

        def get_line_points(candles, line_points):
            idx = candles.index
            line_i = len(candles) - len(line_points)
            assert line_i >= 0
            points = [(idx[i], line_points[i - line_i]) for i in range(line_i, len(candles))]
            return points

        s_seq2 = get_line_points(candles, support_line_c)
        return s_seq2

    def slope(tups):
        x1, y1 = tups[0]
        x2, y2 = tups[-1]
        return (y2 - y1) / (x2 - x1)

    def find_patterns(max_min, ticker, ticker_name, pattern_tuple):
        patterns = defaultdict(list)
        patterns_sup = defaultdict(list)
        patterns_res = defaultdict(list)
        pat_name, pat_pts = pattern_tuple[0], pattern_tuple[1]

        # max_min range
        price_scope = max_min[ticker_name].max() - max_min[ticker_name].min()
        fixed_point = len(max_min) - 1
        selected_list = [[i for i in range(-pat_pts, 0)]]

        for l in selected_list:
            
            window = max_min.iloc[l]
            idx_list = window.index.to_list()

            if pat_name == 'double_top':
                r"""
                Double Top
                   b     d
                  / \   / \
                 /   \ /   \
                a     c     e
                """

                a, b, c, d, e = window[ticker_name]
                if a < b and abs(b - d) <= b * 0.025 and c < min(b, d) and d > e:
                    if (e > c and abs(c - e) <= c * 0.025) or (e < c and abs(e - c) <= c * 0.05):
                        s = neckline(max_min.iloc[l], ticker_name, neckline_idx=2)
                        patterns['double_top'].append(window.index.to_list())
                        patterns_sup['double_top'].append(s)

            if pat_name == 'double_bottom':
                r"""
                Double Bottom
                A     C    E
                 \   /\   /
                  \ /  \ /
                   B    D
                """
                a, b, c, d, e = window[ticker_name]
                if a > b and abs(b - d) <= b * 0.025 and c > max(b, d) and d < e:
                    if (e < c and abs(c - e) <= c * 0.025) or (e > c and abs(e - c) <= c * 0.05):
                        s = neckline(max_min.iloc[l], ticker_name, neckline_idx=2)
                        patterns['double_bottom'].append(window.index.to_list())
                        patterns_sup['double_bottom'].append(s)

            if pat_name == 'head_and_shoulders':
                r"""
                Head and Shoulders
                           D
                          /\   
                   B     /  \    F
                  / \   /    \  / \
                 /   \/_______\/___\____neckline
                /     C       E     \
               A                     G
                """
                a, b, c, d, e, f, g = window[ticker_name]
                if a < b and d > max(b, f) and max(c, e) < min(b, f) and abs(c - e) <= c * 0.025 and f > g and abs(c - a) <= c * 0.05:
                    if (g > min(c, e) and abs(g - min(c, e)) <= min(c, e) * 0.025) or (g < min(c, e) and abs(g - min(c, e)) <= min(c, e) * 0.05):
                        patterns['head_and_shoulders'].append(window.index.to_list())
                        s, r = hs_neckline(max_min.iloc[l], ticker_name)
                        patterns_sup['head_and_shoulders'].append(s)

            if pat_name == 'reverse_head_and_shoulders':
                a, b, c, d, e, f, g = window[ticker_name]
                if a > b and d < min(b, f) and min(c, e) > max(b, f) and abs(c - e) <= c * 0.025 and f < g and abs(c - a) <= c * 0.05:
                    if (g < max(c, e) and abs(g - max(c, e)) <= max(c, e) * 0.025) or (g > max(c, e) and abs(g - max(c, e)) <= max(c, e) * 0.05):
                        patterns['reverse_head_and_shoulders'].append(window.index.to_list())
                        s = neckline(max_min.iloc[l], ticker_name, neckline_idx=np.argmax([c, d, e]) + 2)
                        s, r = hs_neckline(max_min.iloc[l], ticker_name)
                        patterns_sup['reverse_head_and_shoulders'].append(s)

            if pat_name == 'rising_wedge':
                r"""
                Rising Wedge
                                 f
                           d    /\
                    b     /\   /  g
                   /\    /  \ /    
                  /  \  /    e
                 /    \/      
                /      c
               a
                """
                a, b, c, d, e, f, g = window[ticker_name]
                slope_threshold = 1.1  # TODO

                def wedge_cons(line, idx, item, idx_lst, start):
                    idx_ = (lambda i: idx_lst[i] - idx_lst[start])
                    return abs(line[idx_(idx)][1] - item) <= max(line[idx_(idx)][1], item) * 0.025

                if b<d<f and c<e<g and b>c and c<d and d>e and e<f and f>g:
                    s, r = support_resistance(max_min.iloc[l: l + 1], max_min.iloc[l], ticker_name)
                    i_list = window[ticker_name].index.to_list()
                    c1 = wedge_cons(r, 2, b, i_list, 1) and wedge_cons(s, 4, e, i_list, 1) and wedge_cons(s, 6, g, i_list, 1)
                    c2 = wedge_cons(s, 2, c, i_list, 1) and wedge_cons(s, 4, e, i_list, 1) and wedge_cons(s, 6, g, i_list, 1)
                    if c1 and c2:
                        if slope(s) / slope(r) > slope_threshold:
                            s, r = wedge_support_resistance(max_min.iloc[l: l + 1], max_min.iloc[l], ticker_name)
                            patterns['rising_wedge'].append(window.index.to_list())
                            patterns_sup['rising_wedge'].append(s)
                            patterns_res['rising_wedge'].append(r)

                elif b<d<f and c<e<g and b<c and c>d and d<e and e>f and f<g:
                    s, r = support_resistance(max_min.iloc[l: l + 1], max_min.iloc[l], ticker_name)
                    i_list = window[ticker_name].index.to_list()
                    c1 = wedge_cons(s, 1, b, i_list, 1) and wedge_cons(s, 3, d, i_list, 1) and wedge_cons(s, 5, f, i_list, 1)
                    c2 = wedge_cons(r, 2, c, i_list, 1) and wedge_cons(r, 4, e, i_list, 1) and wedge_cons(r, 6, g, i_list, 1)
                    if c1 and c2:
                        if slope(s) / slope(r) > slope_threshold:
                            s, r = wedge_support_resistance(max_min.iloc[l: l + 1], max_min.iloc[l], ticker_name)
                            patterns['rising_wedge'].append(window.index.to_list())
                            patterns_sup['rising_wedge'].append(s)
                            patterns_res['rising_wedge'].append(r)

                elif a<c<e and b<d<f and a>b and b<c and c>d and d<e and e>f:
                    s, r = wedge_support_resistance(max_min.iloc[l: l + 1], max_min.iloc[l], ticker_name)
                    if slope(s) / slope(r) > slope_threshold:
                        i_list = window[ticker_name].index.to_list()
                        c1 = wedge_cons(r, 0, a, i_list, 0) and wedge_cons(r, 2, c, i_list, 0) and wedge_cons(r, 4, e, i_list, 0)
                        c2 = wedge_cons(s, 1, b, i_list, 0) and wedge_cons(s, 3, d, i_list, 0) and wedge_cons(s, 5, f, i_list, 0)
                        if c1 and c2:
                            patterns['rising_wedge'].append(window.index.to_list())
                            patterns_sup['rising_wedge'].append(s)
                            patterns_res['rising_wedge'].append(r)

                elif a<c<e and b<d<f and a<b and b>c and c<d and d>e and e<f:
                    s, r = wedge_support_resistance(max_min.iloc[l: l + 1], max_min.iloc[l], ticker_name)
                    if slope(s) / slope(r) > slope_threshold:
                        i_list = window[ticker_name].index.to_list()
                        c1 = wedge_cons(s, 0, a, i_list, 0) and wedge_cons(s, 2, c, i_list, 0) and wedge_cons(s, 4, e, i_list, 0)
                        c2 = wedge_cons(r, 1, b, i_list, 0) and wedge_cons(r, 3, d, i_list, 0) and wedge_cons(r, 5, f, i_list, 0)
                        if c1 and c2:
                            patterns['rising_wedge'].append(window.index.to_list())
                            patterns_sup['rising_wedge'].append(s)
                            patterns_res['rising_wedge'].append(r)

            if pat_name == 'falling_wedge':
                a, b, c, d, e, f, g = window[ticker_name]

                # NOTE: same structure as 'rising_wedge' but with downward sloping lines.
                slope_threshold = 1.1  # TODO: tune

                # local helper used throughout wedge checks
                def wedge_cons(line, idx, item, idx_lst, start):
                    # distance between candidate point and the trend line at that x-index
                    _rel = idx_lst[idx] - idx_lst[start]
                    return abs(line[_rel][1] - item) <= max(line[_rel][1], item) * 0.025

                if a>c>e and b>d>f and a>b and b<c and c>d and d<e and e>f:
                    s, r = support_resistance(max_min.iloc[1:1], max_min.iloc[1], ticker_name)
                    i_list = window[ticker_name].index.to_list()
                    c1 = wedge_cons(r, 2, c, i_list, 1) and wedge_cons(r, 4, e, i_list, 1)
                    c2 = wedge_cons(s, 3, d, i_list, 1) and wedge_cons(s, 5, f, i_list, 1)
                    if c1 and c2 and (slope(r) / slope(s)) > slope_threshold:
                        patterns['falling_wedge'].append(window.index.to_list())
                        patterns_sup['falling_wedge'].append(s)
                        patterns_res['falling_wedge'].append(r)

                elif a>c>e and b>d>f and a<b and b>c and c<d and d>e and e<f:
                    s, r = support_resistance(max_min.iloc[1:1], max_min.iloc[1], ticker_name)
                    i_list = window[ticker_name].index.to_list()
                    c1 = wedge_cons(r, 2, c, i_list, 1) and wedge_cons(r, 4, e, i_list, 1)
                    c2 = wedge_cons(s, 3, d, i_list, 1) and wedge_cons(s, 5, f, i_list, 1)
                    if c1 and c2 and (slope(r) / slope(s)) > slope_threshold:
                        patterns['falling_wedge'].append(window.index.to_list())
                        patterns_sup['falling_wedge'].append(s)
                        patterns_res['falling_wedge'].append(r)

                elif b>d>f and c>e>g and b>c and c<d and d>e and e<f and f>g:
                    s, r = support_resistance(max_min.iloc[1:1], max_min.iloc[1], ticker_name)
                    i_list = window[ticker_name].index.to_list()
                    c1 = wedge_cons(r, 2, c, i_list, 1) and wedge_cons(r, 4, e, i_list, 1)
                    c2 = wedge_cons(s, 3, d, i_list, 1) and wedge_cons(s, 5, f, i_list, 1)
                    if c1 and c2 and (slope(r) / slope(s)) > slope_threshold:
                        patterns['falling_wedge'].append(window.index.to_list())
                        patterns_sup['falling_wedge'].append(s)
                        patterns_res['falling_wedge'].append(r)

                elif b>d>f and c>e>g and b<c and c>d and d<e and e>f and f<g:
                    s, r = support_resistance(max_min.iloc[1:1], max_min.iloc[1], ticker_name)
                    i_list = window[ticker_name].index.to_list()
                    c1 = wedge_cons(r, 2, c, i_list, 1) and wedge_cons(r, 4, e, i_list, 1)
                    c2 = wedge_cons(s, 3, d, i_list, 1) and wedge_cons(s, 5, f, i_list, 1)
                    if c1 and c2 and (slope(r) / slope(s)) > slope_threshold:
                        patterns['falling_wedge'].append(window.index.to_list())
                        patterns_sup['falling_wedge'].append(s)
                        patterns_res['falling_wedge'].append(r)

        if pat_name in ['rising_wedge', 'falling_wedge']:
            return patterns, patterns_sup, patterns_res
        else:
            return patterns, patterns_sup, None
        
    def plot_patterns(patterns, patterns_sup, patterns_res, ticker_name, threshold, df): 
        if not patterns:
            return False, False, None
        else:
            for key in patterns.keys():
                if patterns_res: 
                    for l, s, r in zip(patterns[key], patterns_sup[key], patterns_res[key]):
                        broken = None
                        project = None
                        if key in ['rising_wedge', 'falling_wedge']:
                            broken = s[-1][-1] > max_min[ticker_name].to_list()[-1] or r[-1][-1] < max_min[ticker_name].to_list()[-1]
                else:
                    for l,s in zip(patterns[key], patterns_sup[key]):
                        broken = None
                        project = None
                        if key in ['head_and_shoulders', 'double_top']:
                            broken = s[-1][-1] > max_min[ticker_name].to_list()[-1]
                            if key == 'double_top':
                                project = s[0][-1] - max_min[ticker_name].to_list()[0]
                            elif key == 'head_and_shoulders':
                                head_ = max_min.loc[1][ticker_name].to_list()[3]
                                date_ = max_min.loc[1]['Date'].to_list()[3]
                                index_ = ticker.iloc[[e[0] for e in s]]['Date'].to_list().index(date_)
                                project = head_ - [e[1] for e in s][index_]

                        elif key in ['reverse_head_and_shoulders', 'double_bottom']:
                            broken = s[-1][-1] < max_min[ticker_name].to_list()[-1]

                            if key == 'double_bottom':
                                project = max_min[ticker_name].to_list()[0] - s[0][-1]
                            elif key == 'reverse_head_and_shoulders':
                                head_ = max_min.loc[1][ticker_name].to_list()[3]  
                                date_ = max_min.loc[1]['Date'].to_list()[3]
                                index_ = ticker.iloc[[e[0] for e in s]]['Date'].to_list().index(date_)
                                project = [e[1] for e in s][index_] - head_

            # return on first pattern family processed
            broken_flag = 'b' if broken is True else ('u' if broken is False else None)
            return ticker_name, list(patterns.keys())[0], broken_flag, project

    max_min = get_max_min(ticker, ticker_name, threashold=thd, include_final=True)
    pat_dic, sup_dic, res_dic = find_patterns(max_min, ticker, ticker_name, pattern_tuple) 
    tkk, ptt, b_idc, p_profit = plot_patterns (pat_dic, sup_dic, res_dic, ticker_name, thd, df)

    return tkk, ptt, b_idc, p_profit

def trendline_scanner(ticker_name, close_df, date_list, df, price_type='daily', back_window=-252):
    if price_type == 'daily':
        scan_dates = date_list[back_window:]
    elif price_type == 'weekly':
        scan_dates = [d for d in date_list if d.weekday() == date_list[-1].weekday()][back_window:]

    ticker = close_df[['Date', ticker_name]]
    ticker = ticker[ticker['Date'].isin(scan_dates)]
    ticker.set_index('Date', inplace=True)
    ticker.reset_index(inplace=True)  # keep a clean continuous index if needed

    # inner util: support & resistance trendlines for a window
    def support_resistance(window, tname):
        candles = window[tname]
        # fill missing indices (linear interpolation across gaps)
        df_reindexed = candles.reindex(range(candles.index.min(), candles.index.max() + 1))
        candles = df_reindexed.interpolate(method='linear')

        # line of best fit (coefs[0]=slope, coefs[1]=intercept)
        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles.to_numpy())
        support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
        resist_line_c  = resist_coefs_c[0]  * np.arange(len(candles)) + resist_coefs_c[1]

        def get_line_points(candles_, line_points):
            idx = candles_.index
            line_i = len(candles_) - len(line_points)
            assert line_i >= 0
            pts = []
            for i in range(line_i, len(candles_)):
                pts.append((idx[i], line_points[i - line_i]))
            return pts

        s_seq2 = get_line_points(candles, support_line_c)
        r_seq2 = get_line_points(candles, resist_line_c)
        return s_seq2, r_seq2

    # small helpers shared by wedge checks
    def slope(tups):
        x1, y1 = tups[0]
        x2, y2 = tups[-1]
        return (y2 - y1) / (x2 - x1)

    def ascending_support(prices, tname, _df): 
        s, r = support_resistance(prices, ticker_name)

        condition1 = (
            np.absolute(prices[ticker_name].to_numpy()[-1] - s[-1][1])
            / prices[ticker_name].to_numpy()[-1]
            <= 0.03
        )

        top3 = np.sort(
            prices[ticker_name].to_numpy() - np.array([item[1] for item in s])
        )[:3] #[::-1][:3] TODO
        top3_idx = np.argsort(
            prices[ticker_name].to_numpy() - np.array([item[1] for item in s])
        )[:3] #[::-1][:3] TODO

        condition2 = [a <= 3 for a in top3].count(True) == 3
        condition3 = slope(s) > 0
        condition4 = (
            [
                abs(tup[0] - tup[1]) >= 0.1 * len(prices[ticker_name].to_numpy())
                for tup in list(combinations(top3_idx, 2))
            ].count(True)
            == 3
        )

        if condition1 and condition2 and condition3 and condition4:
  
            if prices[ticker_name].to_numpy()[-1] - s[-1][1] > 0:
                broken = "unbroken"
            else:
                broken = "broken"
            return ticker_name, "ascending_support", broken
        else:
            return False, False, False

    def descending_resistance(prices, ticker_name, df):
        s, r = support_resistance(prices, ticker_name)

        condition1 = (
            np.absolute(prices[ticker_name].to_numpy()[-1] - r[-1][1])
            / prices[ticker_name].to_numpy()[-1]
            <= 0.03
        )

        top3 = np.sort(np.array([item[1] for item in r]) - prices[ticker_name].to_numpy())[:3]
        top3_idx = np.argsort(np.array([item[1] for item in r]) - prices[ticker_name].to_numpy())[:3]
        condition2 = [a <= 3 for a in top3].count(True) == 3

        condition3 = slope(r) < 0

        condition4 = (
            [
                abs(tup[0] - tup[1]) >= 0.1 * len(prices[ticker_name].to_numpy())
                for tup in list(combinations(top3_idx, 2))
            ].count(True)
            == 3
        )

        if condition1 and condition2 and condition3 and condition4:
            if prices[ticker_name].to_numpy()[-1] - r[-1][1] < 0:
                broken = "unbroken"
            else:
                broken = "broken"
            return ticker_name, "descending_resistance", broken
        else:
            return False, False, False
        
    atkk, aptt, ab = ascending_support(ticker, ticker_name, df) 
    dtkk, dptt, db = descending_resistance(ticker, ticker_name, df)

    return atkk, aptt, dtkk, dptt, ab, db

def into_trading_book(ticker, pattern, profit, date, last_price, index_last_price, threshold):
    threshold = f"{int(threshold*100)}"

    if os.path.exists(f'./strategy_pattern_record_{threshold}.pickle'):
        with open(f'./strategy_pattern_record_{threshold}.pickle', 'rb') as handle:
            portfolio_record = pickle.load(handle)
    else:
        portfolio_record = {}

    if pattern == 'double_top' or pattern == 'head_and_shoulders':
        position = -1
    elif pattern == 'double_bottom' or pattern == 'reverse_head_and_shoulders':
        position = 1

    if (ticker, pattern) not in portfolio_record.keys():
        portfolio_record[(ticker, pattern)] = [profit, date, position, last_price, index_last_price]
        with open(f'./strategy_pattern_record_{threshold}.pickle', 'wb') as handle:
            pickle.dump(portfolio_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        pass





'''daily singal and trade'''
def signal(close_df, lag=None):
    if lag:
        close_df_ = close_df.copy().iloc[:lag]
        date_list_ = date_list[:lag]
        date_format_ = date_list_[-1].strftime('%Y%m%d')
    else:
        close_df_ = close_df.copy()
        date_list_ = date_list
        date_format_ = date_list_[-1].strftime('%Y%m%d')

    pattern_tuples = [
        ('double_top', 5),
        ('double_bottom', 5),
        ('head_and_shoulders', 7),
        ('reverse_head_and_shoulders', 7),
    ]

    for ticker_name in tqdm(liq_ticker_names):  # #liq_ticker_names + pm_list

        for ptt in pattern_tuples:
            for thd_ in granularity_list: #four levels of granularity, which can be further explored
                try:
                    dttkk, dptt, db_idc, d_profit = price_ticker_scanner(
                        ticker_name, close_df_, date_list_, thd=thd_, pattern_tuple=ptt, df=date_format_, price_type='daily'
                    )
                    if dttkk and dptt:
                        # auto trading
                        if db_idc == 'b' and d_profit > 0:
                            into_trading_book(
                                dttkk, dptt, d_profit, date_list_[-1],
                                close_df_[ticker_name].to_list()[-1],
                                close_df_['BSE500 Index'].to_list()[-1],
                                thd_
                            )
                except:
                    pass

def trade(thr, close_df, lag=None, stop_day=42):
    threshold = f"{int(thr*100)}"

    if lag:
        close_df_ = close_df.copy().iloc[:lag]
        date_list_ = date_list[:lag]
    else:
        close_df_ = close_df.copy()
        date_list_ = date_list

    if os.path.exists(f'./strategy_pattern_trading_{threshold}.pickle'):
        with open(f'./strategy_pattern_trading_{threshold}.pickle', 'rb') as handle:
            portfolio_book = pickle.load(handle)
    else:
        portfolio_book = defaultdict(list)

    if os.path.exists(f'./strategy_pattern_daily_return_{threshold}.pickle'):
        with open(f'./strategy_pattern_daily_return_{threshold}.pickle', 'rb') as handle:
            df_daily = pickle.load(handle)
    else:
        df_daily = defaultdict(lambda: defaultdict(dict))

    if os.path.exists(f'./strategy_pattern_record_{threshold}.pickle'):
        with open(f'./strategy_pattern_record_{threshold}.pickle', 'rb') as handle:
            portfolio_record = pickle.load(handle)
    else:
        portfolio_record = {}

    for key, value in portfolio_record.copy().items():
        # stop threshold 1
        profit_loss = portfolio_record[key][2] * (close_df_[key[0]].to_list()[-1] - portfolio_record[key][3])
        rtn = portfolio_record[key][2] * (close_df_[key[0]].to_list()[-1] / portfolio_record[key][3] - 1)
        hedge_rtn = -1 * portfolio_record[key][2] * (close_df_['index'].to_list()[-1] / portfolio_record[key][4] - 1)  # added hedge
        df_daily[key][date_list_[-1]] = rtn + hedge_rtn

        # end trading
        if (
            profit_loss >= portfolio_record[key][0]
            or profit_loss <= -1/3 * portfolio_record[key][0]
            or date_list_.index(date_list_[-1]) - date_list_.index(portfolio_record[key][1]) > stop_day
        ):
            portfolio_book['Ticker'].append(key[0])
            portfolio_book['Pattern'].append(key[1])
            portfolio_book['TradingStartDate'].append(portfolio_record[key][1])
            portfolio_book['TradingStartClose'].append(portfolio_record[key][3])
            portfolio_book['Position'].append(portfolio_record[key][2])
            portfolio_book['TradingEndDate'].append(date_list_[-1])
            portfolio_book['TradingEndClose'].append(close_df_[key[0]].to_list()[-1])
            portfolio_book['TickerReturn'].append(rtn)
            portfolio_book['IndexHedgeReturn'].append(hedge_rtn)
            portfolio_book['Return'].append(rtn + hedge_rtn)
            del portfolio_record[key]
            # print(f'{key} ended trading with return {rtn}')

    with open(f'./strategy_pattern_record_{threshold}.pickle', 'wb') as handle:
        pickle.dump(portfolio_record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if len(portfolio_book) > 0:
        with open(f'./strategy_pattern_trading_{threshold}.pickle', 'wb') as handle:
            pickle.dump(portfolio_book, handle, protocol=pickle.HIGHEST_PROTOCOL)
        df_book = pd.DataFrame.from_dict(portfolio_book)
        df_book.to_csv(f'./strategy_pattern_trading_{threshold}.csv', index=False)

    df_daily = dict(df_daily)
    with open(f'./strategy_pattern_daily_return_{threshold}.pickle', 'wb') as handle:
        pickle.dump(df_daily, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df_daily = pd.DataFrame(df_daily).T  # transpose after converting to DataFrame
    df_daily = df_daily[sorted(df_daily.columns)]
    df_daily.to_csv(f'./strategy_pattern_daily_return_{threshold}.csv')



if __name__ == "__main__":

    '''fixed paras'''
    granularity_list = [0.03, 0.06, 0.09, 0.12] #four levels of granularity, which can be further explored



    '''load data'''
    # daily closing prices
    close_df = pd.read_csv('DailyClosingPrices.csv', header=None) #TODO load your own data path
    close_df['Date'] = pd.to_datetime(close_df['Date'])
    date_list = close_df['Date'].dt.date.unique().tolist()
    date_format = date_list[-1].strftime('%d%m%Y')
    ref_df = pd.read_csv('TickerName.csv') #TODO load your own data path
    liq_ticker_names = list(ref_df['Ticker-NSE'].values)



    '''backtesting for the past 5 years'''
    for i in range(-252*5+1, 0, 1):
        print(f'lag: {i}')
        signal(i)
        for thd_ in granularity_list:
            trade(thd_, i)

    signal()
    for thd_ in granularity_list:
        trade(thd_)



    '''daily pnl generate (from returns to date to daily return)'''
    for threshold in [3, 6, 9, 12]:
        df_daily = pd.read_csv(f'./strategy_pattern_daily_return_{threshold}.csv')
        df_daily.replace(0.0, np.nan, inplace=True)

        def conditional_operation(df):
            df_new = df.copy()
            for row in range(df.shape[0]):  
                for col in range(1, df.shape[1]): 
                    current_cell = df.iloc[row, col]
                    left_cell = df.iloc[row, col - 1]
                    if not pd.isna(current_cell) and not pd.isna(left_cell):
                        df_new.iloc[row, col] = (1 + current_cell) * (1 + left_cell) - 1
            return df_new

        updated_df = conditional_operation(df_daily)
        assert df_daily.isna().sum().sum() == updated_df.isna().sum().sum()
        updated_df.to_csv(f'./strategy_pattern_daily_return_{threshold}_daily.csv', index=False)
