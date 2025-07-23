import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from math import floor, ceil
from datetime import datetime as dt
from collections import OrderedDict
import calendar
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import spm1d

from string import ascii_uppercase
import seaborn as sns

from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway

import utils
from utils import run_lmm, run_glm


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datadir',
        default='./data',
        type=str,
        help='folder containing the activity_data.csv and sleep_data.csv files',
    )

    parser.add_argument(
        '--outputdir',
        default='./figures/oura_collective_behavior_public',
        type=str,
        help='output directory',
    )

    return parser.parse_args(args)

def plot_daily_compliance(activity_lbl_data_list, sleep_lbl_data_list, add_tgv_lines=True, ax=None, lgd_kwargs=None, outputfile=None):
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure
    
    if not isinstance(activity_lbl_data_list, list):
        activity_lbl_data_list = [activity_lbl_data_list]
    if not isinstance(sleep_lbl_data_list, list):
        sleep_lbl_data_list = [sleep_lbl_data_list]

    
    def _get_avg(data):
        res = data.groupby('summary_date')['record_id'].nunique() / data['record_id'].nunique()
        return res

    colors = plt.get_cmap('tab10').colors
    for i, item in enumerate(activity_lbl_data_list):
        try:
            activity_data, lbl = item
        except ValueError:
            activity_data = item
            lbl = None
        ax.plot(_get_avg(activity_data).index, _get_avg(activity_data), label=lbl, linestyle='solid', color=colors[i])
    for i, item in enumerate(sleep_lbl_data_list):
        try:
            sleep_data, lbl = item
        except ValueError:
            sleep_data = item
            lbl = None
        ax.plot(_get_avg(sleep_data).index, _get_avg(sleep_data), label=lbl, linestyle='dashed', color=colors[i])

    datelocator_major = mdates.MonthLocator(bymonthday=[1, 7, 14, 21, 28])
    ax.xaxis.set_major_locator(datelocator_major)
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_locator(datelocator_major)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    for label in ax.get_xmajorticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
        
    if add_tgv_lines:
        ax.axvline(dt.strptime('2022-11-19', '%Y-%m-%d'), color='black', linestyle='dotted')
        ax.axvline(dt.strptime('2022-11-27', '%Y-%m-%d'), color='black', linestyle='dotted')
        ax.text(dt.strptime('2022-11-20 23:00', '%Y-%m-%d %H:%M'), 0.66, 'school\nbreak', ha='left', fontsize=8)
        ax.annotate(xy=(dt.strptime('2022-11-21', '%Y-%m-%d'), 0.675),
                    xytext=(dt.strptime('2022-11-18 20:00', '%Y-%m-%d %H:%M'), 0.675),
                    text='',
                    arrowprops=dict(arrowstyle='<-')
                )
        ax.annotate(xy=(dt.strptime('2022-11-25 11:00', '%Y-%m-%d %H:%M'), 0.675),
                    xytext=(dt.strptime('2022-11-27 17:00', '%Y-%m-%d %H:%M'), 0.675),
                    text='',
                    arrowprops=dict(arrowstyle='<-')
                )
    
        ax.axvline(dt.strptime('2022-12-12', '%Y-%m-%d'), color='black', linestyle='dashdot', linewidth=1)
        ax.axvline(dt.strptime('2022-12-16', '%Y-%m-%d'), color='black', linestyle='dashdot', linewidth=1)
        ax.text(dt.strptime('2022-12-14 00:00', '%Y-%m-%d %H:%M'), 0.6, 'final\nexams', ha='center', fontsize=8)
        ax.annotate(xy=(dt.strptime('2022-12-11 20:00', '%Y-%m-%d %H:%M'), 0.635),
                    xytext=(dt.strptime('2022-12-16 16:00', '%Y-%m-%d %H:%M'), 0.635),
                    text='',
                    arrowprops=dict(arrowstyle='<->')
                )
    
    # ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0, xmax=1))
    handles, labels = ax.get_legend_handles_labels()
    
    num_extra_labels = len(activity_lbl_data_list)

    handles = [
               Line2D([0], [0], color='black'),
               Line2D([0], [0], color='black', linestyle='dashed'),
           ] + handles[:num_extra_labels]
    labels = ['activity', 'sleep'] + labels[:num_extra_labels]

    if lgd_kwargs is None:
        lgd_kwargs = {}
    ax.legend(handles=handles, labels=labels, **lgd_kwargs)
    
    
    ax.grid(visible=True, which='both', axis='y')
    ax.set_ylabel('% of participants')
    
    if outputfile is not None:
        utils.savefig_multext(fig, outputfile)
        
    return ax


class UserPctWear(object):
    def __init__(self, activity_data, mode='average', maxdays=None):
        self._activity_data = activity_data
        self.set_user_pctwear_5min_full(mode=mode, maxdays=maxdays)
        
    @property
    def activity_data(self):
        return self._activity_data

    @property
    def user_pctwear_5min_full(self):
        return self._user_pctwear_5min_full

    @property
    def user_wear_days_5min_full(self):
        return self._user_wear_days_5min_full

    def set_user_pctwear_5min_full(self, mode='average', maxdays=None):
        df = self.activity_data.copy()
        df['class_5min_binary'] = df['class_5min_binary'].map(lambda x: np.array([int(i) for i in list(x)]))

        def _average_arrays(data, mode, maxdays=None):
            # Concatenate all arrays in the group into one large array
            all_arrays = np.stack(data.values)
            # Calculate the average
            if mode == 'average':
                res = all_arrays.sum(axis=0) / len(data)
            elif mode == 'max':
                assert maxdays is not None
                res = all_arrays.sum(axis=0) / maxdays

            return res

        srs = df.groupby('record_id')['class_5min_binary'].agg(
            lambda x: _average_arrays(x, mode, maxdays))
        srs_size = df.groupby('record_id').agg('size')
        srs.rename_axis('record_id', inplace=True)

        self._user_pctwear_5min_full = srs
        self._user_wear_days_5min_full = srs_size
    
    def get_user_pctwear_5min(self, min_days_with_data=None):
        srs = self.user_pctwear_5min_full
        srs_size = self._user_wear_days_5min_full

        if min_days_with_data is not None:
            idx_to_incl = srs_size[srs_size >= min_days_with_data].index
            print(len(srs))
            srs = srs.loc[idx_to_incl]
            print(len(srs))

        return srs
    

def plot_user_pctwear(user_pctwear_obj, min_days_with_data_list=None, ax=None, label_prefix=None, xaxis_as_time=True,
                      add_min_days_to_lbl=True,
                      outputfile=None,
                      **plot_kwargs):
    if min_days_with_data_list is None:
        min_days_with_data_list = [None]
    
    if ax is None:
        fig, ax = plt.subplots(**plot_kwargs)
    
    if label_prefix is None:
        label_prefix = ''

    start_timestamp = pd.Timestamp(year=2000, month=1, day=1, hour=4)
    for i, min_days_with_data in enumerate(min_days_with_data_list):
        srs = user_pctwear_obj.get_user_pctwear_5min(min_days_with_data=min_days_with_data)
        data = pd.DataFrame(srs.tolist(), index=srs.index)
        if xaxis_as_time:
            data.columns = [start_timestamp + pd.Timedelta(minutes=5*j) for j in data.columns]
        else:
            data.columns = data.columns * 5 / 60
        spl = data.stack().rename('pct')
        spl.index.rename({None: 'time'}, inplace=True)
        spl.index = spl.index.swaplevel(1,0)
        spl = spl.reset_index()
        if label_prefix:
            lbl = f'{label_prefix}'
        else:
            lbl = None
        if add_min_days_to_lbl:
            lbl = f'{lbl}min_days_with_data={min_days_with_data}'
        sns.lineplot(data=spl, x='time', y='pct', ax=ax, label=lbl, **plot_kwargs)
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

        ax.set_ylabel('% of days ring wear avgd over all users')
        
        if xaxis_as_time:
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(range(4, 24, 3)) + list(range(1, 4, 3))))
            ax.xaxis.set_minor_locator(mdates.HourLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%I%p'))
            # ax.set_xlim(left=start_timestamp - pd.Timedelta(hours=1))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    if outputfile is not None:
        utils.savefig_multext(outputfile)


def plot_user_pctwear_in_days_with_data(activity_data_full, ax=None,
                                        outputfile=None,
                                        userpctwear_kwargs=None,
                                        userpctwear_weekend_kwargs=None,
                                        add_min_days_to_lbl=True,
                                        plot_kwargs=None,
                                        weekday_lbl_prefix=None, weekend_lbl_prefix=None):
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure
    
    if userpctwear_kwargs is None:
        userpctwear_kwargs = {}
    
    if userpctwear_weekend_kwargs is None:
        userpctwear_weekend_kwargs = userpctwear_kwargs
        
    if plot_kwargs is None:
        plot_kwargs = {}

    if weekday_lbl_prefix is None:
        weekday_lbl_prefix = ''
    if weekend_lbl_prefix is None:
        weekend_lbl_prefix = ''

    weekday_lbl = f'{weekday_lbl_prefix}weekday'
    weekend_lbl = f'{weekend_lbl_prefix}weekend'
    if add_min_days_to_lbl:
        weekday_lbl = f'{weekday_lbl}_'
        weekend_lbl = f'{weekend_lbl}_'
    user_pctwear_weekday = UserPctWear(activity_data_full.query('dayofweek < 5'), **userpctwear_kwargs)
    plot_user_pctwear(user_pctwear_weekday,
                      ax=ax,
                      label_prefix=weekday_lbl,
                      add_min_days_to_lbl=add_min_days_to_lbl,
                      **plot_kwargs
                     )
    
    user_pctwear_weekend = UserPctWear(activity_data_full.query('dayofweek >= 5'), **userpctwear_weekend_kwargs)
    plot_user_pctwear(user_pctwear_weekend,
                      ax=ax,
                      label_prefix=weekend_lbl,
                      add_min_days_to_lbl=add_min_days_to_lbl,
                      **plot_kwargs
                     )
    
    ax.legend(fontsize=7)
    
    if outputfile is not None:
        utils.savefig_multext(fig, outputfile)
    
    return ax


class Sleeptime(object):
    def __init__(self, sleep_df, cols=None, non_timecols=None):
        if cols is None:
            cols = ['bedtime_start_delta', 'bedtime_end_delta']
        if non_timecols is None:
            non_timecols = ['is_weekday', 'dayofweek', 'gender_fmnb',
                            'dysfunc_depr_anx']
    
        self._cols = cols
        self._non_timecols = non_timecols

        self.data = self.set_data(sleep_df)

    @property
    def cols(self):
        return self._cols
    
    @property
    def non_timecols(self):
        return self._non_timecols 
    
    def set_data(self, sleep_df):
        cols = self.cols
        non_timecols = self.non_timecols
        sleeptimes = sleep_df.set_index(['record_id', 'weeknum', 'summary_date'])[cols + non_timecols].copy()
        for col in cols:
            sleeptimes[col] = sleeptimes[col].astype(int)
            sleeptimes[f'{col}_dt'] = sleeptimes[col].map(lambda x: dt.strftime(dt.strptime('2000-01-01', '%Y-%m-%d') + pd.Timedelta(seconds=x), '%I:%M %p'))
        
        # clock positions
        for col in cols:
            sleeptimes[f'{col}_adj'] = sleeptimes[f'{col}_dt'].map(lambda x:
                                                                    # pd.Timedelta(seconds=((dt.strptime(x, '%I:%M %p').hour % 12)-3)*3600 + dt.strptime(x, '%I:%M %p').minute * 60)
                                                                    (dt.strptime(x, '%I:%M %p').hour % 12)*3600 + dt.strptime(x, '%I:%M %p').minute * 60
                                                                  )
        
        return sleeptimes

    def get_histogram(self, vals):
        counts, angles = np.histogram(vals,
                                      weights=[1/len(vals) for i in range(len(vals))],
                                      bins=np.arange(0, 3600*12 + 60, 60))
        
        return counts, angles
    
    def plot_clock_sleeptime(self, vals , ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        zeniths = np.arange(0, 2, 1) # bare minimum to create the colors
        
        counts, angles = self.get_histogram(vals)

        # adjust azimuths so the grids aren't centered for pcolormesh
        # to understand this, try creating the same grid with fewer grids in the azimuths
        # we also add this correction factor because we are going clockwise
        corr_factor = 360/len(angles)/2
        azimuths = np.radians(angles / 120 + corr_factor)
        r, theta = np.meshgrid(zeniths, azimuths)

        values = np.concatenate([counts[i]*np.ones(zeniths.size-1).reshape(1, -1) for i in np.arange(azimuths.size-1)])
        
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2)
        mesh = ax.pcolormesh(theta, r, values, **kwargs)

        # ax.xaxis.set_major_locator(ticker.FixedLocator(np.radians([i*360/12 for i in range(12)])))
        ax.yaxis.set_visible(False)
        ax.set_thetagrids(np.arange(0, 360, 30),
                          labels=np.concatenate([[12], np.arange(1,12)]),
                          fontsize=12,
                          # lines=[None for i in range(12)]
                         )
        ax.grid(False)
        ax.set_xticklabels(np.concatenate([[12], np.arange(1,12)]))
        ax.tick_params(top=False,
                       bottom=False,
                       left=False,
                       right=False,
                       labelleft=True,
                       labelbottom=True)
        
        return ax, mesh
    
    def plot_sleeptime(self, data=None, outputfile=None):
        if data is None:
            data = self.data
        fig = plt.figure()
        gs = GridSpec(nrows=1, ncols=3, width_ratios=[0.5, 0.5, 0.05], figure=fig,
                      wspace=0.5)
        
        titles = OrderedDict([
            ('bedtime_start_delta_adj', 'bedtime'),
            ('bedtime_end_delta_adj', 'get-up time')
        ])
        
        # get vmax
        vmax = 0 # initialize
        for _, col in enumerate(titles.keys()):
            counts, angles = self.get_histogram(self.data[col])
            max_count = np.max(counts)
            if max_count > vmax:
                vmax = max_count

        for i, col in enumerate(titles.keys()):
            ax = fig.add_subplot(gs[i], projection='polar')
            ax, mesh = self.plot_clock_sleeptime(data[col], ax=ax, vmin=0, vmax=vmax, cmap='magma')
            ax.set_title(titles[col])
        
        # we can use the same colorbar for both meshes because we set vmax to be the same for both
        cax = fig.add_subplot(gs[-1])
        cb = fig.colorbar(mesh, cax=cax, format=ticker.PercentFormatter(xmax=1))
        cb.ax.set_ylabel('% of user-day combinations')
        
        if outputfile is not None:
            utils.savefig_multext(fig, outputfile, exts=['.png', '.pdf'])
            
    def plot_sleeptime_weekly(self, weeks=None, outputfile=None):
        if weeks is None:
            weeks = sorted(self.data.index.get_level_values('weeknum').unique())

        fig = plt.figure(figsize=(8, 4*len(weeks)))
        gs = GridSpec(nrows=len(weeks), ncols=4, width_ratios=[0.05, 0.5, 0.5, 0.05], figure=fig,
                      wspace=0.5)
        
        titles = {'bedtime_start_delta_adj': 'bedtime',
                  'bedtime_end_delta_adj': 'get-up time'
                 }
        # axs = gs.subplots(subplot_kw=dict(projection='polar'))
        
        for i, week in enumerate(weeks):
            ax_lbl = fig.add_subplot(gs[i, 0])
            ax_lbl.text(0.5, 0.5, f'week {week}', transform=ax_lbl.transAxes, rotation=90)
            ax_lbl.axis('off')

            # get vmax
            vmax = 0 # initialize
            for _, col in enumerate(titles.keys()):
                counts, angles = self.get_histogram(self.data[col].loc[:, week])
                max_count = np.max(counts)
                if max_count > vmax:
                    vmax = max_count

            for j, col in enumerate(['bedtime_start_delta_adj', 'bedtime_end_delta_adj']):
                ax = fig.add_subplot(gs[i, j+1], projection='polar')
                ax, mesh = self.plot_clock_sleeptime(self.data[col].loc[:, week], ax=ax, vmax=vmax, cmap=plt.get_cmap('magma'))
                ax.set_title(titles[col])
        
            cax = fig.add_subplot(gs[i, -1])
            cb = fig.colorbar(mesh, cax=cax, format=ticker.PercentFormatter(xmax=1))
            cb.ax.set_ylabel('% of user-day combinations')

        if outputfile is not None:
            utils.savefig_multext(fig, outputfile, exts=['.png', '.pdf'])


class ActivityAggData(object):
    def __init__(self, raw_activity=None, raw_sleep=None, *, query_str=None, mean_fxn=np.nanmean,
                 min_count_per_user_dayofweek=None,
                 resolution_mins=5,
                ):
        self.set_activity_data_by_user(raw_activity, query_str=query_str, resolution_mins=resolution_mins)
        if raw_sleep is not None:
            self.set_sleep_data_by_user(raw_sleep, query_str=query_str)
        else:
            self._sleep_data_by_user = None
        self.set_user_idxs(min_count_per_user_dayofweek=min_count_per_user_dayofweek)    
        self.set_data_gb_userdayofweek(mean_fxn=mean_fxn)
        self.set_data_gb_dayofweek(mean_fxn=mean_fxn)
        self.get_ts_by_dayofweek(resolution_mins=resolution_mins)
        if raw_sleep is not None:
            self.set_sleep_gb_by_userdayofweek(mean_fxn=mean_fxn)

    @property
    def activity_data_by_user(self):
        return self._activity_data_by_user

    @property
    def sleep_data_by_user(self):
        return self._sleep_data_by_user

    @property
    def data_gb_userdayofweek(self):
        return self._data_gb_userdayofweek

    @property
    def user_idxs(self):
        return self._user_idxs

    @property
    def user_day_idxs_dict(self):
        return self._user_day_idxs_dict

    @property
    def data_gb_dayofweek(self):
        return self._data_gb_dayofweek
    
    @property
    def ts_gb_dayofweek(self):
        return self._ts_gb_dayofweek

    @property
    def sleep_gb_by_userdayofweek(self):
        return self._sleep_gb_by_userdayofweek
    
    @classmethod
    def resample_class_arr(cls, srs_of_lists, resolution_mins, mean_fxn=np.nanmean):
        idx_names = srs_of_lists.index.names
        if idx_names == [None]:
            idx_names = ['temp_idx']
            srs_of_lists.index.names = idx_names

        srs = srs_of_lists.map(lambda x: [(i,k) for i, k in enumerate(x)])
        d = srs.rename('class_arr').explode().to_frame()
        d['time_idx'], d['act_lvl'] = d['class_arr'].str
        d['time_idx_agg'] = d['time_idx'] // (resolution_mins / 5)
        s = d.groupby(idx_names + ['time_idx_agg'])['act_lvl'].agg(mean_fxn).to_frame()
        res = s['act_lvl'].groupby(idx_names).apply(lambda srs: srs.tolist())
        res.index.names = srs_of_lists.index.names

        return res        
        
    def set_activity_data_by_user (self, raw_activity, query_str=None, resolution_mins=5):
        data = utils.add_dayofweek(raw_activity)
        if query_str is not None:
            data = data.query(query_str).copy()

        data['class_arr'] = data['class_5min'].map(lambda x: np.asarray([int(i) if i!='0' else np.nan for i in list(x)]))
        
        if resolution_mins != 5:
            data['class_arr'] = self.resample_class_arr(data['class_arr'], resolution_mins)

        self._activity_data_by_user = data

    def set_sleep_data_by_user(self, raw_sleep, query_str=None):
        data = utils.add_dayofweek(raw_sleep)
        if query_str is not None:
            data = data.query(query_str).copy()
        
        try:
            data[['bedtime_start_delta', 'bedtime_end_delta']] = data[
                ['bedtime_start_delta', 'bedtime_end_delta']
            ].astype(int)
            data['duration'] = data['duration'].astype(float)
        except KeyError:
            pass
        self._sleep_data_by_user = data
        
    def set_user_idxs(self, min_count_per_user_dayofweek=None):
        gb_dict = {}
        list_keys = ['activity', 'sleep']
        if self.sleep_data_by_user is None:
            list_keys.pop('sleep')
        for i in list_keys:
            gb_dict[i] = getattr(self, f'{i}_data_by_user').groupby(['record_id', 'dayofweek', 'day_name'])
        if min_count_per_user_dayofweek is not None:
            user_idxs_dict = {}
            user_day_idxs_dict = {}
            for i, gb in gb_dict.items():
                size = gb.size()
                user_day_idxs = size[size >= min_count_per_user_dayofweek].index
                user_day_idxs_dict[i] = user_day_idxs
                user_idxs_dict[i] = user_day_idxs.to_frame(index=False)[['record_id']].drop_duplicates()
            
            if len(list_keys) == 2:
                user_idxs = pd.merge(user_idxs_dict['activity'],
                                     user_idxs_dict['sleep'],
                                     on='record_id',
                                     how='inner')
            else:
                user_idxs = user_idxs_dict['activity']
                
            # Restrict only to common users
            for i, user_day_idxs in user_day_idxs_dict.items():
                df = user_day_idxs.to_frame(index=False).merge(user_idxs, on='record_id', how='inner')
                user_day_idxs_dict[i] = pd.MultiIndex.from_frame(df)
        
            self._user_idxs = user_idxs
            self._user_day_idxs_dict = user_day_idxs_dict
            
    def set_data_gb_userdayofweek(self, mean_fxn=np.nanmean):
        gb = self.activity_data_by_user.groupby(['record_id', 'dayofweek', 'day_name'])
        # do not include non-wear in average by default (np.nanmean)
        data = gb['class_arr'].agg(
            lambda x: mean_fxn(np.stack(x), axis=0))
        if self.user_day_idxs_dict is not None:
            data = data.loc[self.user_day_idxs_dict['activity']]

        self._data_gb_userdayofweek = data
    
    @classmethod
    def compute_gb_dayofweek(cls, data_gb_userdayofweek_srs,
                             gb_cols=['dayofweek', 'day_name'],
                             mean_fxn=np.nanmean):
        data = data_gb_userdayofweek_srs.to_frame().groupby(gb_cols)['class_arr'].agg(
            lambda x: mean_fxn(np.stack(x), axis=0)).rename('central_tendency').to_frame()
        
        return data
        
    def set_data_gb_dayofweek(self, mean_fxn=np.nanmean):
        try:
            srs = self.data_gb_userdayofweek
        except:
            raise NameError('Run set_gb_userdayofweek')

        self._data_gb_dayofweek =  self.compute_gb_dayofweek(srs, mean_fxn=mean_fxn)
        
    @classmethod
    def compute_ts_by_dayofweek(cls, srs_of_lists, resolution_mins=5):
        gb_dayofweek_mean = pd.DataFrame(np.array(srs_of_lists.tolist()), index=srs_of_lists.index).T
        gb_dayofweek_mean.index = [pd.Timestamp('04:00:00') + pd.Timedelta(minutes=resolution_mins*i) for i in range(int(60*24/resolution_mins))]
        gb_dayofweek_mean.columns = gb_dayofweek_mean.columns.droplevel('dayofweek')
        
        return gb_dayofweek_mean
        
    def get_ts_by_dayofweek(self, resolution_mins=5):
        srs_of_lists = self.data_gb_dayofweek['central_tendency']
        gb_dayofweek_mean = self.compute_ts_by_dayofweek(srs_of_lists, resolution_mins=resolution_mins)
        
        self._ts_gb_dayofweek = gb_dayofweek_mean

    def set_sleep_gb_by_userdayofweek(self, mean_fxn=np.nanmean):
        gb = self.sleep_data_by_user.groupby(['record_id', 'dayofweek', 'day_name'])
        data = gb.agg(mean_fxn)
        if self.user_day_idxs_dict is not None:
            data = data.loc[self.user_day_idxs_dict['sleep']]
        
        for col in ['bedtime_start', 'bedtime_end']:
            if f'{col}_delta' in data.columns:
                data[f'{col}_time'] = data[f'{col}_delta'].map(
                    lambda x: pd.Timestamp('00:00:00') + pd.Timedelta(seconds=x)
                )
        self._sleep_gb_by_userdayofweek = data


def plot_activity_by_dayofweek(gb_dayofweek_mean, bedtime_df,
                               ax=None, ax_sleep=None,
                               ylabel=None, add_hlines=True,
                               title=None, title_fontsize=None,
                               plot_kwargs=None,
                               outputfile=None,
                               dayofweek_list=None,
                               plot_activity=True,
                               plot_bedtime_start=True,
                               plot_bedtime_end=True,
                               label_suffix=None,
                              ):
    colors = plt.get_cmap('tab10').colors
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure

    xvals = pd.to_datetime(pd.Series(gb_dayofweek_mean.index))
    
    bedtime_df = bedtime_df.copy()

    # bedtime end plot
    if ax_sleep is None:
        ax_sleep = ax.twinx()
    
    bedtime_df['bedtime_start_time_adj'] = bedtime_df['bedtime_start_time'] + pd.Timedelta(days=1)

    colors = plt.get_cmap('tab10').colors
    if plot_kwargs is None:
        plot_kwargs = {}
    if dayofweek_list is None:
        dayofweek_list = range(7)
    elif isinstance(dayofweek_list, int):
        dayofweek_list = [dayofweek_list]

    for dayofweek in dayofweek_list:
        if plot_bedtime_end:
            sns.kdeplot(data=bedtime_df.query(f'dayofweek == {dayofweek}'),
                        x='bedtime_end_time',
                        # fill=colors[dayofweek], alpha=0.1,
                        ax=ax_sleep,
                        linestyle='dashed',
                        alpha=0.5,
                        color=colors[dayofweek],
                        **plot_kwargs
                       )
        
        if plot_bedtime_start:
            # note that the summary_date for bedtime_start is the day one wakes up, so we adjust the colors
            sns.kdeplot(data=bedtime_df.query(f'dayofweek == {(dayofweek+1) % 7}'),
                        x='bedtime_start_time_adj',
                        # fill=colors[dayofweek], alpha=0.1,
                        ax=ax_sleep,
                        linestyle='dashdot',
                        alpha=0.5,
                        color=colors[dayofweek],
                        **plot_kwargs
                       )
    ax.spines['right'].set_linestyle((0,(10,8)))
    # ax.spines['right'].set_visible(False)

    # activity plot
    dayint_to_dayname = dict(enumerate(list(calendar.day_name)))
    dayint_to_dayabbr = dict(enumerate(list(calendar.day_abbr)))
    if label_suffix is None:
        label_suffix = ''
    for dayofweek in dayofweek_list:
        col = dayint_to_dayname[dayofweek]
        yvals = gb_dayofweek_mean[col]
        label = f'{dayint_to_dayabbr[dayofweek]}{label_suffix}'
        ax.plot(xvals, yvals,
                label=label,
                color=colors[dayofweek], **plot_kwargs)
    # gb_dayofweek_mean.plot(ax=ax, linewidth=1)
    if ylabel is None:
        ylabel = 'Oura activity: mean of per-user means'
    ax.set_ylabel(ylabel)
    ax_sleep.set_ylabel(f'density (mean time per user)')
    if add_hlines:
        ax.axhline(2, linestyle=(0, (1,5)), color='black', linewidth=1)
        ax.axhline(1, linestyle=(0, (1,5)), color='black', linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt='%-I %p'))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    
    days = pd.Series(bedtime_df['bedtime_end_time'].map(
        lambda x: x.date()).unique(), name='_day').sort_index(ascending=False).reset_index(drop=True)
    ax.set_xlim(pd.Timestamp(days.iloc[-1]) + pd.Timedelta(hours=3, minutes=55, seconds=0),
                pd.Timestamp(days.iloc[-1]) + pd.Timedelta(days=1, hours=4, minutes=5, seconds=0))
    ax.grid(axis='x', which='both')
    ax.set_ylim(1,3)
    
    handles, labels = ax.get_legend_handles_labels()
    handles = (handles +
               [
                   Line2D([0], [0], color='black'),
                   Line2D([0], [0], color='black', linestyle='dashed'),
                   Line2D([0], [0], color='black', linestyle='dashdot'),
               ]) 
    labels = labels + ['activity level', 'mean get-up time', 'mean bedtime']
    
    ax.legend(handles=handles, labels=labels, loc='lower right', bbox_to_anchor=(0.75, 0.05), bbox_transform=ax.transAxes,
              fontsize=9,
             )

    ax_sleep.set_zorder(1)  # default zorder is 0 for ax1 and ax2
    # ax.set_zorder(2)  # default zorder is 0 for ax1 and ax2
    ax_sleep.set_frame_on(False)  # prevents ax1 from hiding ax2
    ax_sleep.set_ylim(0,14)
    
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if outputfile is not None:
        utils.savefig_multext(fig, outputfile)

    return ax, ax_sleep


def plot_activity_school_vs_tgv(activity_agg_data, activity_agg_data_tgv, outputfile=None):
    nrows = 2
    ncols = 1
    fig = plt.figure(figsize=(6*ncols, 4*nrows))

    gs_big = GridSpec(nrows=2, ncols=1, height_ratios=[0.8, 0.2], hspace=0.15)
    gs_main = GridSpecFromSubplotSpec(nrows=1, ncols=3, width_ratios=[0.1, 0.7, 0.1], wspace=0.2, subplot_spec=gs_big[0])
    gs = GridSpecFromSubplotSpec(nrows=nrows, ncols=ncols, subplot_spec=gs_main[1], hspace=0.45)

    act_agg_list = [
        ('school in session', activity_agg_data),
        ('Thanksgiving break', activity_agg_data_tgv)
    ]

    for i, (lbl, act_agg) in enumerate(act_agg_list):
        ax = fig.add_subplot(gs[i])
        ax, ax_twin = plot_activity_by_dayofweek(
            act_agg.ts_gb_dayofweek,
            act_agg.sleep_gb_by_userdayofweek,
            ax=ax, ylabel=None, add_hlines=True,
        )
        for ticklbl in ax.get_xmajorticklabels():
            ticklbl.set_rotation(45)
            ticklbl.set_ha('right')
            ticklbl.set_fontsize(9)

        ax.text(0.05, 0.95, f'{ascii_uppercase[i]}', fontsize=10, fontweight='bold',
                transform=ax.transAxes, ha='left', va='top')
        ax.set_title(lbl, fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        
        ylabel1, ylabel2 = ax.get_ylabel(), ax_twin.get_ylabel()
        ax.set_ylabel('')
        ax_twin.set_ylabel('')
        ax.set_ylim(1,3)
        ax_twin.set_ylim(0,10)
        

    handles = (handles +
               [
                   Line2D([0], [0], color='black'),
                   Line2D([0], [0], color='black', linestyle='dashed'),
                   Line2D([0], [0], color='black', linestyle='dashdot'),
               ]) 
    labels = labels + ['activity level', 'mean get-up time', 'mean bedtime']
    
    ax_lgd = fig.add_subplot(gs_big[-1])
    ax_lgd.legend(handles=handles, labels=labels, loc='center',
                  bbox_to_anchor=(0.5,0.5),
                  bbox_transform=ax_lgd.transAxes,
                  fontsize=9,
                  ncols=3
                 )
    ax_lgd.axis('off')
    
    ax_ylbl_left = fig.add_subplot(gs_main[0])
    ax_ylbl_right = fig.add_subplot(gs_main[2])
    
    ax_ylbl_left.text(0.5, 0.5, ylabel1, transform=ax_ylbl_left.transAxes, rotation=90, ha='center', va='center', fontsize=10)
    ax_ylbl_right.text(0.5, 0.5, ylabel2, transform=ax_ylbl_right.transAxes, rotation=90, ha='center', va='center', fontsize=10)
    
    ax_ylbl_left.axis('off')
    ax_ylbl_right.axis('off')
    
    if outputfile is not None:
        print(outputfile)
        utils.savefig_multext(fig, outputfile)

    count_df = utils.get_pct_srs(pd.Series(
        [
            activity_agg_data.data_gb_userdayofweek.index.get_level_values('record_id').nunique(),
             activity_agg_data_tgv.data_gb_userdayofweek.index.get_level_values('record_id').nunique(),
        ],index=['school', 'tgv']
    ))
    count_df.loc['total', :] = count_df.sum()
    print(count_df)

    count2_df = utils.get_pct_srs(pd.Series(
        [
            activity_agg_data.sleep_gb_by_userdayofweek.index.get_level_values('record_id').nunique(),
             activity_agg_data_tgv.sleep_gb_by_userdayofweek.index.get_level_values('record_id').nunique(),
        ],index=['school', 'tgv']
    ))
    print(count2_df)


class SubsetDifferences():
    def __init__(self, activity_agg_data_subset, subset_key_list, dayofweek_list=None, plot=True,
                 stat_test='ttest_ind', fdrcorrect=True, **kwargs):
        self.agg_subset = activity_agg_data_subset
        self.subset_key_list = subset_key_list
        self.set_dflist(activity_agg_data_subset, subset_key_list)
        self.stat_test = stat_test
        self.stats = self.test_differences_in_subset(activity_agg_data_subset, subset_key_list,
                                                     stat_test=stat_test,
                                                     fdrcorrect=fdrcorrect)
        self.time_index = activity_agg_data_subset[subset_key_list[0]].ts_gb_dayofweek.index
        if plot and dayofweek_list is not None:
            self.plot(dayofweek_list, **kwargs)

    @property
    def spm_xfill_between_dict(self):
        return self._spm_xfill_between_dict
    
    def _fdr_corrected_pvals(self, pval_srs):
        idx = pval_srs.dropna().index
        corr_pvals = fdrcorrection(pval_srs.dropna())
        # print(len(corr_pvals[0]))
        new_pvals = pd.Series(corr_pvals[1], index=idx).reindex(pval_srs.index)

        new_pvals = new_pvals.fillna(pval_srs).rename('pval_corr')

        return new_pvals

    def set_dflist(self, activity_agg_data_subset, subset_key_list, as_prop=True):
        """
        Creates a list of dataframes of data, each corresponding to a different subset.
        """
        srs_list = [activity_agg_data_subset[subset_key].data_gb_userdayofweek for subset_key in subset_key_list]
        dflist = [pd.DataFrame(srs.tolist(), index=srs.index) for srs in srs_list]
        
        if as_prop:
            self.dflist = dflist
        else:
            return dflist
        
    def test_differences_in_subset(self, activity_agg_data_subset, subset_key_list, stat_test, fdrcorrect):
        # assert len(subset_key_list) == 2
        dflist = self.dflist

        if stat_test.startswith("mann"):
            stat_fxn = mannwhitneyu
        elif stat_test in ["t-test", "ttest_ind"]:
            stat_fxn = ttest_ind
        elif 'anova' in stat_test.lower():
            stat_fxn = f_oneway
        res_dict = {}
        for dayofweek in range(7):
            for col in dflist[0].columns:
                mwu_srs_list = [df.loc[pd.IndexSlice[:, dayofweek,:], col].dropna() for df in dflist]
                res_dict[(dayofweek, col)] = stat_fxn(*mwu_srs_list)

        res = pd.DataFrame.from_dict(res_dict).T
        res.columns = ['stat', 'pval']
        res.index.names = ['dayofweek', 'timepoint']

        if fdrcorrect:
            res['pval'] = res.groupby(['dayofweek'])['pval'].transform(self._fdr_corrected_pvals)
            
        return res

    def plot(self, dayofweek_list, dayofweek_list_plot=None,
             thresh=0.05, fresh_plot=False, ax=None,
             plot_what='ttest_multiple'):
        if dayofweek_list is None:
            dayofweek_list = range(7)
        
        colors = plt.get_cmap('tab10').colors

        if fresh_plot and dayofweek_list_plot is None:
            if ax is None:
                _, ax = plt.subplots()
                ax.set_ylim(0,1)
        else:
            if dayofweek_list_plot is None:
                dayofweek_list_plot = dayofweek_list
            ax, ax_sleep = plot_compare_activity_by_dayofweek_subset(self.agg_subset,
                                                      self.subset_key_list,
                                                      label_list=None,
                                                      outputfile=None,
                                                      dayofweek_list=dayofweek_list_plot,
                                                      ax=ax,
                                                    )
            # ax.get_legend().remove()
        fig = ax.figure

        ylim_list = np.linspace(0, 1, len(dayofweek_list) + 1)
        for i, dayofweek in enumerate(dayofweek_list[::-1]):
            print(plot_what)
            if plot_what.startswith('ttest'):
                ax.fill_between(self.time_index, ylim_list[i], ylim_list[i+1],
                                where=self.stats.loc[dayofweek]['pval'] < thresh,
                                transform=ax.get_xaxis_transform(),
                                alpha=0.3,
                                color=colors[dayofweek]
                               )
            elif plot_what.startswith('spm'):
                xfill_between_dict = self.spm_xfill_between_dict
                for xfill_between in xfill_between_dict[dayofweek]:
                    ax.fill_between(xfill_between,
                                    ylim_list[i], ylim_list[i+1],
                                    step='post',
                                    transform=ax.get_xaxis_transform(),
                                    alpha=0.3,
                                    color=colors[dayofweek],
                                   )
        
        if fresh_plot:
            midpoints = (ylim_list[1:] + ylim_list[:-1]) / 2
            ax.yaxis.set_major_locator(ticker.FixedLocator(midpoints))
            ax.yaxis.set_ticklabels(calendar.day_abbr[::-1], va='center')
            ax.set_ylim(0,1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt='%-I %p'))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
    
        # ax.set_xlim(pd.Timestamp(days.iloc[-1]) + pd.Timedelta(hours=3, minutes=55, seconds=0),
        #             pd.Timestamp(days.iloc[-1]) + pd.Timedelta(days=1, hours=4, minutes=5, seconds=0))
        
        if fresh_plot:
            return ax
        else:
            return ax, ax_sleep
    
    def get_spm_differences_in_subset(self, a=None, b=None, alpha=0.05,
                                      interp=False,
                                      stattest_kwargs=None, inference_kwargs=None,
                                      as_prop=True):
        dflist = self.dflist
        if stattest_kwargs is None:
            stattest_kwargs = {}
        if inference_kwargs is None:
            inference_kwargs = {}
        
        if self.stat_test.startswith('ttest'):
            stattest = 'ttest2'
        elif self.stat_test.startswith('anova'):
            stattest = 'anova1'

        xfill_dayofweek_dict = {}
        for dayofweek in range(7):
            Ylist = [df.loc[pd.IndexSlice[:, dayofweek, :]].iloc[:,a:b].dropna(axis=0) for df in dflist]
            for Y in Ylist:
                assert Y.isna().sum().sum() == 0
            if stattest == 'ttest2':
                t = getattr(spm1d.stats, stattest)(*Ylist, **stattest_kwargs)
            elif stattest == 'anova1':
                t = getattr(spm1d.stats, stattest)([Y.to_numpy() for Y in Ylist], **stattest_kwargs)
            ti = t.inference(alpha=alpha, interp=interp, **inference_kwargs)

            # we adjust for taking only a:b of the Y
            adj = a
            if adj is None:
                adj = 0
            xfill_between = [] # list of lists
            for cluster in ti.clusters:
                fill_between = [cluster.endpoints[0] + adj, cluster.endpoints[1] + 1 + adj]
                xfill_between.append(self.time_index[fill_between])
            
            xfill_dayofweek_dict[dayofweek] = xfill_between

        if as_prop:
            self._spm_xfill_between_dict = xfill_dayofweek_dict
        else:
            return xfill_dayofweek_dict

def plot_compare_activity_by_dayofweek_subset(activity_agg_data_subset, subset_keys_list,
                                              label_list=None,
                                              ax=None,
                                              outputfile=None,
                                              **kwargs
                                             ):
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure
    

    if label_list is None:
        label_list = [f"_{i.replace('dysfunc', 'impairment').replace('no_', 'no ')}" for i in subset_keys_list]
        
    lw_list = [i+1 for i in range(len(subset_keys_list))][::-1]
        
    for i, (lbl, subset_key) in enumerate(zip(label_list, subset_keys_list)):
        if i < len(subset_keys_list) - 1:
            ax, ax_twin = plot_activity_by_dayofweek(activity_agg_data_subset[subset_key].ts_gb_dayofweek,
                                       activity_agg_data_subset[subset_key].sleep_gb_by_userdayofweek,
                                       ax=ax, ylabel='', add_hlines=True,
                                       outputfile=None,
                                       plot_kwargs=dict(linewidth=lw_list[i]),
                                       label_suffix=lbl,
                                       **kwargs
                                      )
            ax_twin.yaxis.set_visible(False)
        else:
            ax, ax_twin = plot_activity_by_dayofweek(activity_agg_data_subset[subset_key].ts_gb_dayofweek,
                                       activity_agg_data_subset[subset_key].sleep_gb_by_userdayofweek,
                                       ax=ax, ylabel=None, add_hlines=True,
                                       outputfile=outputfile,
                                       plot_kwargs=dict(linewidth=lw_list[i]),
                                       label_suffix=lbl,
                                       **kwargs
                                      )
    
    count_df = utils.get_pct_srs(pd.Series(
        [activity_agg_data_subset[i].data_gb_userdayofweek.index.get_level_values('record_id').nunique()
         for i in subset_keys_list
        ],index=subset_keys_list
    ))
    count_df.loc['total', :] = count_df.sum()
    print(count_df)
    
    return ax, ax_twin


def combine_subset_difference_plots(activity_agg_data_subset, subset_key_list, dayofweek_list=None,
                                    sd_kwargs=dict(stat_test='ttest_ind', fdrcorrected=True),
                                    outputfile=None,
                                    plot_what='ttest',
                                    a=None, b=None
                                   ):
    # take the "plot" kwarg out:
    sd_kwargs.get('plot', '')

    fig = plt.figure(figsize=(8,8))
    # gs_big = GridSpec(nrows=2, ncols=1, height_ratios=[0.2, 0.8])
    # ax_summ = fig.add_subplot(gs_big[0])
    
    # gs_main = GridSpecFromSubplotSpec(nrows=1, ncols=3,
    #                                   width_ratios=[0.005, 0.99, 0.005],
    #                                   subplot_spec=gs_big[1])
    gs_main = GridSpec(nrows=1, ncols=3,
                       width_ratios=[0.005, 0.99, 0.005]
                      )
    
    nrows = 4
    ncols = 2

    gs = GridSpecFromSubplotSpec(nrows=nrows, ncols=ncols,
                                 subplot_spec=gs_main[1],
                                 wspace=0.1
                                )

    if dayofweek_list is None:
        dayofweek_list = range(7)
    
    sd = SubsetDifferences(
        activity_agg_data_subset, subset_key_list,
        plot=False,
        **sd_kwargs
    )
    if plot_what == 'spm':
        sd.get_spm_differences_in_subset(a=a, b=b)
    
    for i, dayofweek in enumerate(dayofweek_list):
        idx = np.unravel_index(i, (nrows,ncols))
        ax = fig.add_subplot(gs[idx])
            
        ax, ax_sleep = sd.plot(dayofweek_list=[dayofweek],
                fresh_plot=False,
                ax=ax,
                plot_what=plot_what
               )
        
        ax.text(0.05, 0.95, calendar.day_abbr[i], ha='left', va='top',
                transform=ax.transAxes, fontsize=8)
        for ticklabel in ax.get_ymajorticklabels():
            ticklabel.set_fontsize(8)
        for ticklabel in ax_sleep.get_ymajorticklabels():
            ticklabel.set_fontsize(8)
            
        # overwrites each time but it's okay, the labels are the same anyway
        ylbl1 = ax.get_ylabel()
        ylbl2 = ax_sleep.get_ylabel()
        
        ax.set_ylabel('')
        ax_sleep.set_ylabel('')
        if idx[1] == 0:
            ax_sleep.set_yticklabels('')
        elif idx[1] == 1:
            ax.set_yticklabels('')
        

        h, l = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        
        if i < len(dayofweek_list) - 2:
            ax.set_xticklabels('')
        else:
            for ticklabel in ax.get_xmajorticklabels():
                ticklabel.set_rotation(90)
                ticklabel.set_fontsize(8)
        xlim = ax.get_xlim()
    
    # ax_summ.set_xlim(xlim)
    
    lgd_lbls = [x.split('_', maxsplit=1)[-1] for x in l]
    lgd_handles = [Line2D([0], [0], color='black', linewidth=handle.get_linewidth())
                   for handle in h]
    lgd_handles = (lgd_handles +
               [
                   Line2D([0], [0], color='black', linestyle='dashed'),
                   Line2D([0], [0], color='black', linestyle='dashdot'),
               ]) 
    ax_lgd = fig.add_subplot(gs[-1, -1])
    ax_lgd.legend(handles=lgd_handles, labels=lgd_lbls + ['mean get-up time', 'mean bedtime'], 
                  loc='lower center',
                  fontsize=9,
                 )
    ax_lgd.axis('off')
    
    ax_ylbl1 = fig.add_subplot(gs_main[0])
    ax_ylbl1.text(0, 0.5, ylbl1, rotation=90, ha='right', va='center', transform=ax_ylbl1.transAxes)
    ax_ylbl2 = fig.add_subplot(gs_main[-1])
    ax_ylbl2.text(1, 0.5, ylbl2, rotation=90, ha='left', va='center', transform=ax_ylbl2.transAxes)
    
    ax_ylbl1.axis('off')
    ax_ylbl2.axis('off')
    
    if outputfile is not None:
        if plot_what == 'spm':
            outputfile = outputfile.parent / Path(f'{outputfile.stem}_spm{outputfile.suffix}')
        utils.savefig_multext(fig, outputfile)

def generate_activityaggdata_subset(activity_data, sleep_data, demog_subset_query_dict=None, aggdata_kwargs=None):
    if aggdata_kwargs is None:
        aggdata_kwargs = {}
    
    if demog_subset_query_dict is None:
        demog_subset_query_dict = OrderedDict([
            ('all', None),
            ('m', 'gender_fmnb == "m"'),
            ('f', 'gender_fmnb == "f"'),
            ('nb', 'gender_fmnb != "m" & gender_fmnb != "f"'),
            ('dysfunc', 'dysfunc_depr_anx == 1'),
            ('no_dysfunc', 'dysfunc_depr_anx == 0'),
            ('dysfunc+f', 'dysfunc_depr_anx == 1 & gender_fmnb == "f"'),
            ('dysfunc+m', 'dysfunc_depr_anx == 1 & gender_fmnb == "m"'),
            ('no_dysfunc+f', 'dysfunc_depr_anx == 0 & gender_fmnb == "f"'),
            ('no_dysfunc+m', 'dysfunc_depr_anx == 0 & gender_fmnb == "m"'),
        ])
    activity_agg_data_subset = {}
    for k, q in demog_subset_query_dict.items():
        if q is not None:
            act = activity_data.query(q)
            sleep = sleep_data.query(q)
        else:
            act = activity_data
            sleep = sleep_data
        activity_agg_data_subset[k] = ActivityAggData(
            act,
            sleep,
            **aggdata_kwargs
        )
    
    return activity_agg_data_subset
def make_sleep_reg_data(sleep_data_by_user):
    cat_cols = ['record_id', 'summary_date', 'dayofweek', 'day_name', 'weeknum', 'gender_fmnb', 'dysfunc_depr_anx',
           ]
    reg_data = sleep_data_by_user[cat_cols + [
        'bedtime_start_delta', 'bedtime_end_delta',
        'duration',
        'efficiency',
        'is_weekday',
        'midpoint_comp_from_midnight',
    ]].copy()
    for k in ['start', 'end']:
        reg_data[f'bedtime_{k}_delta_hrs'] = reg_data[f'bedtime_{k}_delta'] / 3600
    for k in ['duration',
              'midpoint_comp_from_midnight']:
        reg_data[f'{k}_hrs'] = (reg_data[k] / 3600).astype(float)
    reg_data['dayofweek_int'] = reg_data['dayofweek']

    for i in ['m', 'f', 'nb']:
        reg_data[f'gender_{i}0'] = reg_data['gender_fmnb'].map(lambda x: 0 if x == i else 1)

    cat_cols = ['record_id',
                'summary_date',
                'weeknum',
                'dayofweek',
                # 'gender_fmnb',
                'dysfunc_depr_anx',
               # ] + [f'gender_{i}0' for i in ['m', 'f', 'nb'],
                'is_weekday'
               ]


    for col in cat_cols:
        reg_data[col] = reg_data[col].astype(str)

    return reg_data


def compute_chronotype_midpointtime(sleep_data, rename_col_to='midpoint_comp_from_midnight_rest', as_hours=False,
                                    demog_df=None):
    cols = [
                'midpoint_comp_from_midnight',
                'bedtime_start_delta',
                'bedtime_end_delta',
                'duration',
            ]
    _d = sleep_data.set_index(['record_id', 'summary_date', 'dayofweek', 'day_name', 'weeknum'])[cols]
    print(_d.index.get_level_values('record_id').nunique())
    _d = _d.groupby('record_id')[cols].agg(np.nanmean)
    print(_d.index.nunique())
    if demog_df is not None:
        _d = _d.merge(survey.demog, left_index=True, right_index=True)
    if as_hours:
        for col in cols:
            _d[col] = (_d[col] / 3600).astype(float)
    
    if rename_col_to is not None:
        _d.rename(columns={'midpoint_comp_from_midnight': rename_col_to}, inplace=True)
    else:
        rename_col_to = midpoint_comp_from_midnight

    cols.remove('midpoint_comp_from_midnight')
    cols_new = [f'{col}_rest' for col in cols]
    _d.rename(columns=dict(zip(cols, cols_new)), inplace=True)

    return _d.loc[:, [rename_col_to] + cols_new]


def plot_chronotype_dependencies(regdata, outputfile=None):
    fig = plt.figure(figsize=(6,6))
    gs = GridSpec(nrows=2, ncols=3, hspace=0.4, height_ratios=[0.6, 0.4])

    def _time_fmt(t_after_midnight, pos):
        lbl = pd.Timestamp('00:00') + pd.Timedelta(hours=t_after_midnight)
        res = lbl.strftime('%-I%p')

        return res
    
    lbl_dict = {
        'bedtime_start_delta_rest': 'sleep period start',
        'bedtime_end_delta_rest': 'sleep period end',
        'duration_rest': 'sleep period duration (hrs)',
        'midpoint_comp_from_midnight_rest': 'sleep period midpoint during Thanksgiving'
               }
    chronotype_col = 'midpoint_comp_from_midnight_rest'
    
    ax = fig.add_subplot(gs[0,:])
    sns.histplot(regdata[chronotype_col], bins=np.arange(2, 12, 0.25), ax=ax, color='cornflowerblue', stat='proportion')
    ax.axvline(regdata[chronotype_col].median(), color='black', linestyle='dashed', linewidth=3)
    ax.set_ylabel('% of participants')
    ax.set_xlabel(lbl_dict[chronotype_col])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_time_fmt))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.text(-0.14, 1, 'A', transform=ax.transAxes, va='top', fontweight='bold', fontsize=10)
    
    for i, col in enumerate(['bedtime_start_delta_rest', 'bedtime_end_delta_rest', 'duration_rest']):
        ax = fig.add_subplot(gs[1, i])
        sns.scatterplot(data=regdata, x=col, y=chronotype_col, ax=ax, s=5)
        ax.set_ylim(2, 12.5)
        if i != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('sleep midpoint')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(_time_fmt))
            ax.text(-0.5, 1.2, 'B', transform=ax.transAxes, va='top', fontweight='bold', fontsize=10)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        if not col.startswith('duration'):
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_time_fmt))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        
        ax.set_xlabel(lbl_dict[col], fontsize=9)
    
    fig.align_xlabels()
    fig.align_ylabels()
    
    if outputfile is not None:
        utils.savefig_multext(fig, outputfile)


class JetlagInfo(object):
    def __init__(self, sleep_data, chronotype_data, groupby_cols=None, agg_cols=None, min_count_per_user_dayofweek=None):
        self._raw_data = self.get_jetlag_info_raw(sleep_data, chronotype_data)
        self._agg_data = self.get_jetlag_info_agg(groupby_cols=groupby_cols, agg_cols=agg_cols,
                                                  min_count_per_user_dayofweek=min_count_per_user_dayofweek)
    @property
    def raw_data(self):
        return self._raw_data

    @property
    def agg_data(self):
        return self._agg_data

    def get_jetlag_info_raw(self, sleep_data, chronotype_data):
        jetlag_data = chronotype_data.merge(
            sleep_data.set_index('record_id'),
            left_index=True,
            right_index=True,
            how='inner'
        ).reset_index()
        jetlag_data['midpoint_comp_from_midnight'] = jetlag_data['midpoint_comp_from_midnight'].astype(float)
        jetlag_data['jetlag'] = jetlag_data['midpoint_comp_from_midnight_rest'] - jetlag_data['midpoint_comp_from_midnight']
        jetlag_data['jetlag'] = jetlag_data['jetlag'].astype(float)
        jetlag_data['jetlag_hrs'] = jetlag_data['jetlag'] / 3600
        
        for col in ['bedtime_start_delta', 'bedtime_end_delta', 'duration']:
            jetlag_data[f'{col}_jetlag'] = jetlag_data[f'{col}_rest'] - jetlag_data[col]
            jetlag_data[f'{col}_jetlag_hrs'] = (jetlag_data[f'{col}_jetlag'] / 3600).astype(float)

        for col in ['bedtime_start_delta', 'bedtime_end_delta', 'duration', 'midpoint_comp_from_midnight']:
            jetlag_data[f'{col}_hrs'] = (jetlag_data[col] / 3600).astype(float)
        jetlag_data['dayofweek'] = jetlag_data['dayofweek'].astype(str)
        jetlag_data['is_school'] = jetlag_data['weeknum'].map(lambda x: 0 if x == 5 else 1)

        return jetlag_data
    
    def get_jetlag_info_agg(self, groupby_cols=None, agg_cols=None, min_count_per_user_dayofweek=None):
        if groupby_cols is None:
            groupby_cols = ['record_id', 'gender_fmnb', 'dysfunc_depr_anx', 'dayofweek', 'day_name']

        if agg_cols is None:
            agg_cols = (['jetlag', 'midpoint_comp_from_midnight_rest', 'midpoint_comp_from_midnight',
                        'duration',
                        'bedtime_start_delta',
                        'bedtime_end_delta',
                        'jetlag_hrs',                        
                       ] +
                        [f'{col}_jetlag' for col in ['duration', 'bedtime_start_delta', 'bedtime_end_delta']] +
                        [f'{col}_jetlag_hrs' for col in ['duration', 'bedtime_start_delta', 'bedtime_end_delta']]
                         )
            
        gb = self.raw_data.groupby(groupby_cols)
        data = gb.agg({col: np.nanmean for col in agg_cols}).dropna()

        if min_count_per_user_dayofweek is not None:
            # print(min_count_per_user_dayofweek)
            size = gb.size()
            user_day_idxs = size[size >= min_count_per_user_dayofweek].index
            print(len(user_day_idxs))
        else:
            user_day_idxs = None
            print(user_day_idxs)
        if user_day_idxs is not None:
            data = data.loc[user_day_idxs]
        
        return data
    

def plot_jetlag(jetlag_info_obj, col='jetlag', ax=None, outputfile=None,
                as_hrs=True,
                major_locator=1, minor_locator=0.25):
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure
    
    data = jetlag_info_obj.agg_data.copy()
    if as_hrs:
        data[col] = data[col] / 3600
    # print(data[col])
    g = sns.kdeplot(data=data, x=col, hue='day_name', ax=ax)
    
    xlabel_dict = {
        'jetlag': r'social jetlag (hrs)',
        'bedtime_start_delta': 'bedtime start (hrs from 12mn)',
        'bedtime_end_delta': 'bedtime end (hrs from 12mn)',
        'duration': 'bedtime duration (hrs)',
        'duration_jetlag': r'free $-$ school (duration) (hrs)',
        'bedtime_start_delta_jetlag': r'free $-$ school (bedtime start) (hrs)',
        'bedtime_end_delta_jetlag': r'free $-$ school (bedtime end)(hrs)',
    }
    xlabel = xlabel_dict.get(col, col)
    ax.set_xlabel(xlabel)

    if 'jetlag' in col:
        ax.axvline(0, ls='dotted', color='black')

    if major_locator is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_locator))
    if minor_locator is not None:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_locator))
    g.legend_.set_title(None)
    
    if outputfile is not None:
        utils.savefig_multext(outputfile)
    else:
        return g


def plot_jetlag_combined(jetlag_info_obj, cols=['jetlag', 'bedtime_start_delta', 'bedtime_end_delta', 'duration'],
                        outputfile=None,
                        set_uniform_ylims=True,
                        ):
    ncols = 2
    nrows = int(np.ceil(len(cols) / 2))
    remainder = len(cols) % 2
    
    fig = plt.figure(figsize=(3*ncols, 3*nrows))
    
    if remainder == 0:
        gs_main = GridSpec(nrows=2, ncols=2, width_ratios=[0.01, 0.99], height_ratios=[0.87, 0.13])
    else:
        gs_main = GridSpec(nrows=1, ncols=2, width_ratios=[0.01, 0.99])
        
    if set_uniform_ylims:
        wspace = 0.15
    else:
        wspace = 0.25
    gs = GridSpecFromSubplotSpec(nrows=nrows, ncols=ncols, subplot_spec=gs_main[0,1], wspace=wspace, hspace=0.4)
        


    ax_ylabel = fig.add_subplot(gs_main[0,0])
    ax_ylabel.text(1, 0.5, 'density', transform=ax_ylabel.transAxes, ha='right', va='center', rotation=90, fontsize=9)
    ax_ylabel.axis('off')

    for i, col in enumerate(cols):
        idx = np.unravel_index(i, (nrows, ncols))
        ax = fig.add_subplot(gs[idx])
        ax.text(0.05, 0.95, ascii_uppercase[i], transform=ax.transAxes, fontweight='bold', fontsize=10, ha='left', va='top')
        if '_delta_jetlag' not in col:
            g = plot_jetlag(jetlag_info_obj, col=col, ax=ax, major_locator=1, minor_locator=0.5)
        else:
            g = plot_jetlag(jetlag_info_obj, col=col, ax=ax, major_locator=2, minor_locator=1)
        
        if col == 'bedtime_start_delta_jetlag':
            ax.set_xlim(-6, 10)
        h, l = g.legend_.legend_handles, [i.get_text() for i in g.legend_.texts]
        # print(h,l)
        ax.get_legend().remove()
        ax.set_ylabel('')
        for ticklbl in ax.get_yticklabels():
            ticklbl.set_fontsize(7)
        for ticklbl in ax.get_xticklabels():
            ticklbl.set_fontsize(7)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        
        if set_uniform_ylims:
            if i % 2 != 0:
                ax.set_yticklabels('')
            ax.set_ylim(0,0.075)

    
    if remainder == 0:
        ax_lgd = fig.add_subplot(gs_main[1,:])
        ax_lgd.legend(handles=h, labels=l, ncols=4, fontsize=8, loc='lower center')
    else:
        ax_lgd = fig.add_subplot(gs[-1, -1])
        ax_lgd.legend(handles=h, labels=l, ncols=2, fontsize=8, loc='center')
    ax_lgd.axis('off')
    
    if outputfile is not None:
        utils.savefig_multext(fig, outputfile)
    

def make_regdata_jetlag_activity(jetlag_raw_data, activity_data):
    d = jetlag_raw_data.merge(activity_data[['record_id', 'summary_date',
                                            #  'steps',
                                             'cal_active',
                                            #  'cal_total'
                                             ]],
                               on=['record_id', 'summary_date'],
                               how='inner'
                              )
    d[[f'{i}_hrs' for i in ['jetlag',
                            'duration']]] = (d[[i for i in ['jetlag',
                                                            'duration']]] / 3600).astype(float)
    d['cal_active'] = d['cal_active'].astype(float)
    d['is_weekday'] = d['is_weekday'].astype(int)
    d['cal_active_adjforlog'] = d['cal_active'].map(lambda x: x if x!= 0 else 1e-1)
    d['cal_active_log'] = d['cal_active_adjforlog'].map(np.log10)

    return d
    

def plot_calactive_adjforlog(data, ax=None, outputfile=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4,3))
    fig = ax.figure
    sns.kdeplot(regdata_jetlag_activity, x='cal_active_adjforlog', log_scale=True, ax=ax)
    ax.set_xlabel('active calories (kcal)')
    
    if outputfile is not None:
        utils.savefig_multext(fig, outputfile)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    datadir = Path(args.datadir)
    outputdir = Path(args.outputdir)
    
    print(f'Reading data from: {datadir}')
    print(f'Output directory:{outputdir}')

    # Read data files
    assert datadir.exists()
    outputdir.mkdir(parents=True, exist_ok=True)

    activity_data = pd.read_csv(datadir / 'activity_data.csv')
    sleep_data = pd.read_csv(datadir / 'sleep_data.csv')

    activity_data['summary_date'] = pd.to_datetime(activity_data['summary_date'])
    sleep_data['summary_date'] = pd.to_datetime(sleep_data['summary_date'])

    # Daily compliance
    plot_daily_compliance([
            activity_data,
        ],
        [
            sleep_data,
        ],
        outputfile = outputdir / 'compliance_studyperiod.png'
    )

    # Ring wear across a day
    plot_user_pctwear_in_days_with_data(activity_data.query('weeknum != 5'),
                                        outputfile=outputdir / 'pctwear_excltgv_denom_max.png',
                                        userpctwear_kwargs=dict(mode='max', maxdays=5*7),
                                        userpctwear_weekend_kwargs=dict(mode='max', maxdays=2*7),
                                        add_min_days_to_lbl=False
                                    )


    # Bedtime start and end times
    sleeptime = Sleeptime(sleep_data)
    sleeptime.plot_sleeptime(sleeptime.data.query('weeknum != 5'), outputfile=outputdir / 'sleep_clockmap_n=582.png')

    # School vs Thanksgiving activity

    activity_agg_data = ActivityAggData(activity_data,
                                        sleep_data,
                                        query_str='weeknum != 5',
                                        min_count_per_user_dayofweek=3,
                                    )
    activity_agg_data_tgv = ActivityAggData(activity_data,
                                        sleep_data,
                                        query_str='weeknum == 5',
                                        min_count_per_user_dayofweek=1,
                                    )

    assert activity_agg_data.user_idxs['record_id'].nunique() == 566

    plot_activity_school_vs_tgv(activity_agg_data, activity_agg_data_tgv,
                                outputfile=outputdir / 'activity_bedtime_by_dayofweek_school_vs_tgv.png'
                            )

    # Compare activity using SPM
    activity_agg_data_subset = generate_activityaggdata_subset(activity_data,
                                                            sleep_data,
                                                            demog_subset_query_dict=None,
                                                            aggdata_kwargs=dict(
                                                                    query_str='weeknum != 5',
                                                                    min_count_per_user_dayofweek=3,
                                                                )
                                                            )
    sd = SubsetDifferences(activity_agg_data_subset, ['m','f', 'nb'],
                    dayofweek_list=range(7),
                    fresh_plot=False,
                    stat_test='anova',
                    plot=False
                    )

    combine_subset_difference_plots(activity_agg_data_subset, ['m', 'f'],
                                outputfile=outputdir / 'compare_activityts_daily_m_vs_f.png',
                                    plot_what='spm',
                                a=3, b=-5)



    # Sleep midpoint
    chronotype_df = compute_chronotype_midpointtime(sleep_data.query('weeknum == 5'), as_hours=False)
    chronotype_df_hrs = compute_chronotype_midpointtime(sleep_data.query('weeknum == 5'), as_hours=True)

    demog = sleep_data[['record_id', 'gender_fmnb', 'dysfunc_depr_anx']].drop_duplicates().set_index('record_id')
    chronotype_reg_data = chronotype_df_hrs.merge(demog, left_index=True, right_index=True)
    plot_chronotype_dependencies(chronotype_reg_data,
                                outputfile=outputdir / 'chronotype_vs_bedtime.png'
                                )

    run_glm(chronotype_reg_data,
            'midpoint_comp_from_midnight_rest ~ gender_fmnb + dysfunc_depr_anx',
            family='gaussian',
            control='optimizer="bobyqa",optCtrl=list(maxfun=2e5)',
        )

    run_glm(chronotype_reg_data,
            'midpoint_comp_from_midnight_rest ~ bedtime_start_delta_rest',
            family='gaussian',
            control='optimizer="bobyqa",optCtrl=list(maxfun=2e5)',
        )

    run_glm(chronotype_reg_data,
            'midpoint_comp_from_midnight_rest ~ bedtime_end_delta_rest',
            family='gaussian',
            control='optimizer="bobyqa",optCtrl=list(maxfun=2e5)',
        )

    run_glm(chronotype_reg_data,
            'midpoint_comp_from_midnight_rest ~ duration_rest',
            family='gaussian',
            control='optimizer="bobyqa",optCtrl=list(maxfun=2e5)',
        )

    # Jetlag
    jetlag_info = JetlagInfo(sleep_data.query('weeknum != 5'), chronotype_df, min_count_per_user_dayofweek=1)
    jetlag_info_tgv = JetlagInfo(sleep_data.query('weeknum == 5'), chronotype_df, min_count_per_user_dayofweek=1)
    jetlag_info_all = JetlagInfo(sleep_data, chronotype_df)


    plot_jetlag_combined(jetlag_info, cols=['jetlag', 'bedtime_start_delta', 'bedtime_end_delta', 'duration'],
                        outputfile=outputdir / 'jetlag_kdeplots.png')

    # sample regression: jetlag
    run_lmm(jetlag_info.raw_data,
        'jetlag_hrs ~ '
        '+ gender_fmnb'
        '+ dysfunc_depr_anx'
        '+ is_weekday'
        # '+ dayofweek'
        '+ (1|record_id)'
        '+ (1|weeknum)'
            ,
        family='gaussian', verbose=True,
        control='optimizer="bobyqa",optCtrl=list(maxfun=2e5)',
        # factors={"dayofweek": [str(i) for i in [5,6,0,1,2,3,4]]}
        )


    run_lmm(jetlag_info.raw_data,
        'jetlag_hrs ~ '
        '+ gender_fmnb'
        '+ dysfunc_depr_anx'
        #    '+ is_weekday'
        '+ dayofweek'
        '+ (1|record_id)'
        '+ (1|weeknum)'
            ,
        family='gaussian', verbose=True,
        control='optimizer="bobyqa",optCtrl=list(maxfun=2e5)',
        # factors={"dayofweek": [str(i) for i in [5,6,0,1,2,3,4]]}
        )

    # regression active calories
    regdata_jetlag_activity_all = make_regdata_jetlag_activity(jetlag_info_all.raw_data, activity_data)
    print(regdata_jetlag_activity_all['record_id'].nunique())
    run_lmm(regdata_jetlag_activity_all,
        'cal_active_log ~ '
        'jetlag_hrs'
        '+ duration_hrs'
        '+ is_weekday'
        '+ dysfunc_depr_anx'
        '+ gender_fmnb'
            '+ is_school'
        '+ (1|record_id) + (1|weeknum)'
            ,
        family='gaussian', verbose=True,
        control='optimizer="bobyqa",optCtrl=list(maxfun=2e5)',
        # factors={"dayofweek": [str(i) for i in [5,6,0,1,2,3,4]]}
        )
    
if __name__ == "__main__":
    main()