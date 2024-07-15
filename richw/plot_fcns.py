#! /usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import data_fcns
import warnings

def plot_rtc(df_rtc_ts_wind,figfile,df_site):
  POL_RATIO_PLOT = False
  BURST_ID = df_rtc_ts_wind['burst_id'].iloc[0]
  SITE_ID = df_rtc_ts_wind['site_id'].iloc[0]
  fig, ax1 = plt.subplots(figsize=(10, 5))
  ax1.plot(df_rtc_ts_wind['datetime'], df_rtc_ts_wind['vv_avg'], marker='o', color='tab:blue', label='vv_avg')
 
  ax1.set_xlabel('Datetime')
  ax1.set_ylabel('vv_avg', color='tab:blue')
  ax1.tick_params(axis='y', labelcolor='tab:blue')
  ax2 = ax1.twinx()
  ax2.plot(df_rtc_ts_wind['datetime'], df_rtc_ts_wind['vh_avg'], marker='v', color='tab:brown', label='vh_avg')
  ax2.set_ylabel('vh_avg', color='tab:brown')
  ax2.tick_params(axis='y', labelcolor='tab:brown')
  if POL_RATIO_PLOT:
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  
    ax3.plot(df_rtc_ts_wind['datetime'], df_rtc_ts_wind['vv/vh_avg'], marker='P', color='tab:purple', label='vv/vh_avg')
    ax3.set_ylabel('vv/vh_avg', color='tab:purple')
    ax3.tick_params(axis='y', labelcolor='tab:purple')

  change_type = df_site.change_type.iloc[0]
  plt.title(f'Change type {change_type}; {BURST_ID=}; {SITE_ID=}')
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    ax1.set_xticks(df_rtc_ts_wind['datetime'].tolist())
    ax1.set_xticklabels(df_rtc_ts_wind['datetime'].tolist(), rotation=90)

  # Subsample the xticks by N
  N = 2
  ticks = plt.gca().get_xticks()
  new_ticks = ticks[::N]
  new_tick_labels = [str(int(tick)) for tick in ticks[::N]]
  plt.gca().set_xticks(new_ticks)
  plt.gca().set_xticklabels(new_tick_labels)

  ax1.grid(True)

  last_observed_time = df_site['last_observation_time'][0]

  if last_observed_time == pd.NaT:
    ax1.axvline(x=last_observed_time, color='b', linestyle='--', label=f'Last observation time ({last_observed_time})')
  ax2.legend(loc='upper left')

  change_time = df_site['change_time'][0]
  ax1.axvline(x=change_time, color='r', linestyle='--', label=f'Change time ({change_time})')
  ax1.legend()
  fig.savefig(figfile,dpi=300,bbox_inches="tight")
  plt.close(fig)

