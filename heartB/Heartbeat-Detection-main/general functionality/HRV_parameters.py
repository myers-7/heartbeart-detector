import numpy as np 

def get_time_domain_features(nn_intervals):
    
    """
    Returns a dictionary containing time domain features for HRV analysis.
    Mostly used on long term recordings (24h) but some studies use some of those features on
    short term recordings, from 1 to 5 minutes window.

    Parameters
    ----------
    nn_intervals : list
        list of Normal to Normal Interval

    Returns
    -------
    time_domain_features : dict
        dictionary containing time domain features for HRV analyses. There are details
        about each features below.

    Notes
    -----
    Here are some details about feature engineering...

    - **mean_nni**: The mean of RR-intervals.

    - **sdnn** : The standard deviation of the time interval between successive normal heart beats \
    (i.e. the RR-intervals).

    - **sdsd**: The standard deviation of differences between adjacent RR-intervals

    - **rmssd**: The square root of the mean of the sum of the squares of differences between \
    adjacent NN-intervals. Reflects high frequency (fast or parasympathetic) influences on hrV \
    (*i.e.*, those influencing larger changes from one beat to the next).

    - **median_nni**: Median Absolute values of the successive differences between the RR-intervals.

    - **nni_50**: Number of interval differences of successive RR-intervals greater than 50 ms.

    - **pnni_50**: The proportion derived by dividing nni_50 (The number of interval differences \
    of successive RR-intervals greater than 50 ms) by the total number of RR-intervals.

    - **nni_20**: Number of interval differences of successive RR-intervals greater than 20 ms.

    - **pnni_20**: The proportion derived by dividing nni_20 (The number of interval differences \
    of successive RR-intervals greater than 20 ms) by the total number of RR-intervals.

    - **range_nni**: difference between the maximum and minimum nn_interval.

    - **cvsd**: Coefficient of variation of successive differences equal to the rmssd divided by \
    mean_nni.

    - **cvnni**: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.

    - **mean_hr**: The mean Heart Rate.

    - **max_hr**: Max heart rate.

    - **min_hr**: Min heart rate.

    - **std_hr**: Standard deviation of heart rate.

    References
    ----------
    .. [1] Heart rate variability - Standards of measurement, physiological interpretation, and \
    clinical use, Task Force of The European Society of Cardiology and The North American Society \
    of Pacing and Electrophysiology, 1996
    """
    import numpy as np

    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)

    # Basic statistics
    mean_nni = np.mean(nn_intervals)
    median_hrv =abs(np.median(diff_nni))
    range_nni = max(nn_intervals) - min(nn_intervals)

    sdsd = np.std(diff_nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / length_int

    # Feature found on github and not in documentation
    cvsd = rmssd / mean_nni

    # Features only for long term recordings
    sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1
    cvnni = sdnn / mean_nni

    # Heart Rate equivalent features
    heart_rate_list = np.divide(60000, nn_intervals)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)

    time_domain_features = {
        'mean_nni': mean_nni,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_20': nni_20,
        'pnni_20': pnni_20,
        'rmssd': rmssd,
        'median_hrv': median_hrv,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr,
    }

    return time_domain_features

def get_hrv_features(Peak_indx,fs):
    
    RR_int=np.diff(Peak_indx)/fs    
    time_domain_features = get_time_domain_features(RR_int*1000)
    
    return time_domain_features
