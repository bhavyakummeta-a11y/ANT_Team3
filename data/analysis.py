import mne
import numpy as np
from scipy.signal import hilbert
from scipy.stats import ttest_ind

def run_pipeline_authenticity_check(file_path, real_event_times, target_ch='C4', control_ch='O1', pre_win=1.0, post_win=2.0, n_surrogates=100):
    """
    Runs Surrogate Shuffle Testing and Off-Target Control Channel Verification
    to prove signals are biologically authentic and not processing artifacts.
    """
    try:
        raw = mne.io.read_raw_ant(file_path, preload=True)
    except Exception:
        raw = mne.io.read_raw_cnt(file_path, data_format='int32', preload=True)

    sfreq = raw.info['sfreq']
    raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=45.0, verbose=False)
    
    # Helper to calculate peak composite dip for a list of timestamps on a given channel
    def get_dips_for_channel(channel_name, timestamps):
        signal = raw_filtered[channel_name, :][0].squeeze() * 1e6
        mu_band = mne.filter.filter_data(signal, sfreq, l_freq=8.0, h_freq=13.0, verbose=False)
        beta_band = mne.filter.filter_data(signal, sfreq, l_freq=13.0, h_freq=30.0, verbose=False)
        
        mu_pow = np.convolve(np.abs(hilbert(mu_band))**2, np.ones(int(0.25*sfreq))/int(0.25*sfreq), mode='same')
        beta_pow = np.convolve(np.abs(hilbert(beta_band))**2, np.ones(int(0.25*sfreq))/int(0.25*sfreq), mode='same')
        
        dips = []
        for t in timestamps:
            s_idx = max(0, int((t - pre_win) * sfreq))
            e_idx = int(t * sfreq)
            end_idx = min(len(signal), int((t + post_win) * sfreq))
            
            b_mu, b_beta = np.mean(mu_pow[s_idx:e_idx]), np.mean(beta_pow[s_idx:e_idx])
            if b_mu == 0 or b_beta == 0: continue
            
            erd_mu = ((mu_pow[e_idx:end_idx] - b_mu) / b_mu) * 100
            erd_beta = ((beta_pow[e_idx:end_idx] - b_beta) / b_beta) * 100
            erd_comp = 0.5 * erd_mu + 0.5 * erd_beta
            dips.append(np.min(erd_comp))
        return dips

    # 1. Real Event Dips on Target Channel (C3/C4)
    target_dips = get_dips_for_channel(target_ch, real_event_times)
    
    # 2. Control Channel Dips (O1 or Fz)
    control_dips = get_dips_for_channel(control_ch if control_ch in raw.ch_names else raw.ch_names[0], real_event_times)

    # 3. Surrogate / Random Timestamps Test
    rec_duration = raw.times[-1]
    np.random.seed(42)
    random_times = np.random.uniform(pre_win + 1.0, rec_duration - post_win - 1.0, n_surrogates)
    surrogate_dips = get_dips_for_channel(target_ch, random_times)

    # 4. Statistical Tests
    t_stat_surr, p_val_surr = ttest_ind(target_dips, surrogate_dips, alternative='less')
    t_stat_ctrl, p_val_ctrl = ttest_ind(target_dips, control_dips, alternative='less')

    print("\n" + "="*70)
    print("      SIGNAL AUTHENTICITY & ARTIFACT DIAGNOSTIC REPORT      ")
    print("="*70)
    print(f"  Target Motor Channel ({target_ch}) Mean Dip : {np.mean(target_dips):.2f}%")
    print(f"  Control Channel ({control_ch}) Mean Dip      : {np.mean(control_dips):.2f}%")
    print(f"  Surrogate (Random Triggers) Mean Dip: {np.mean(surrogate_dips):.2f}%")
    print("-" * 70)
    print(f"  Surrogate Permutation Test p-value  : {p_val_surr:.4e}")
    if p_val_surr < 0.01:
        print("  --> ASSURANCE PASSED: Dips are statistically locked to events, NOT processing artifacts.")
    else:
        print("  --> WARNING: Dips may be driven by baseline noise or filtering artifacts.")
        
    print(f"  Spatial Specificity Test p-value    : {p_val_ctrl:.4e}")
    if p_val_ctrl < 0.05:
        print(f"  --> ASSURANCE PASSED: Signal is localized to motor channel {target_ch}.")
    else:
        print("  --> WARNING: Control channel shows similar dips (potential global artifact).")
    print("="*70 + "\n")

# Run Verification on Left Hand Thought
run_pipeline_authenticity_check(
    file_path="data/rotation_two/EthanImaginaryLeft_Team 3_2026-11-07_16-24-16.cnt",
    real_event_times=[5.86, 8.68, 11.24, 14.02, 17.65, 23.16],
    target_ch='C4',
    control_ch='Fp1'  # Frontal off-target control
)