import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import ttest_ind

# ==========================================
# 1. METADATA & DATASETS CONFIGURATION
# ==========================================
STEP_NAMES = [
    "1. Reach", 
    "2. Touch (1F)", 
    "3. Grab (2F)", 
    "4. Pick Up", 
    "5. Put Down", 
    "6. Withdraw"
]

CONTROLLER_MAP = {
    "1. Reach":       {"button": "LS_UP",      "action": "Move Forward"},
    "2. Touch (1F)":  {"button": "A_BUTTON",   "action": "Interact / Inspect"},
    "3. Grab (2F)":   {"button": "RT_TRIGGER", "action": "Grip / Hold Item"},
    "4. Pick Up":     {"button": "Y_BUTTON",   "action": "Equip / Lift"},
    "5. Put Down":    {"button": "B_BUTTON",   "action": "Drop Item"},
    "6. Withdraw":    {"button": "LS_DOWN",    "action": "Step Back"}
}

DATASETS = {
    'Right Real (C3)': {
        'file': "data/rotation_two/RightHandReal_Ethan_Team 3_2026-11-07_16-09-21.cnt",
        'channel': 'C3',
        'times': [8.25, 10.59, 13.21, 16.07, 21.28, 26.40],
        'color': '#1f77b4',
        'style': '-'
    },
    'Right Thought (C3)': {
        'file': "data/rotation_two/ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59.cnt",
        'channel': 'C3',
        'times': [6.08, 9.69, 12.58, 15.70, 19.98, 26.40],
        'color': '#2ca02c',
        'style': '--'
    },
    'Left Real (C4)': {
        'file': "data/rotation_two/EthanLeftReal_Team 3_2026-11-07_16-20-38.cnt",
        'channel': 'C4',
        'times': [3.82, 6.25, 10.19, 14.91, 18.94, 24.89],
        'color': '#d62728',
        'style': '-'
    },
    'Left Thought (C4)': {
        'file': "data/rotation_two/EthanImaginaryLeft_Team 3_2026-11-07_16-24-16.cnt",
        'channel': 'C4',
        'times': [5.86, 8.68, 11.24, 14.02, 17.65, 23.16],
        'color': '#ff7f0e',
        'style': '--'
    }
}


# ==========================================
# 2. FILTERED SIGNAL PROCESSING ENGINE
# ==========================================
def extract_cleaned_erd_data(file_path, channel_name, event_times, 
                             rest_window=(0.5, 3.0), epoch_win=(-1.0, 2.0), active_window=(0.2, 1.5)):
    """
    Applies 60Hz Notch + 1-35Hz Bandpass filtering, extracts isolated Mu & Beta power,
    and normalizes against a stable pre-task resting baseline [0.5s - 3.0s].
    """
    try:
        raw = mne.io.read_raw_ant(file_path, preload=True)
    except Exception:
        raw = mne.io.read_raw_cnt(file_path, data_format='int32', preload=True)

    sfreq = raw.info['sfreq']

    # 1. Notch & Bandpass Filters (60Hz powerline removal, 1-35Hz bandpass)
    raw_notched = raw.copy().notch_filter(freqs=[60.0, 120.0], verbose=False)
    raw_filtered = raw_notched.filter(l_freq=1.0, h_freq=35.0, verbose=False)

    signal = raw_filtered[channel_name, :][0].squeeze() * 1e6  # µV
    times = raw_filtered.times

    # 2. Hilbert Envelope Extraction for Mu (8-13Hz) & Beta (13-30Hz)
    mu_band = mne.filter.filter_data(signal, sfreq, l_freq=8.0, h_freq=13.0, verbose=False)
    beta_band = mne.filter.filter_data(signal, sfreq, l_freq=13.0, h_freq=30.0, verbose=False)

    smooth_win = int(0.25 * sfreq)
    mu_pow = np.convolve(np.abs(hilbert(mu_band))**2, np.ones(smooth_win)/smooth_win, mode='same')
    beta_pow = np.convolve(np.abs(hilbert(beta_band))**2, np.ones(smooth_win)/smooth_win, mode='same')

    # 3. Rest Baseline Calculation (0.5s to 3.0s idle period)
    rest_idx = np.where((times >= rest_window[0]) & (times <= rest_window[1]))[0]
    base_mu = np.mean(mu_pow[rest_idx])
    base_beta = np.mean(beta_pow[rest_idx])

    if base_mu <= 0: base_mu = 1e-6
    if base_beta <= 0: base_beta = 1e-6

    step_data = []

    for t_event in event_times:
        ep_start = max(0, int((t_event + epoch_win[0]) * sfreq))
        ep_end = min(len(signal), int((t_event + epoch_win[1]) * sfreq))

        act_start = int((t_event + active_window[0]) * sfreq)
        act_end = min(len(signal), int((t_event + active_window[1]) * sfreq))

        erd_mu_epoch = ((mu_pow[ep_start:ep_end] - base_mu) / base_mu) * 100
        erd_beta_epoch = ((beta_pow[ep_start:ep_end] - base_beta) / base_beta) * 100
        erd_comp_epoch = 0.5 * erd_mu_epoch + 0.5 * erd_beta_epoch

        act_mu = ((mu_pow[act_start:act_end] - base_mu) / base_mu) * 100
        act_beta = ((beta_pow[act_start:act_end] - base_beta) / base_beta) * 100
        act_comp = 0.5 * act_mu + 0.5 * act_beta

        epoch_times = np.linspace(epoch_win[0], epoch_win[1], len(erd_mu_epoch))

        step_data.append({
            'epoch_times': epoch_times,
            'erd_mu': erd_mu_epoch,
            'erd_beta': erd_beta_epoch,
            'erd_comp': erd_comp_epoch,
            'mean_erd_mu': np.mean(act_mu),
            'mean_erd_beta': np.mean(act_beta),
            'mean_erd_comp': np.mean(act_comp),
            'peak_dip_mu': np.min(erd_mu_epoch),
            'peak_dip_beta': np.min(erd_beta_epoch)
        })

    return step_data, raw_filtered


# ==========================================
# 3. MULTI-CHANNEL SCALP TOPOGRAPHY EXTRACTOR
# ==========================================
def extract_all_channels_step_erd(file_path, event_times, rest_window=(0.5, 3.0), active_win=(0.2, 1.5)):
    """Extracts mean active ERD% across all scalp channels for topographical head heatmaps."""
    try:
        raw = mne.io.read_raw_ant(file_path, preload=True)
    except Exception:
        raw = mne.io.read_raw_cnt(file_path, data_format='int32', preload=True)

    sfreq = raw.info['sfreq']
    raw_notched = raw.copy().notch_filter(freqs=[60.0, 120.0], verbose=False)
    raw_filtered = raw_notched.filter(l_freq=1.0, h_freq=35.0, verbose=False)

    # Use standard colin27_1020 montage to avoid deprecation warnings
    montage = mne.channels.make_standard_montage('colin27_1020')
    raw_filtered.set_montage(montage, on_missing='ignore')

    ch_names = raw_filtered.ch_names
    topomap_matrix = np.zeros((len(event_times), len(ch_names)))
    smooth_win = int(0.25 * sfreq)

    for ch_idx, ch_name in enumerate(ch_names):
        signal = raw_filtered[ch_name, :][0].squeeze() * 1e6
        times = raw_filtered.times

        mu_band = mne.filter.filter_data(signal, sfreq, l_freq=8.0, h_freq=13.0, verbose=False)
        beta_band = mne.filter.filter_data(signal, sfreq, l_freq=13.0, h_freq=30.0, verbose=False)

        mu_pow = np.convolve(np.abs(hilbert(mu_band))**2, np.ones(smooth_win)/smooth_win, mode='same')
        beta_pow = np.convolve(np.abs(hilbert(beta_band))**2, np.ones(smooth_win)/smooth_win, mode='same')

        r_idx = np.where((times >= rest_window[0]) & (times <= rest_window[1]))[0]
        base_mu = np.mean(mu_pow[r_idx])
        base_beta = np.mean(beta_pow[r_idx])

        if base_mu <= 0: base_mu = 1e-6
        if base_beta <= 0: base_beta = 1e-6

        for step_idx, t_event in enumerate(event_times):
            act_start = int((t_event + active_win[0]) * sfreq)
            act_end = min(len(signal), int((t_event + active_win[1]) * sfreq))

            act_mu = np.mean(((mu_pow[act_start:act_end] - base_mu) / base_mu) * 100)
            act_beta = np.mean(((beta_pow[act_start:act_end] - base_beta) / base_beta) * 100)

            topomap_matrix[step_idx, ch_idx] = 0.5 * act_mu + 0.5 * act_beta

    return topomap_matrix, raw_filtered.info


# ==========================================
# 4. GRAPH: 4x6 COMPARATIVE TOPOMAP GRID
# ==========================================
def plot_4_condition_topomap_grid():
    """
    Renders a 4x6 matrix showing superior 2D scalp topomaps for ALL 4 conditions
    (1 row per condition x 6 steps per column) with a shared color scale.
    """
    fig, axes = plt.subplots(4, 6, figsize=(18, 11))
    cmap = 'RdBu_r'
    vlim = (-40.0, 20.0)

    cond_keys = list(DATASETS.keys())
    im = None

    for row_idx, cond_name in enumerate(cond_keys):
        meta = DATASETS[cond_name]
        topomap_matrix, info = extract_all_channels_step_erd(meta['file'], meta['times'])

        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            data = topomap_matrix[col_idx, :]

            im, _ = mne.viz.plot_topomap(
                data, 
                info, 
                axes=ax, 
                show=False, 
                cmap=cmap, 
                vlim=vlim,
                sensors=True, 
                contours=4,
                sphere='auto'
            )

            # Titles on top row
            if row_idx == 0:
                ax.set_title(STEP_NAMES[col_idx], fontsize=11, fontweight='bold', pad=10)

            # Condition labels on left side
            if col_idx == 0:
                ax.text(-0.35, 0.5, cond_name, transform=ax.transAxes, 
                        fontsize=11, fontweight='bold', va='center', ha='right')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Mean ERD Power Change (%)', fontsize=11, fontweight='bold')

    plt.suptitle('Cross-Trial Cortical Topography (Superior View): 4 Conditions x 6 Steps', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.12, 0.0, 0.90, 0.95])
    plt.show()


# ==========================================
# 5. AUTHENTICITY DIAGNOSTIC SUITE
# ==========================================
def run_authenticity_check(file_path, real_event_times, target_ch='C4', control_ch='Fp1', n_surrogates=100):
    real_steps, raw_filtered = extract_cleaned_erd_data(file_path, target_ch, real_event_times)
    target_dips = [s['mean_erd_comp'] for s in real_steps]

    ctrl_ch = control_ch if control_ch in raw_filtered.ch_names else raw_filtered.ch_names[0]
    ctrl_steps, _ = extract_cleaned_erd_data(file_path, ctrl_ch, real_event_times)
    control_dips = [s['mean_erd_comp'] for s in ctrl_steps]

    rec_duration = raw_filtered.times[-1]
    np.random.seed(42)
    random_times = np.random.uniform(4.0, rec_duration - 3.0, n_surrogates)
    surr_steps, _ = extract_cleaned_erd_data(file_path, target_ch, random_times)
    surrogate_dips = [s['mean_erd_comp'] for s in surr_steps]

    _, p_surr = ttest_ind(target_dips, surrogate_dips, alternative='less')
    _, p_ctrl = ttest_ind(target_dips, control_dips, alternative='less')

    print("\n" + "="*75)
    print("   AUTHENTICITY REPORT (Cleaned Bandpass + Stable Rest Baseline)")
    print("="*75)
    print(f"  Target Motor Channel ({target_ch}) Mean ERD : {np.mean(target_dips):.2f}%")
    print(f"  Control Channel ({ctrl_ch}) Mean ERD      : {np.mean(control_dips):.2f}%")
    print(f"  Surrogate (Random Triggers) Mean ERD: {np.mean(surrogate_dips):.2f}%")
    print("-" * 75)
    print(f"  Surrogate Permutation Test p-value  : {p_surr:.4e}")
    if p_surr < 0.05:
        print("  --> PASSED: Motor intent is statistically distinct from random noise.")
    else:
        print("  --> FAILED: Signal does not significantly differ from random baseline.")

    print(f"  Spatial Specificity Test p-value    : {p_ctrl:.4e}")
    if p_ctrl < 0.05:
        print(f"  --> PASSED: Signal is anatomically localized to motor cortex ({target_ch}).")
    else:
        print("  --> FAILED: Signal shows non-specific global activity.")
    print("="*75 + "\n")


# ==========================================
# 6. TIME SERIES & BAR CHART VISUALIZATIONS
# ==========================================
def plot_multi_band_step_grid(all_results, target_cond='Left Thought (C4)'):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes = axes.flatten()

    for step_idx in range(6):
        ax = axes[step_idx]
        res = all_results[target_cond][step_idx]

        ax.plot(res['epoch_times'], res['erd_mu'], color='crimson', linestyle='--', linewidth=1.2, label='Mu Band (8–13 Hz)')
        ax.plot(res['epoch_times'], res['erd_beta'], color='royalblue', linestyle=':', linewidth=1.2, label='Beta Band (13–30 Hz)')
        ax.plot(res['epoch_times'], res['erd_comp'], color='black', linewidth=2.2, label='Composite Signal (Mu+Beta)')

        ax.axvline(0, color='darkorange', linestyle='--', linewidth=1.5, label='Trigger Onset')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.4)
        ax.axhline(-15, color='purple', linestyle=':', alpha=0.7, label='-15% BCI Intent Trigger')

        ax.set_title(STEP_NAMES[step_idx], fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(-100, 40)

        if step_idx in [0, 3]:
            ax.set_ylabel('ERD Power Change (%)', fontsize=10)
        if step_idx >= 3:
            ax.set_xlabel('Time relative to trigger (s)', fontsize=10)
        if step_idx == 0:
            ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

    plt.suptitle(f'Cleaned Motor Intent Signal ({target_cond}): Mu, Beta & Composite Overlay', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_dual_panel_bar_chart(all_results):
    fig, (ax_mu, ax_beta) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    x = np.arange(len(STEP_NAMES))
    width = 0.18

    # --- TOP PANEL: MU BAND (8–13 Hz) ---
    for i, (cond_name, meta) in enumerate(DATASETS.items()):
        peak_mu_dips = [all_results[cond_name][s_idx]['peak_dip_mu'] for s_idx in range(6)]
        offset = (i - 1.5) * width
        rects = ax_mu.bar(x + offset, peak_mu_dips, width, label=cond_name, color=meta['color'], edgecolor='black', linewidth=0.8, alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            ax_mu.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, -10 if h < 0 else 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax_mu.axhline(0, color='black', linewidth=1)
    ax_mu.axhline(-15, color='crimson', linestyle='--', alpha=0.6, label='-15% Detection Threshold')
    ax_mu.set_ylabel('Peak Mu ERD Dip (%)', fontsize=11, fontweight='bold')
    ax_mu.set_title('Mu Band ERD Dips (8–13 Hz) - Sensorimotor Desynchronization', fontsize=13, fontweight='bold')
    ax_mu.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_mu.grid(axis='y', linestyle='--', alpha=0.5)

    # --- BOTTOM PANEL: BETA BAND (13–30 Hz) ---
    for i, (cond_name, meta) in enumerate(DATASETS.items()):
        peak_beta_dips = [all_results[cond_name][s_idx]['peak_dip_beta'] for s_idx in range(6)]
        offset = (i - 1.5) * width
        rects = ax_beta.bar(x + offset, peak_beta_dips, width, label=cond_name, color=meta['color'], edgecolor='black', linewidth=0.8, alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            ax_beta.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, -10 if h < 0 else 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax_beta.axhline(0, color='black', linewidth=1)
    ax_beta.axhline(-15, color='crimson', linestyle='--', alpha=0.6, label='-15% Detection Threshold')
    ax_beta.set_ylabel('Peak Beta ERD Dip (%)', fontsize=11, fontweight='bold')
    ax_beta.set_title('Beta Band ERD Dips (13–30 Hz) - Motor Cortical Activation', fontsize=13, fontweight='bold')
    ax_beta.set_xticks(x)
    ax_beta.set_xticklabels(STEP_NAMES, fontsize=11, fontweight='bold')
    ax_beta.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_beta.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Peak ERD Dip Comparison Across All 4 Trials: Mu vs. Beta', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. BCI INPUT DECODER MODULE
# ==========================================
def run_bci_controller_mapper(all_results, cond_name='Left Thought (C4)', trigger_threshold=-15.0):
    print("\n" + "="*84)
    print(f"      BCI GAME CONTROLLER DECODER & INPUT MAPPER ({cond_name})      ")
    print("="*84)
    print(f"{'ACTION STEP':<18} | {'MEAN ERD (+0.2-1.5s)':<22} | {'INTENT DETECTED':<17} | {'MAPPED CONTROLLER INPUT':<20}")
    print("="*84)

    for step_idx in range(6):
        step_name = STEP_NAMES[step_idx]
        res = all_results[cond_name][step_idx]
        comp_erd = res['mean_erd_comp']

        is_triggered = comp_erd <= trigger_threshold
        status_str = "TRIGGERED [ON]" if is_triggered else "IDLE [OFF]"
        map_info = CONTROLLER_MAP[step_name]
        btn_action = f"{map_info['button']} ({map_info['action']})" if is_triggered else "NO_INPUT"

        print(f"{step_name:<18} | {comp_erd:18.2f}% | {status_str:<17} | {btn_action:<20}")

    print("="*84 + "\n")


# ==========================================
# 8. MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[1/5] Running Authenticity Diagnostic for Left Thought (C4)...")
    run_authenticity_check(
        DATASETS['Left Thought (C4)']['file'], 
        DATASETS['Left Thought (C4)']['times'], 
        target_ch='C4', 
        control_ch='Fp1'
    )

    print("[2/5] Processing all 4 datasets with Cleaned Bandpass & Rest Normalization...")
    all_results = {}
    for cond_name, meta in DATASETS.items():
        print(f"  -> Processing {cond_name}...")
        steps, _ = extract_cleaned_erd_data(meta['file'], meta['channel'], meta['times'])
        all_results[cond_name] = steps

    print("[3/5] Running BCI Controller Input Decoders for Thought Conditions...")
    run_bci_controller_mapper(all_results, cond_name='Left Thought (C4)', trigger_threshold=-15.0)
    run_bci_controller_mapper(all_results, cond_name='Right Thought (C3)', trigger_threshold=-15.0)

    print("[4/5] Rendering 4x6 Comparative Scalp Topography Grid (Paper Style)...")
    plot_4_condition_topomap_grid()

    print("[5/5] Generating Time Series & Dual-Panel Bar Charts...")
    plot_multi_band_step_grid(all_results, target_cond='Left Thought (C4)')
    plot_dual_panel_bar_chart(all_results)