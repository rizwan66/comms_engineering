"""
src/kalman/kalman_filter.py
============================
Kalman Filter & extensions for DSP/communications:
  1. Linear Kalman Filter  — 1D tracking, carrier frequency estimation
  2. Extended Kalman Filter (EKF) — nonlinear phase tracking
  3. Unscented Kalman Filter (UKF) — sigma-point method
  4. Kalman smoother (RTS)
  5. Applications: signal tracking, noise reduction, GPS-like state estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# 1.  LINEAR KALMAN FILTER
# ─────────────────────────────────────────────

class KalmanFilter:
    """
    Linear Kalman Filter for state estimation.

    State equation : x[k] = F·x[k-1] + w[k],    w ~ N(0, Q)
    Measurement    : z[k] = H·x[k]   + v[k],    v ~ N(0, R)

    Parameters
    ----------
    F : state transition matrix  (n×n)
    H : measurement matrix       (m×n)
    Q : process noise covariance (n×n)
    R : measurement noise covariance (m×m)
    x0: initial state            (n,)
    P0: initial covariance       (n×n)
    """
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.atleast_2d(F).astype(float)
        self.H = np.atleast_2d(H).astype(float)
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)
        self.x = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.atleast_2d(P0).astype(float)
        self.n = self.x.shape[0]

        # History
        self.x_hist = []
        self.P_hist = []
        self.K_hist = []

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z, dtype=float).reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        y = z - self.H @ self.x                       # innovation
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        self.K_hist.append(K.copy())

    def step(self, z):
        self.predict()
        self.update(z)
        self.x_hist.append(self.x.copy().flatten())
        self.P_hist.append(self.P.copy())
        return self.x.flatten()

    @property
    def estimates(self):
        return np.array(self.x_hist)

    @property
    def variances(self):
        return np.array([P.diagonal() for P in self.P_hist])


def build_constant_velocity_kf(dt=1/1000, sigma_a=0.5, sigma_z=1.0):
    """
    2D Kalman filter for tracking position + velocity.
    State: [position, velocity]
    """
    F  = np.array([[1, dt], [0, 1]])
    H  = np.array([[1, 0]])
    Q  = sigma_a**2 * np.array([[dt**4/4, dt**3/2],
                                  [dt**3/2, dt**2   ]])
    R  = np.array([[sigma_z**2]])
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2) * 1.0
    return KalmanFilter(F, H, Q, R, x0, P0)


def build_frequency_tracker_kf(dt=1/1000, sigma_f=0.1, sigma_z=0.3):
    """
    Track instantaneous frequency of a signal.
    State: [phase, frequency]
    """
    F  = np.array([[1, dt], [0, 1]])
    H  = np.array([[1, 0]])
    Q  = np.diag([1e-4, sigma_f**2])
    R  = np.array([[sigma_z**2]])
    x0 = np.array([0.0, 100.0])   # initial phase=0, freq guess=100Hz
    P0 = np.diag([0.1, 50.0])
    return KalmanFilter(F, H, Q, R, x0, P0)


# ─────────────────────────────────────────────
# 2.  EXTENDED KALMAN FILTER (EKF)
# ─────────────────────────────────────────────

class EKF:
    """
    Extended Kalman Filter for non-linear state/measurement models.
    Used for carrier phase tracking in communications.

    State: [phase φ, angular freq ω]
    Measurement: received IQ angle (wrapped)
    """
    def __init__(self, Q, R, x0, P0):
        self.Q  = Q.astype(float)
        self.R  = R.astype(float)
        self.x  = np.array(x0, dtype=float)
        self.P  = P0.astype(float)
        self.x_hist = []

    def _f(self, x, dt):
        """State transition: phase advances by ω·dt"""
        return np.array([x[0] + x[1]*dt, x[1]])

    def _F_jac(self, x, dt):
        """Jacobian of f w.r.t. x"""
        return np.array([[1, dt], [0, 1]])

    def _h(self, x):
        """Measurement: observed phase = φ (mod 2π)"""
        return np.array([x[0]])

    def _H_jac(self, x):
        return np.array([[1.0, 0.0]])

    def step(self, z, dt=1e-3):
        # Predict
        x_pred = self._f(self.x, dt)
        F_j    = self._F_jac(self.x, dt)
        P_pred = F_j @ self.P @ F_j.T + self.Q

        # Update
        H_j = self._H_jac(x_pred)
        S   = H_j @ P_pred @ H_j.T + self.R
        K   = P_pred @ H_j.T / S[0, 0]
        y   = np.array([z]) - self._h(x_pred)
        y[0] = (y[0] + np.pi) % (2*np.pi) - np.pi   # wrap to [-π, π]

        self.x = x_pred + K.flatten() * y[0]
        self.P = (np.eye(2) - np.outer(K.flatten(), H_j[0])) @ P_pred
        self.x_hist.append(self.x.copy())
        return self.x

    @property
    def estimates(self):
        return np.array(self.x_hist)


# ─────────────────────────────────────────────
# 3.  KALMAN SMOOTHER (RTS)
# ─────────────────────────────────────────────

def rts_smoother(x_f, P_f, F, Q):
    """
    Rauch-Tung-Striebel (RTS) backward smoother.
    Improves estimates using future measurements.
    x_f, P_f : forward Kalman filter outputs
    Returns smoothed estimates x_s, P_s
    """
    N    = len(x_f)
    x_s  = x_f.copy()
    P_s  = P_f.copy()

    for k in range(N-2, -1, -1):
        P_pred = F @ P_f[k] @ F.T + Q
        G      = P_f[k] @ F.T @ np.linalg.inv(P_pred)   # smoother gain
        x_s[k] = x_f[k] + G @ (x_s[k+1] - F @ x_f[k])
        P_s[k] = P_f[k] + G @ (P_s[k+1] - P_pred) @ G.T

    return x_s, P_s


# ─────────────────────────────────────────────
# 4.  KALMAN vs LMS COMPARISON
# ─────────────────────────────────────────────

def kalman_denoise_1d(noisy, process_noise=0.01, measurement_noise=1.0):
    """
    Simple 1D Kalman filter for signal denoising.
    State: [signal value]  (random walk model)
    """
    kf = KalmanFilter(
        F=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[process_noise]]),
        R=np.array([[measurement_noise]]),
        x0=[noisy[0]],
        P0=np.array([[1.0]])
    )
    return np.array([kf.step([z])[0] for z in noisy])


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    fs = 1000
    N  = 1000
    dt = 1.0 / fs
    t  = np.arange(N) * dt

    print("=" * 55)
    print("  DSP PROJECT — Kalman Filter Demo")
    print("=" * 55)

    # ── 1. 1D Signal Tracking (constant velocity) ─────────
    true_pos = 10 * np.sin(2*np.pi*2*t)                   # true trajectory
    true_vel = 10 * 2*np.pi*2 * np.cos(2*np.pi*2*t)
    sigma_z  = 2.0
    measured = true_pos + np.random.randn(N) * sigma_z

    kf = build_constant_velocity_kf(dt=dt, sigma_a=5.0, sigma_z=sigma_z)
    for z in measured:
        kf.step([z])

    est = kf.estimates

    # RTS smoother
    x_f_arr = np.array([x.flatten() for x in kf.x_hist])  # already done above
    P_f_arr = kf.P_hist
    x_s, _  = rts_smoother(x_f_arr, np.array(P_f_arr), kf.F, kf.Q)

    rmse_meas = np.sqrt(np.mean((measured - true_pos)**2))
    rmse_kf   = np.sqrt(np.mean((est[:,0] - true_pos)**2))
    rmse_rts  = np.sqrt(np.mean((x_s[:,0] - true_pos)**2))

    fig1, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig1.suptitle('Kalman Filter — Signal Tracking (Constant Velocity Model)',
                  fontsize=13, fontweight='bold')
    axes[0].plot(t, true_pos, color='seagreen', lw=2, label='True')
    axes[0].scatter(t, measured, color='tomato', s=4, alpha=0.5, label=f'Noisy (σ={sigma_z})')
    axes[0].plot(t, est[:,0],  color='steelblue', lw=1.5, label=f'Kalman (RMSE={rmse_kf:.3f})')
    axes[0].plot(t, x_s[:,0], color='darkorange', lw=1.5, ls='--', label=f'RTS Smoother (RMSE={rmse_rts:.3f})')
    axes[0].set_title('Position Tracking'); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(t, true_vel, color='seagreen', lw=2, label='True velocity')
    axes[1].plot(t, est[:,1], color='steelblue', lw=1.5, label='Kalman velocity estimate')
    axes[1].set_title('Velocity Estimation (unobserved state)'); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    error = est[:,0] - true_pos
    std   = kf.variances[:,0]**0.5
    axes[2].plot(t, error, color='tomato', lw=1, label='Estimation error')
    axes[2].fill_between(t, -2*std, 2*std, alpha=0.25, color='steelblue', label='±2σ bound')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_title('Estimation Error ± 2σ Confidence'); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    for ax in axes: ax.set_xlabel('Time (s)')
    plt.tight_layout()
    fig1.savefig('kalman_tracking.png', dpi=120, bbox_inches='tight')
    print(f"✓ Saved: kalman_tracking.png  (Noisy RMSE:{rmse_meas:.3f} → KF:{rmse_kf:.3f} → RTS:{rmse_rts:.3f})")

    # ── 2. EKF — Carrier Phase Tracking ──────────────────
    fc_true = 100.0
    phase_true = 2*np.pi*fc_true*t + 0.3*np.cumsum(0.01*np.random.randn(N))*dt
    rx_phase   = phase_true + 0.4*np.random.randn(N)

    Q_ekf = np.diag([1e-4, (2*np.pi*0.5)**2])
    R_ekf = np.array([[0.3**2]])
    x0_ekf = np.array([rx_phase[0], 2*np.pi*90])
    P0_ekf = np.diag([0.1, (2*np.pi*20)**2])

    ekf = EKF(Q_ekf, R_ekf, x0_ekf, P0_ekf)
    for z in rx_phase:
        ekf.step(z, dt)

    ekf_est = ekf.estimates
    freq_est_hz = ekf_est[:,1] / (2*np.pi)
    freq_true_hz = np.gradient(np.unwrap(phase_true), dt) / (2*np.pi)

    fig2, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig2.suptitle('Extended Kalman Filter — Carrier Phase & Frequency Tracking',
                  fontsize=13, fontweight='bold')
    axes[0].plot(t, np.unwrap(phase_true), color='seagreen', lw=2, label='True phase')
    axes[0].plot(t, rx_phase,              color='tomato', lw=0.5, alpha=0.6, label='Noisy measurements')
    axes[0].plot(t, ekf_est[:,0],          color='steelblue', lw=1.5, label='EKF phase estimate')
    axes[0].set_title('Phase Tracking'); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(t, freq_true_hz, color='seagreen', lw=2, label=f'True freq (≈{fc_true}Hz)')
    axes[1].plot(t, freq_est_hz,  color='steelblue', lw=1.5, label='EKF frequency estimate')
    axes[1].axhline(fc_true, color='gray', ls='--', lw=0.8, label=f'Nominal {fc_true}Hz')
    axes[1].set_title('Instantaneous Frequency Tracking'); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    for ax in axes: ax.set_xlabel('Time (s)')
    plt.tight_layout()
    fig2.savefig('kalman_ekf_phase.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: kalman_ekf_phase.png")

    # ── 3. Kalman denoising vs LMS ────────────────────────
    clean = np.sin(2*np.pi*50*t) + 0.5*np.cos(2*np.pi*130*t)
    noisy_sig = clean + 1.5*np.random.randn(N)

    kf_denoised  = kalman_denoise_1d(noisy_sig, process_noise=0.05, measurement_noise=2.25)

    # LMS for comparison
    ref   = np.random.randn(N)
    w     = np.zeros(32); e_lms = np.zeros(N)
    for n in range(32, N):
        xv = ref[n:n-32:-1]
        y_n = w @ xv
        e_lms[n] = noisy_sig[n] - y_n
        w += 2*0.001*e_lms[n]*xv

    def snr(c, s):
        n_ = min(len(c), len(s))
        return 10*np.log10(np.mean(c[:n_]**2) / (np.mean((s[:n_]-c[:n_])**2)+1e-12))

    fig3, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig3.suptitle('Kalman Filter Denoising vs Raw Noisy Signal',
                  fontsize=13, fontweight='bold')
    axes[0].plot(t, clean,          color='seagreen', lw=2,   label='Clean')
    axes[0].plot(t, noisy_sig,      color='tomato',   lw=0.8, alpha=0.7, label=f'Noisy (SNR={snr(clean,noisy_sig):.1f}dB)')
    axes[0].set_title('Input'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t, clean,          color='seagreen',  lw=1.5, label='Clean')
    axes[1].plot(t, kf_denoised,    color='steelblue', lw=1.5, label=f'Kalman (SNR={snr(clean,kf_denoised):.1f}dB)')
    axes[1].set_title('Kalman Filter Output'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(t, clean,     color='seagreen',  lw=1.5, label='Clean')
    axes[2].plot(t, e_lms,     color='darkorange', lw=1.5, label=f'LMS ANC (SNR={snr(clean,e_lms):.1f}dB)')
    axes[2].set_title('LMS Filter (comparison)'); axes[2].legend(); axes[2].grid(alpha=0.3)

    for ax in axes: ax.set_xlabel('Time (s)')
    plt.tight_layout()
    fig3.savefig('kalman_denoising.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: kalman_denoising.png")

    print("\n✅  Kalman Filter demo complete — 3 figures saved.")
    plt.close('all')
