"""
Adaptive/Robust cascaded controller for a quadcopter under payload variation and wind.
Compares: (1) baseline cascaded PID   (2) Adaptive PID + Disturbance Observer (DOB)
Also models: obstacle crows flying at different heights (the drone dodges up/down/sideways), battery drain + voltage sag, low-battery failsafe (return-to-home / emergency landing),
motor degradation, GPS noise/dropout, and logs mass, velocity, acceleration.
Run:  python quadcopter_sim.py            (all scenarios: PNG plots, CSV logs and quadcopter_results.xlsx in ./results)
      python quadcopter_sim.py --custom --wind const,gust,turb --fault motor,gps --payload sched --crows --T 150
                                           (your own mix: several winds and several faults at once)
      python quadcopter_sim.py --show     (also opens plot windows)
Needs: numpy, matplotlib, pandas
"""
import sys, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- physical parameters ----------------
G, M0, L, KC, TMAX = 9.81, 1.2, 0.23, 0.016, 7.0      # gravity, mass, arm, torque/thrust ratio, max motor thrust
I0 = np.array([0.012, 0.012, 0.022])                   # nominal inertia
PO = np.array([0.12, 0.08])                            # payload attachment offset (x,y) from centre
# ---------------- controller gains ----------------
KP, KD, KI = np.array([3.2, 3.2, 7]), np.array([3.6, 3.6, 5]), np.array([.5, .5, 1.5])
KAP, KAD, KAI = np.array([80, 80, 16.]), np.array([16, 16, 7.]), np.array([25, 25, 0.])
clip = np.clip


# ---------------- environment models ----------------
def payload(t, mode):
    """fraction of nominal mass added"""
    if isinstance(mode, float): return mode if t >= 10 else 0.0       # single payload step at t=10 s
    if mode == "none": return 0.0
    if mode == "fix30": return 0.3
    return 0.0 if t < 10 else 0.1 if t < 20 else 0.2 if t < 30 else 0.3 if t < 52 else 0.0


def gust(t, t0, A, d):
    return A * 0.5 * (1 - np.cos(2 * np.pi * (t - t0) / d)) if t0 < t < t0 + d else 0.0


def wind(t, mode, ou):
    """mode: one name or a list of names to combine ('const', 'gust', 'var', 'turb'); 'mix' = const + gust + turb"""
    ms = ["const", "gust", "turb"] if mode == "mix" else (list(mode) if isinstance(mode, (list, tuple)) else [mode])
    n = len(ms); w = np.zeros(3)
    if "const" in ms: w += np.array([5, 2, 0]) * (.6 if n > 1 else 1)
    if "gust" in ms: w += np.array([gust(t, 10, 9, 4) + gust(t, 30, 7, 5), gust(t, 30, -3, 5), 0])
    if "var" in ms: w += (.6 if n > 1 else 1) * np.array([4 + 3 * np.sin(.4 * t), 2 * np.sin(.25 * t + 1), .6 * np.sin(.7 * t)])
    if "turb" in ms: w = w + ou * (.5 if n > 1 else 1) + (np.array([3, 1, 0]) if n == 1 else 0)
    return w


def reference(t, traj):
    z0, h = 8.0, min(1.0, t / 5)
    rz = z0 * (3 * h**2 - 2 * h**3)
    vz = z0 * (6 * h - 6 * h**2) / 5 if t < 5 else 0.0
    az = z0 * (6 - 12 * h) / 25 if t < 5 else 0.0
    if traj == "hover":
        return np.array([12, 0, rz]), np.array([0, 0, vz]), np.array([0, 0, az])
    R, w = 12.0, 0.25                                   # circle: x=R cos, y=R sin, z=z0
    tau = max(0.0, t - 5); ex = np.exp(-tau / 3)
    th, td, tdd = w * (tau - 3 * (1 - ex)), w * (1 - ex), (w / 3 * ex if tau > 0 else 0)
    c, s = np.cos(th), np.sin(th)
    return (np.array([R * c, R * s, rz]), np.array([-R * s * td, R * c * td, vz]),
            np.array([-R * (c * td**2 + s * tdd), R * (-s * td**2 + c * tdd), az]))


# ---------------- quadcopter: nonlinear 6-DoF + cascaded control ----------------
class Quad:
    def __init__(self, adaptive):
        self.ad = adaptive
        self.p = np.array([12, 0, .15]); self.v = np.zeros(3); self.e = np.zeros(3); self.w = np.zeros(3)
        self.T = np.zeros(4); self.mh = M0
        self.iP = np.zeros(3); self.iA = np.zeros(3)
        self.dh = np.zeros(3); self.dq = np.zeros(3)                    # DOB estimates (accel / ang. accel)
        self.lA = np.array([0, 0, -G]); self.lAl = np.zeros(3); self.aT = np.zeros(3); self.aL = np.zeros(3)
        self.Ih, self.mhU = I0.copy(), M0
        self.used, self.soc, self.vr, self.TM = 0.0, 1.0, 1.0, TMAX      # battery
        self.mode, self.rf, self.lx, self.dead = "fly", None, np.array([12., 0.]), None
        self.cool, self.strikes = 0.0, 0                                   # bird-strike cool-down and counter
        self.rel, self.sat, self.tt, self.gt = False, 0.0, 0.0, 0.0                                   # payload jettisoned?, motor-saturation timer
        self.eff, self.pm = np.ones(4), self.p.copy()                     # motor efficiency, measured position

    # ---- faults / sensing ----
    def sense(self, t, dt, fault, rng):
        if "gps" in fault:                       # noise + 5 s dropout (dead-reckoning)
            if not (25 < t < 30): self.pm = self.p + 0.08 * rng.standard_normal(3)
            else: self.pm = self.pm + (self.v + np.array([.3, -.2, 0])) * dt   # dead-reckoning, IMU bias
        else: self.pm = self.p.copy()
        self.eff = np.array([1, .8, 1, 1.]) if ("motor" in fault and t > 20) else np.ones(4)   # motor 2 loses 20%

    # ---- battery: energy drain and voltage sag (max thrust ~ V^2) ----
    def battery(self, dt, cap):
        self.used += (5 * (self.T * self.eff) ** 1.5).sum() * dt
        if cap > 0: self.soc = max(0.0, 1 - self.used / cap); self.vr = (.72 + .28 * np.sqrt(self.soc)) * (.55 + .45 * min(1.0, self.soc / .03))   # cutoff near empty
        else: self.soc, self.vr = 1.0, 1.0
        self.TM = TMAX * self.vr ** 2

    # ---- failsafe: LOW BATTERY (<20%) -> return home, CRITICAL (<6%) -> emergency land ----
    def go(self, m, rp, rv):
        if self.mode == "fly": self.rf = [rp.copy(), rv.copy(), np.zeros(3)]
        self.lx = np.array([12., 0.]) if self.mode == "rth" else self.p[:2].copy()
        self.mode = m

    def fsm(self, dt, t, rp, rv, fs, pl):
        self.gt = self.gt + dt if np.linalg.norm(rp - self.p) > 25 else 0.0
        self.tt = self.tt + dt if np.abs(self.e[:2]).max() > .7 else 0.0       # sustained extreme tilt
        if fs and t > 8:
            if self.mode == "fly" and (self.soc < .35 or self.gt > 3): self.go("rth", rp, rv)   # low battery OR tracking lost -> return home
            need = M0 * (1 + (0 if self.rel else pl)) * G / (4 * self.TM)         # fraction of max thrust needed to hover
            if self.mode in ("fly", "rth") and (self.soc < .12 or need > .7 or self.sat > 3 or self.tt > .3):
                self.go("land", rp, rv); self.rel = True                           # emergency: jettison payload + land now
            if self.mode == "rth" and np.hypot(self.p[0] - 12, self.p[1]) < 1.2 and np.hypot(*self.v[:2]) < 1:
                self.go("land", rp, rv); self.rel = True                           # unload package at the pad, then land
        if self.mode != "fly":
            p, v, a = self.rf; wn = .7
            tg = np.array([12., 0, 8]) if self.mode == "rth" else np.array([*self.lx, -.5])
            a[:] = wn * wn * (tg - p) - 2 * wn * v; v += a * dt; p += v * dt
            if self.mode == "land" and self.p[2] <= .16 and np.linalg.norm(self.v) < .7: self.dead = "LANDED"

    def control(self, dt, rp, rv, ra):
        ph, th, ps = self.e
        if self.ad:   # disturbance observers: d_hat = LPF(measured - nominal-model prediction)
            gv = np.array([0, 0, G])
            self.dh += (((self.lA - (self.aT - gv)) - self.dh) * min(1, dt * 8))
            self.dq += (((self.lAl - self.aL) - self.dq) * min(1, dt * 30))
        mh = self.mh if self.ad else M0
        Ih = I0 * (1 + 1.5 * (mh / M0 - 1))
        ep = rp - self.pm
        if self.ad and self.p[2] > 1:   # mass adaptation law
            self.mh = clip(self.mh + dt * .1 * (rv[2] - self.v[2] + 1.5 * ep[2]) * G, .9 * M0, 1.7 * M0)
        self.iP = clip(self.iP + ep * dt, -4, 4)
        A = ra + KP * ep + KD * (rv - self.v) + KI * self.iP - (self.dh if self.ad else 0)   # u = u_PID - d_hat
        fz = max(3.0, G + A[2]); ct = max(.5, np.cos(ph) * np.cos(th))
        Tc = clip(mh * fz / ct, 0, 4 * self.TM * .95)
        cp, sp = np.cos(ps), np.sin(ps)
        pd = clip((A[0] * sp - A[1] * cp) / fz, -.45, .45); td = clip((A[0] * cp + A[1] * sp) / fz, -.45, .45)   # tilt limit 26 deg
        er = np.array([pd - ph, td - th, -np.arctan2(sp, cp)])
        self.iA = clip(self.iA + er * dt, -1.2, 1.2)
        al = KAP * er - KAD * self.w + KAI * self.iA - (self.dq if self.ad else 0)
        tau = Ih * al; tau[2] = clip(tau[2], -.05, .05)
        c = np.array([Tc / 4 - tau[1] / (2 * L) - tau[2] / (4 * KC),           # motor allocation
                      Tc / 4 + tau[0] / (2 * L) + tau[2] / (4 * KC),
                      Tc / 4 + tau[1] / (2 * L) - tau[2] / (4 * KC),
                      Tc / 4 - tau[0] / (2 * L) + tau[2] / (4 * KC)])
        self.sat = self.sat + dt if c.max() >= self.TM * .97 else max(0.0, self.sat - dt)    # actuator saturation detector
        self.T += (clip(c, 0, self.TM) - self.T) * (1 - np.exp(-dt / .04))       # motor lag
        self.Ih, self.mhU = Ih, mh

    def dynamics(self, dt, wind_v, pl):
        ph, th, ps = self.e; p, q, r = self.w
        cf, sf, ct, st = np.cos(ph), np.sin(ph), max(.2, np.cos(th)), np.sin(th)
        cp, sp = np.cos(ps), np.sin(ps)
        T = self.T * self.eff; Ta = T.sum(); Ta0 = self.T.sum()          # applied vs commanded thrust
        mp = pl * M0; m = M0 + mp; I = I0 * (1 + 1.5 * pl)
        cgx, cgy = PO[0] * mp / m, PO[1] * mp / m                            # CoG shift from payload
        tm = np.array([L * (T[1] - T[3]), L * (T[2] - T[0]), KC * (-T[0] + T[1] - T[2] + T[3])])
        tx, ty, tz = tm[0] - cgy * Ta, tm[1] + cgx * Ta, tm[2]
        tv = Ta * np.array([cf * st * cp + sf * sp, cf * st * sp - sf * cp, cf * np.cos(th)])   # thrust ~ w^2
        rel = wind_v - self.v
        Fd = .05 * (1 + .3 * pl) * np.linalg.norm(rel) * rel                 # aerodynamic drag from wind
        ac = (tv + Fd) / m - np.array([0, 0, G])
        self.v += ac * dt; self.p += self.v * dt
        al = np.array([(tx + (I[1] - I[2]) * q * r - .01 * p) / I[0],        # Euler's rotational equations
                       (ty + (I[2] - I[0]) * p * r - .01 * q) / I[1],
                       (tz + (I[0] - I[1]) * p * q - .01 * r) / I[2]])
        self.w += al * dt
        tn = st / ct
        self.e += np.array([self.w[0] + (self.w[1] * sf + self.w[2] * cf) * tn,
                            self.w[1] * cf - self.w[2] * sf,
                            (self.w[1] * sf + self.w[2] * cf) / ct]) * dt
        Ih = self.Ih
        self.lA, self.lAl, self.aT = ac, al, tv * (Ta0 / Ta if Ta > 1e-6 else 1) / self.mhU
        T0 = self.T; tm = np.array([L * (T0[1] - T0[3]), L * (T0[2] - T0[0]), KC * (-T0[0] + T0[1] - T0[2] + T0[3])])
        self.aL = np.array([(tm[0] + (Ih[1] - Ih[2]) * q * r) / Ih[0], (tm[1] + (Ih[2] - Ih[0]) * p * r) / Ih[1],
                            (tm[2] + (Ih[0] - Ih[1]) * p * q) / Ih[2]])
        if self.p[2] <= .15:                                                  # ground contact
            if self.v[2] < -2.5: self.dead = "CRASHED"
            self.p[2] = .15; self.v[2] = max(0, self.v[2])
            if Ta < m * G * 1.02: self.v[:] = 0; self.e[:] = 0; self.w[:] = 0


# ---------------- crows: obstacle birds flying at different heights ----------------
class Crows:
    """Each crow flies an ellipse through the drone corridor; its height (3.5-12.5 m, plus a vertical wobble) differs per bird.
    The drone's reference is pushed away from the predicted crow position in 3-D, so a crow passing ABOVE makes the drone dip
    and a crow passing BELOW makes it climb. A crow within 0.6 m gives a bird-strike impulse."""
    def __init__(self, n=6, seed=7):
        r = np.random.default_rng(seed)
        self.a, self.b = 8 + 8 * r.random(n), 8 + 8 * r.random(n); self.cx, self.cy = r.random(n) * 6 - 3, r.random(n) * 6 - 3
        self.w = (.2 + .15 * r.random(n)) * np.where(np.arange(n) % 2, 1, -1); self.ph = 6.28 * r.random(n)
        self.z, self.za, self.wz = 3.5 + 9 * r.random(n), 1.5 + 2.5 * r.random(n), .4 + .8 * r.random(n)
    def pos(self, t):
        return np.stack([self.cx + self.a * np.cos(self.w * t + self.ph), self.cy + self.b * np.sin(self.w * t + self.ph),
                         np.maximum(1.5, self.z + self.za * np.sin(self.wz * t + self.ph))], 1)
    def state(self, t):
        p = self.pos(t); return p, (self.pos(t + .05) - p) / .05
    def avoid(self, ref_p, p, v):
        off = np.zeros(3)
        for pc, vc in zip(p, v):
            d = ref_p - (pc + vc * .7); n = np.linalg.norm(d) + 1e-6
            if n < 9: off += d * ((9 - n) / 9 * 3.5 / n)
        m = np.linalg.norm(off); return off * 4 / m if m > 4 else off


# ---------------- simulation + metrics ----------------
SCENARIOS = {
    "S1_nominal":        dict(traj="circle", wind="none", pl="none",  T=50),
    "S2_payload":        dict(traj="hover",  wind="none", pl="sched", T=45),
    "S3_wind_gust":      dict(traj="hover",  wind="gust", pl="none",  T=30),
    "S4_payload_wind":   dict(traj="circle", wind="mix",  pl="sched", T=60),
    "S5_circle_wind":    dict(traj="circle", wind="turb", pl="none",  T=60),
    "S6_low_battery":    dict(traj="circle", wind="mix",   pl="sched", T=420, cap=34000.0, fs=True),
    "S7_motor_fault":    dict(traj="circle", wind="const", pl="none", T=50, fault="motor"),
    "S8_gps_dropout":    dict(traj="circle", wind="var",  pl="none",  T=50, fault="gps"),
    "S9_multi_hazard":   dict(traj="circle", wind=["const", "gust", "var", "turb"], pl="sched", T=120, cap=34000.0, fault=["motor", "gps", "cell"], crows=True),   # several winds + several faults + crows at different heights
}


def simulate(sc, adaptive, dt=1 / 400, seed=1):
    rng = np.random.default_rng(seed); ou = np.zeros(3)
    q = Quad(adaptive); log = []; n = int(sc["T"] / dt)
    fault, fs = sc.get("fault", "none"), sc.get("fs", True)
    cap = sc.get("cap", 0.0) * (.7 if "cell" in fault else 1.0)          # weak battery fault: -30% capacity
    cols = ["t", "x", "y", "z", "xr", "yr", "zr", "phi", "theta", "psi", "T1", "T2", "T3", "T4", "m_hat", "wind", "payload",
            "vx", "vy", "vz", "ax", "ay", "az", "mass", "soc", "volt", "TM", "mode", "dhx", "dhy", "dhz", "power", "crow_dist", "crow_height", "avoid_dz", "strikes"]
    crows = Crows() if sc.get("crows") else None; offF = np.zeros(3); cd = ch = np.nan; cp = cv = None; j = 0
    def rec(t, rp, w, pl):
        log.append([t, *q.p, *rp, *q.e, *q.T, q.mh, np.linalg.norm(w), pl, *q.v, *q.lA, M0 * (1 + pl), q.soc,
                    16.8 * q.vr, q.TM, q.dead or q.mode, *q.dh, 5 * ((q.T * q.eff) ** 1.5).sum(), cd, ch, offF[2], q.strikes])
    for k in range(n):
        t = k * dt
        ou += -ou * .8 * dt + 2.8 * np.sqrt(dt) * rng.standard_normal(3)
        w, pl = wind(t, sc["wind"], ou), payload(t, sc["pl"])
        rp, rv, ra = reference(t, sc["traj"])
        if crows is not None:                                              # crow avoidance shifts the reference (also vertically)
            cp, cv = crows.state(t); offF = offF + (crows.avoid(rp, cp, cv) - offF) * (1 - np.exp(-dt * 1.0)); rp = rp + offF
            if t > 5: rp[2] = np.clip(rp[2], 2, 16)
            dd = np.linalg.norm(cp - q.p, axis=1); j = dd.argmin(); cd, ch = dd[j], cp[j, 2]
        if q.dead:
            rec(t, rp, w, 0.0 if q.rel else pl); break
        q.sense(t, dt, fault, rng); q.fsm(dt, t, rp, rv, fs, pl); pl = 0.0 if q.rel else pl      # jettisoned payload
        ref = (rp, rv, ra) if q.mode == "fly" else tuple(q.rf)
        q.control(dt, *ref); q.dynamics(dt, w, pl); q.battery(dt, cap)
        q.cool -= dt
        if crows is not None and q.cool <= 0 and cd < .6:                  # bird strike
            d = q.p - cp[j]; q.v += d / (cd + 1e-6) * 2 + cv[j] * .3; q.w[:2] += rng.uniform(-3, 3, 2); q.cool = 1.5; q.strikes += 1
        if k % 10 == 0: rec(t, rp, w, pl)
    return pd.DataFrame(log, columns=cols)


def metrics(df, name, ctrl, gust_end=None):
    d = df[(df.t > 6) & (df['mode'] == 'fly')]
    e = np.linalg.norm(d[["x", "y", "z"]].values - d[["xr", "yr", "zr"]].values, axis=1)
    T = d[["T1", "T2", "T3", "T4"]].values; dt = d.t.diff().mean()
    out = dict(scenario=name, controller=ctrl, MAE=e.mean(), MSE=(e**2).mean(), RMSE=np.sqrt((e**2).mean()),
               max_err=e.max(), max_alt_err=np.abs(d.z - d.zr).max(),
               max_tilt_deg=np.degrees(np.abs(d[["phi", "theta"]].values).max()),
               effort=(T**2).sum() * dt, energy_kJ=(5 * T**1.5).sum() * dt / 1000)
    fs_t = df[df['mode'] != 'fly'].t
    out.update(endurance_s=df.t.iloc[-1], min_battery_pct=100 * df.soc.min(), failsafe_at_s=(fs_t.iloc[0] if len(fs_t) else np.nan),
               final_state=str(df['mode'].iloc[-1]).upper(), pad_dist_m=float(np.hypot(df.x.iloc[-1] - 12, df.y.iloc[-1])))
    if gust_end:   # recovery time: first time after the gust that error stays < 0.3 m
        a = df[df.t > gust_end]; ea = np.linalg.norm(a[["x", "y", "z"]].values - a[["xr", "yr", "zr"]].values, axis=1)
        bad = np.where(ea > .3)[0]
        out["recovery_s"] = (a.t.values[bad[-1]] - gust_end + .025) if len(bad) else 0.0
    return out


def plots(name, res):
    fig = plt.figure(figsize=(14, 9)); fig.suptitle(name)
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    r = res["PID"]; ax.plot(r.xr, r.yr, r.zr, "k--", lw=1, label="reference")
    for c, col in (("PID", "r"), ("Adaptive+DOB", "c")): ax.plot(res[c].x, res[c].y, res[c].z, col, lw=1, label=c)
    ax.set_title("3-D trajectory"); ax.legend()
    ax2 = fig.add_subplot(2, 2, 2); ax3 = fig.add_subplot(2, 2, 3); ax4 = fig.add_subplot(2, 2, 4)
    for c, col in (("PID", "r"), ("Adaptive+DOB", "c")):
        d = res[c]; e = np.linalg.norm(d[["x", "y", "z"]].values - d[["xr", "yr", "zr"]].values, axis=1)
        ax2.plot(d.t, e, col, label=c); ax3.plot(d.t, d.z, col); ax4.plot(d.t, d[["T1", "T2", "T3", "T4"]].sum(axis=1), col)
    d = res["PID"]; ax3.plot(d.t, d.zr, "k--"); ax2.set_title("Tracking error |e| (m)")
    tw = ax2.twinx(); tw.fill_between(d.t, d.wind, color="gray", alpha=.2); tw.plot(d.t, d.payload * 30, "y:"); tw.set_ylabel("wind m/s (grey), payload x30 (yellow)")
    ax3.set_title("Altitude (m)"); ax4.set_title("Total thrust (N)"); ax2.legend()
    for a in (ax2, ax3, ax4): a.set_xlabel("t (s)"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"results/{name}.png", dpi=130)


def flight_data_plot(name, res):
    """every measured quantity: mass, velocity, acceleration, attitude, thrust, power, battery, disturbance estimate"""
    fig, ax = plt.subplots(3, 4, figsize=(19, 11)); fig.suptitle(name + " - measured data (red = PID, cyan = Adaptive+DOB)")
    for c, col in (("PID", "r"), ("Adaptive+DOB", "c")):
        d = res[c]; t = d.t
        ax[0, 0].plot(t, d.m_hat if c != "PID" else np.full(len(d), M0), col)
        ax[0, 1].plot(t, np.linalg.norm(d[["vx", "vy", "vz"]].values, axis=1), col)
        ax[0, 2].plot(t, np.linalg.norm(d[["ax", "ay", "az"]].values, axis=1), col); ax[0, 3].plot(t, d.vz, col)
        ax[1, 0].plot(t, np.degrees(d.phi), col); ax[1, 1].plot(t, np.degrees(d.theta), col)
        ax[1, 2].plot(t, d[["T1", "T2", "T3", "T4"]].sum(axis=1), col); ax[1, 3].plot(t, d.power, col)
        ax[2, 0].plot(t, 100 * d.soc, col); ax[2, 1].plot(t, d.volt, col)
        ax[2, 2].plot(t, d[["T1", "T2", "T3", "T4"]].max(axis=1), col); ax[2, 2].plot(t, d.TM, col + "--", lw=.8)
        if c != "PID":
            for k, cc in zip("xyz", "rgb"): ax[2, 3].plot(t, d["dh" + k], cc, lw=.8, label="d_hat_" + k)
    r = res["PID"]; ax[0, 0].plot(r.t, r.mass, "k--", lw=1); ax[2, 3].legend()
    titles = ["Mass used by controller (kg), dashed = true", "Speed (m/s)", "|acceleration| (m/s^2)", "Vertical velocity (m/s)", "Roll (deg)",
              "Pitch (deg)", "Total thrust (N)", "Electrical power (W)", "Battery (%)", "Voltage (V)", "Max motor thrust (N), dashed = limit",
              "Disturbance estimate by DOB (m/s^2)"]
    for a_, ttl in zip(ax.ravel(), titles): a_.set_title(ttl); a_.grid(alpha=.3); a_.set_xlabel("t (s)")
    fig.tight_layout(); fig.savefig(f"results/{name}_data.png", dpi=110); plt.close(fig)


PAY = [0.0, 0.1, 0.2, 0.3]


def payload_jobs(ex):
    return {(pc, c): ex.submit(simulate, dict(traj="hover", wind="none", pl=(pc if pc > 0 else "none"), T=45), ad)
            for pc in PAY for c, ad in (("PID", False), ("Adaptive+DOB", True))}


def payload_report(res):
    """how payload mass / CoG / inertia change the flight: table + figure"""
    rows = []
    for (pc, c), df in res.items():
        t, e = df.t.values, df.z.values - df.zr.values; post = t >= 10; end = t > 38; mass = M0 * (1 + pc)
        bad = np.where(post & (np.abs(e) > .05))[0]
        tilt = np.degrees(np.maximum(np.abs(df.phi), np.abs(df.theta)).values)
        rows.append(dict(payload_pct=int(pc * 100), controller=c, total_mass_kg=mass, hover_thrust_per_motor_N=mass * G / 4,
                         CoG_shift_mm=np.hypot(*PO) * pc * M0 / mass * 1000, inertia_increase_pct=150 * pc,
                         max_altitude_dip_m=max(0, -e[post].min()), settling_time_s=(t[bad[-1]] - 10) if len(bad) else 0.0,
                         steady_alt_error_m=e[end].mean(), max_tilt_deg=tilt[post].max(), steady_tilt_deg=tilt[end].mean(),
                         avg_power_W=df.power[end].mean(),
                         mass_est_error_pct=(100 * (df.m_hat.iloc[-1] - mass) / mass) if c != "PID" else np.nan))
    tab = pd.DataFrame(rows).round(3)
    fig, ax = plt.subplots(3, 3, figsize=(17, 12)); fig.suptitle("Effect of payload variation (step at t = 10 s, hover at 8 m)")
    cols = plt.cm.viridis(np.linspace(0, .9, len(PAY)))
    for j, c in enumerate(("PID", "Adaptive+DOB")):
        for pc, cc in zip(PAY, cols):
            d = res[(pc, c)]; lab = f"+{int(pc*100)}%"
            ax[0, j].plot(d.t, d.z - d.zr, color=cc, label=lab)
            ax[1, j].plot(d.t, d[["T1", "T2", "T3", "T4"]].sum(axis=1), color=cc, label=lab)
        ax[0, j].set_title(c + ": altitude error (m)"); ax[1, j].set_title(c + ": total thrust (N)"); ax[0, j].legend()
    for pc, cc in zip(PAY, cols):
        d = res[(pc, "Adaptive+DOB")]
        ax[0, 2].plot(d.t, d.m_hat, color=cc); ax[0, 2].axhline(M0 * (1 + pc), color=cc, ls=":")
        ax[1, 2].plot(d.t, np.degrees(np.maximum(np.abs(d.phi), np.abs(d.theta))), color=cc)
    ax[0, 2].set_title("Adaptive: mass estimate (dotted = true mass)"); ax[1, 2].set_title("Adaptive: tilt caused by CoG shift (deg)")
    for k, (col, ttl) in enumerate([("max_altitude_dip_m", "Max altitude dip (m)"), ("settling_time_s", "Settling time to +/-5 cm (s)"),
                                    ("avg_power_W", "Hover power (W)")]):
        for c, off, cc in (("PID", -.2, "r"), ("Adaptive+DOB", .2, "c")):
            q = tab[tab.controller == c]; ax[2, k].bar(np.arange(4) + off, q[col], .4, color=cc, label=c)
        ax[2, k].set_xticks(range(4)); ax[2, k].set_xticklabels([f"+{int(x*100)}%" for x in PAY]); ax[2, k].set_title(ttl); ax[2, k].legend()
    for a_ in ax.ravel(): a_.grid(alpha=.3)
    for a_ in ax[:2].ravel(): a_.set_xlabel("t (s)")
    fig.tight_layout(); fig.savefig("results/payload_effect.png", dpi=120); plt.close(fig)
    tab.to_csv("results/payload_effect.csv", index=False)
    return tab


def export_excel(tab, ptab, sims, psims):
    """Excel workbook: all tables, all raw logs and every graph image (one sheet per graph)"""
    import glob
    path = "results/quadcopter_results.xlsx"
    sc = pd.DataFrame([dict(scenario=n, **{k: (", ".join(v) if isinstance(v, list) else v) for k, v in d.items()}) for n, d in SCENARIOS.items()])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sc.to_excel(xw, sheet_name="Scenarios", index=False); tab.to_excel(xw, sheet_name="Metrics", index=False)
        if ptab is not None: ptab.to_excel(xw, sheet_name="Payload effect", index=False)
        for (name, c), df in sims.items(): df.round(4).to_excel(xw, sheet_name=f"{name.split('_')[0]}_{c.split('+')[0]}"[:31], index=False)
        for (pc, c), df in psims.items(): df.round(4).to_excel(xw, sheet_name=f"payload{int(pc*100)}_{c.split('+')[0]}"[:31], index=False)
    try:
        from openpyxl import load_workbook
        from openpyxl.drawing.image import Image as XImage
        wb = load_workbook(path)
        for png in sorted(glob.glob("results/*.png")):
            ws = wb.create_sheet(("G_" + os.path.basename(png)[:-4])[:31]); im = XImage(png); k = 1100 / im.width
            im.width, im.height = int(im.width * k), int(im.height * k); ws.add_image(im, "A1")
        wb.save(path)
    except Exception as e: print("graph images not added to Excel:", e)


def crow_plot(name, res):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8)); fig.suptitle(name + " - crow heights and the drone's movement (red = PID, cyan = Adaptive+DOB)")
    for c, col in (("PID", "r"), ("Adaptive+DOB", "c")):
        d = res[c]
        ax[0, 0].plot(d.t, d.z, col, label=c + " altitude"); ax[0, 0].plot(d.t, d.crow_height, col + ":", lw=1, label=c + " nearest crow height")
        ax[0, 1].plot(d.t, d.crow_dist, col); ax[1, 0].plot(d.t, d.avoid_dz, col); ax[1, 1].plot(d.t, d.strikes, col)
    ax[0, 0].plot(res["PID"].t, res["PID"].zr, "k--", lw=.8, label="reference altitude (with avoidance)")
    for a_, ttl in zip(ax.ravel(), ["Drone altitude and nearest crow height (m)", "Distance to nearest crow (m)", "Vertical avoidance offset applied to the reference (m)", "Bird strikes (count)"]):
        a_.set_title(ttl); a_.grid(alpha=.3); a_.set_xlabel("t (s)")
    ax[0, 0].legend(fontsize=7); ax[0, 1].axhline(.6, color="k", ls="--", lw=.8)
    fig.tight_layout(); fig.savefig(f"results/{name}_crows.png", dpi=110); plt.close(fig)


def comparison_bars(tab):
    names = list(dict.fromkeys(tab.scenario)); fig, ax = plt.subplots(1, 3, figsize=(18, 5)); x = np.arange(len(names))
    for a_, col, ttl, lg in zip(ax, ["RMSE", "max_err", "energy_kJ"], ["RMSE (m)", "Max error (m)", "Energy (kJ)"], [True, True, False]):
        for c, off, cc in (("PID", -.2, "r"), ("Adaptive+DOB", .2, "c")):
            a_.bar(x + off, tab[tab.controller == c][col].values, .4, color=cc, label=c)
        a_.set_xticks(x); a_.set_xticklabels(names, rotation=35, ha="right"); a_.set_title(ttl); a_.grid(alpha=.3); a_.legend()
        if lg: a_.set_yscale("log")
    fig.tight_layout(); fig.savefig("results/comparison.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--show", action="store_true"); ap.add_argument("--custom", action="store_true", help="run one custom scenario only")
    ap.add_argument("--wind", default="const,gust,turb", help="comma list of: const,gust,var,turb"); ap.add_argument("--fault", default="none", help="comma list of: motor,gps,cell,none")
    ap.add_argument("--payload", default="sched", help="sched | fix30 | none | a number such as 0.2"); ap.add_argument("--traj", default="circle", help="circle | hover")
    ap.add_argument("--cap", type=float, default=34000.0, help="battery energy in joules (0 = unlimited)"); ap.add_argument("--T", type=float, default=150.0); ap.add_argument("--crows", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); rows = []
    if args.custom:
        pl = float(args.payload) if args.payload.replace(".", "", 1).isdigit() else args.payload
        SCENARIOS.clear(); SCENARIOS["Custom"] = dict(traj=args.traj, wind=args.wind.split(","), pl=pl, T=args.T, cap=args.cap, fault=args.fault.split(","), crows=args.crows, fs=True)
    with ProcessPoolExecutor() as ex:                      # independent simulations run in parallel
        jobs = {(n, c): ex.submit(simulate, sc, ad) for n, sc in SCENARIOS.items() for c, ad in (("PID", False), ("Adaptive+DOB", True))}
        pjobs = {} if args.custom else payload_jobs(ex)
        sims = {k: f.result() for k, f in jobs.items()}; psims = {k: f.result() for k, f in pjobs.items()}
    for name, sc in SCENARIOS.items():
        res = {c: sims[(name, c)] for c in ("PID", "Adaptive+DOB")}
        ge = 14 if sc["wind"] == "gust" else None
        for c, df in res.items():
            rows.append(metrics(df, name, c, ge)); df.round(4).to_csv(f"results/{name}_{c.split('+')[0]}.csv", index=False)   # raw measured values
        plots(name, res); plt.close("all"); flight_data_plot(name, res)
        if sc.get("crows"): crow_plot(name, res)
    tab = pd.DataFrame(rows).round(3); tab.to_csv("results/metrics.csv", index=False)
    comparison_bars(tab); ptab = payload_report(psims) if psims else None; export_excel(tab, ptab, sims, psims)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tab.to_string(index=False))
    if ptab is not None: print("\nPAYLOAD EFFECT"); print(ptab.to_string(index=False))
    print("\nAll plots, tables and raw CSV logs are in ./results")
    if "--show" in sys.argv: plt.show()
