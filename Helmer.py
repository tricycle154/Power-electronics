import numpy as np


class HelmerVCBModel:
    """
    Helmer 统计真空断路器电弧重燃模型
    """

    def __init__(self, seed: int = 42):
        # ========== 断路器参数 ==========
        self.I_chop_mean = 5.0
        self.I_chop_sigma = 1.5
        self.U0_rec = 500.0
        self.k_rec = 1e6
        self.I_hf_zero = 0.5
        self.t_open_base = 0.5
        self.dt_async_b = 1.0e-3
        self.dt_async_c = 2.0e-3

        # ========== 随机种子 & 截流值 ==========
        rng = np.random.default_rng(seed)
        self.I_chop = np.array([
            max(0.5, self.I_chop_mean + self.I_chop_sigma * rng.standard_normal()),
            max(0.5, self.I_chop_mean + self.I_chop_sigma * rng.standard_normal()),
            max(0.5, self.I_chop_mean + self.I_chop_sigma * rng.standard_normal()),
        ])

        # ========== 状态变量 ==========
        self.state = np.zeros(3, dtype=int)
        self.t_open = np.full(3, np.inf)
        self.t_last_rec = np.zeros(3)
        self.i_hist = [np.zeros(20) for _ in range(3)]
        self.rec_count = np.zeros(3, dtype=int)

        # ========== 分闸命令时刻 ==========
        self.t_cmd = np.array([
            self.t_open_base,
            self.t_open_base + self.dt_async_b,
            self.t_open_base + self.dt_async_c,
        ])

    def step(self, v_a, v_b, v_c, i_a, i_b, i_c, t1):
        """
        单步计算，输入三相电压电流和时刻

        参数:
            v_a, v_b, v_c: 三相电压 (V)
            i_a, i_b, i_c: 三相电流 (A)
            t1: 时刻 (s)

        返回:
            sw_a, sw_b, sw_c: 开关状态 (1=闭合, 0=断开)
            state_out1: [state_a, state_b, state_c, rec_a, rec_b, rec_c]
        """
        v = [v_a, v_b, v_c]
        i = [i_a, i_b, i_c]
        sw = np.ones(3, dtype=int)

        for ph in range(3):
            self.i_hist[ph] = np.roll(self.i_hist[ph], -1)
            self.i_hist[ph][-1] = i[ph]

            sw[ph], self.state[ph], self.t_open[ph], self.t_last_rec[ph], self.rec_count[ph] = \
                self._helmer_phase(
                    v[ph], i[ph], t1,
                    self.state[ph], self.t_open[ph], self.t_last_rec[ph],
                    self.I_chop[ph], self.t_cmd[ph], self.i_hist[ph], self.rec_count[ph]
                )

        state_out1 = np.concatenate([self.state, self.rec_count])
        return int(sw[0]), int(sw[1]), int(sw[2]), state_out1.tolist()

    def _helmer_phase(self, v, i, t2, state, t_open_p, t_last_rec,
                      I_chop, t_cmd, i_hist, rec_count):
        sw = 1

        if state == 0:
            if t2 >= t_cmd:
                if abs(i) <= I_chop:
                    state = 1
                    t_open_p = t2
                    sw = 0
                elif t2 >= t_cmd + 5e-3:
                    state = 1
                    t_open_p = t2
                    sw = 0

        elif state == 1:
            sw = 0
            dt = t2 - t_open_p
            if dt > 0:
                u_d = self.U0_rec + self.k_rec * dt
                u_trv = abs(v)
                if u_trv > u_d and (t2 - t_last_rec) > 0.1e-3:
                    state = 2
                    t_last_rec = t2
                    rec_count += 1
                    sw = 1

        elif state == 2:
            sw = 1
            dt_reign = t2 - t_last_rec

            hf_quench = False
            if len(i_hist) >= 10 and dt_reign > 0.02e-3:
                for idx in range(1, len(i_hist)):
                    if i_hist[idx - 1] * i_hist[idx] < 0:
                        window_start = max(0, idx - 3)
                        window_end = min(len(i_hist), idx + 4)
                        if np.max(np.abs(i_hist[window_start:window_end])) > 2 * abs(i):
                            hf_quench = True
                            break

            current_quench = (abs(i) < self.I_hf_zero) and (dt_reign > 0.05e-3)

            if hf_quench or current_quench or dt_reign > 0.5e-3:
                state = 1
                t_open_p = t2
                sw = 0

        return sw, state, t_open_p, t_last_rec, rec_count


# ========== 使用示例：外部输入 ==========
if __name__ == '__main__':
    vcb = HelmerVCBModel(seed=42)

    # 示例 1：手动输入一组值
    t = 0.50001
    va, vb, vc = 0.1, -269.5, 269.5
    ia, ib, ic = 0.3, -8.66, 8.66

    sw_a, sw_b, sw_c, state_out = vcb.step(va, vb, vc, ia, ib, ic, t)
    print(f"手动输入: sw=[{sw_a},{sw_b},{sw_c}], states={state_out}")

    # 示例 2：从循环里输入（仿真主循环）
    print("\n--- 仿真主循环 ---")
    for step in range(2000):  # 0.5s ~ 0.52s
        t = 0.5 + step * 1e-5

        # 这里替换成真实输入：从电路方程、文件、或其他模块来
        va = 311 * np.sin(2 * np.pi * 50 * t)
        vb = 311 * np.sin(2 * np.pi * 50 * t - 2 * np.pi / 3)
        vc = 311 * np.sin(2 * np.pi * 50 * t + 2 * np.pi / 3)
        ia = 10 * np.sin(2 * np.pi * 50 * t)
        ib = 10 * np.sin(2 * np.pi * 50 * t - 2 * np.pi / 3)
        ic = 10 * np.sin(2 * np.pi * 50 * t + 2 * np.pi / 3)

        sw_a, sw_b, sw_c, state_out = vcb.step(va, vb, vc, ia, ib, ic, t)

        if step % 500 == 0:
            print(f"t={t * 1000:.2f}ms | sw=[{sw_a},{sw_b},{sw_c}] | states={state_out}")