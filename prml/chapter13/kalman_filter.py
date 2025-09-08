"""
13.3 线性动态系统 - 卡尔曼滤波器 (Kalman Filter)
=================================================

卡尔曼滤波器是处理线性高斯动态系统的最优滤波器。

状态空间模型：
状态方程：x_t = A x_{t-1} + B u_t + w_t, w_t ~ N(0, Q)
观测方程：y_t = C x_t + v_t, v_t ~ N(0, R)

其中：
- x_t: 状态向量
- y_t: 观测向量
- u_t: 控制输入
- A: 状态转移矩阵
- B: 控制矩阵
- C: 观测矩阵
- Q: 过程噪声协方差
- R: 测量噪声协方差

卡尔曼滤波步骤：
1. 预测步（时间更新）：
   x̂_t|t-1 = A x̂_{t-1|t-1} + B u_t
   P_t|t-1 = A P_{t-1|t-1} A^T + Q

2. 更新步（测量更新）：
   K_t = P_t|t-1 C^T (C P_t|t-1 C^T + R)^{-1}  # 卡尔曼增益
   x̂_t|t = x̂_t|t-1 + K_t (y_t - C x̂_t|t-1)   # 状态更新
   P_t|t = (I - K_t C) P_t|t-1                  # 协方差更新

扩展：
- 扩展卡尔曼滤波(EKF)：处理非线性系统
- 无迹卡尔曼滤波(UKF)：更好的非线性近似
- 粒子滤波：任意分布

应用：
- 目标跟踪
- 导航系统
- 信号处理
- 经济预测
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class KalmanFilter:
    """
    标准卡尔曼滤波器
    
    处理线性高斯状态空间模型。
    """
    
    def __init__(self, state_dim: int, obs_dim: int,
                 control_dim: int = 0):
        """
        初始化卡尔曼滤波器
        
        Args:
            state_dim: 状态维度
            obs_dim: 观测维度
            control_dim: 控制输入维度
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.control_dim = control_dim
        
        # 系统矩阵
        self.A = np.eye(state_dim)  # 状态转移矩阵
        self.B = None if control_dim == 0 else np.zeros((state_dim, control_dim))
        self.C = np.zeros((obs_dim, state_dim))  # 观测矩阵
        self.Q = np.eye(state_dim)  # 过程噪声协方差
        self.R = np.eye(obs_dim)    # 测量噪声协方差
        
        # 状态估计
        self.x = np.zeros(state_dim)  # 状态均值
        self.P = np.eye(state_dim)    # 状态协方差
        
        # 历史记录
        self.x_history = []
        self.P_history = []
        
    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测步（时间更新）
        
        Args:
            u: 控制输入
            
        Returns:
            (x_pred, P_pred): 预测的状态和协方差
        """
        # 状态预测
        self.x = self.A @ self.x
        if u is not None and self.B is not None:
            self.x += self.B @ u
        
        # 协方差预测
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新步（测量更新）
        
        Args:
            y: 观测值
            
        Returns:
            (x_post, P_post): 后验状态和协方差
        """
        # 计算卡尔曼增益
        S = self.C @ self.P @ self.C.T + self.R  # 新息协方差
        K = self.P @ self.C.T @ np.linalg.inv(S)  # 卡尔曼增益
        
        # 状态更新
        innovation = y - self.C @ self.x  # 新息（残差）
        self.x = self.x + K @ innovation
        
        # 协方差更新（Joseph形式，数值更稳定）
        I_KC = np.eye(self.state_dim) - K @ self.C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T
        
        # 记录历史
        self.x_history.append(self.x.copy())
        self.P_history.append(self.P.copy())
        
        return self.x.copy(), self.P.copy()
    
    def filter(self, Y: np.ndarray, 
               U: Optional[np.ndarray] = None) -> np.ndarray:
        """
        对整个序列进行滤波
        
        Args:
            Y: 观测序列，shape (T, obs_dim)
            U: 控制序列，shape (T, control_dim)
            
        Returns:
            滤波后的状态序列，shape (T, state_dim)
        """
        T = len(Y)
        X_filtered = np.zeros((T, self.state_dim))
        
        for t in range(T):
            # 预测
            u = U[t] if U is not None else None
            self.predict(u)
            
            # 更新
            self.update(Y[t])
            X_filtered[t] = self.x
        
        return X_filtered
    
    def smooth(self, Y: np.ndarray,
               U: Optional[np.ndarray] = None) -> np.ndarray:
        """
        RTS平滑（Rauch-Tung-Striebel smoother）
        
        后向递推计算平滑估计。
        
        Args:
            Y: 观测序列
            U: 控制序列
            
        Returns:
            平滑后的状态序列
        """
        # 先进行前向滤波
        T = len(Y)
        
        # 存储滤波结果
        x_filt = []
        P_filt = []
        x_pred = []
        P_pred = []
        
        # 前向滤波
        for t in range(T):
            # 预测
            u = U[t] if U is not None else None
            x_p, P_p = self.predict(u)
            x_pred.append(x_p)
            P_pred.append(P_p)
            
            # 更新
            x_f, P_f = self.update(Y[t])
            x_filt.append(x_f)
            P_filt.append(P_f)
        
        # 后向平滑
        x_smooth = [None] * T
        P_smooth = [None] * T
        
        # 初始化最后时刻
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]
        
        # 后向递推
        for t in range(T-2, -1, -1):
            # 平滑增益
            C_t = P_filt[t] @ self.A.T @ np.linalg.inv(P_pred[t+1])
            
            # 平滑估计
            x_smooth[t] = x_filt[t] + C_t @ (x_smooth[t+1] - x_pred[t+1])
            P_smooth[t] = P_filt[t] + C_t @ (P_smooth[t+1] - P_pred[t+1]) @ C_t.T
        
        return np.array(x_smooth)
    
    def likelihood(self, Y: np.ndarray) -> float:
        """
        计算观测序列的对数似然
        
        Args:
            Y: 观测序列
            
        Returns:
            对数似然
        """
        T = len(Y)
        log_likelihood = 0.0
        
        for t in range(T):
            # 预测
            self.predict()
            
            # 新息
            y_pred = self.C @ self.x
            innovation = Y[t] - y_pred
            
            # 新息协方差
            S = self.C @ self.P @ self.C.T + self.R
            
            # 对数似然贡献
            sign, logdet = np.linalg.slogdet(2 * np.pi * S)
            log_likelihood -= 0.5 * (logdet + innovation @ np.linalg.inv(S) @ innovation)
            
            # 更新
            self.update(Y[t])
        
        return log_likelihood


class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器(EKF)
    
    处理非线性系统，通过线性化近似。
    """
    
    def __init__(self, state_dim: int, obs_dim: int,
                 f: callable, h: callable,
                 df_dx: callable, dh_dx: callable):
        """
        初始化EKF
        
        Args:
            state_dim: 状态维度
            obs_dim: 观测维度
            f: 状态转移函数 x_t = f(x_{t-1})
            h: 观测函数 y_t = h(x_t)
            df_dx: f对x的雅可比矩阵
            dh_dx: h对x的雅可比矩阵
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # 非线性函数
        self.f = f
        self.h = h
        self.df_dx = df_dx
        self.dh_dx = dh_dx
        
        # 噪声协方差
        self.Q = np.eye(state_dim)
        self.R = np.eye(obs_dim)
        
        # 状态估计
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """EKF预测步"""
        # 非线性状态预测
        self.x = self.f(self.x)
        
        # 线性化
        F = self.df_dx(self.x)
        
        # 协方差预测
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """EKF更新步"""
        # 线性化观测函数
        H = self.dh_dx(self.x)
        
        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 状态更新
        innovation = y - self.h(self.x)
        self.x = self.x + K @ innovation
        
        # 协方差更新
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P
        
        return self.x.copy(), self.P.copy()


def demonstrate_kalman_filter(show_plot: bool = True) -> None:
    """
    演示卡尔曼滤波器 - 目标跟踪
    """
    print("\n卡尔曼滤波器演示 - 2D目标跟踪")
    print("=" * 60)
    
    # 设置系统参数
    dt = 0.1  # 时间步长
    
    # 状态：[x, y, vx, vy]
    state_dim = 4
    obs_dim = 2  # 只观测位置
    
    # 创建卡尔曼滤波器
    kf = KalmanFilter(state_dim, obs_dim)
    
    # 设置系统矩阵
    # 状态转移矩阵（匀速运动模型）
    kf.A = np.array([
        [1, 0, dt, 0],   # x = x + vx*dt
        [0, 1, 0, dt],   # y = y + vy*dt
        [0, 0, 1, 0],    # vx = vx
        [0, 0, 0, 1]     # vy = vy
    ])
    
    # 观测矩阵（只观测位置）
    kf.C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # 噪声协方差
    q = 0.01  # 过程噪声
    kf.Q = q * np.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt]
    ])
    
    r = 0.1  # 测量噪声
    kf.R = r * np.eye(2)
    
    # 初始状态
    kf.x = np.array([0, 0, 1, 0.5])  # 初始位置和速度
    kf.P = np.eye(4)
    
    # 生成真实轨迹
    T = 100
    true_states = np.zeros((T, 4))
    observations = np.zeros((T, 2))
    
    true_state = np.array([0, 0, 1, 0.5])
    
    np.random.seed(42)
    for t in range(T):
        # 真实状态演化
        true_state = kf.A @ true_state + np.random.multivariate_normal(
            np.zeros(4), kf.Q
        )
        true_states[t] = true_state
        
        # 带噪声的观测
        observation = kf.C @ true_state + np.random.multivariate_normal(
            np.zeros(2), kf.R
        )
        observations[t] = observation
    
    print(f"轨迹长度: {T}个时间步")
    
    # 卡尔曼滤波
    filtered_states = kf.filter(observations)
    
    # 重置滤波器进行平滑
    kf.x = np.array([0, 0, 1, 0.5])
    kf.P = np.eye(4)
    smoothed_states = kf.smooth(observations)
    
    # 计算误差
    filter_error = np.mean(np.linalg.norm(
        filtered_states[:, :2] - true_states[:, :2], axis=1
    ))
    smooth_error = np.mean(np.linalg.norm(
        smoothed_states[:, :2] - true_states[:, :2], axis=1
    ))
    obs_error = np.mean(np.linalg.norm(
        observations - true_states[:, :2], axis=1
    ))
    
    print(f"\n位置误差（平均欧氏距离）:")
    print(f"  观测噪声: {obs_error:.3f}")
    print(f"  滤波估计: {filter_error:.3f}")
    print(f"  平滑估计: {smooth_error:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 2D轨迹
        ax1 = axes[0, 0]
        ax1.plot(true_states[:, 0], true_states[:, 1], 'b-', 
                linewidth=2, label='真实轨迹')
        ax1.scatter(observations[:, 0], observations[:, 1], 
                   c='gray', s=10, alpha=0.5, label='观测')
        ax1.plot(filtered_states[:, 0], filtered_states[:, 1], 'g--',
                linewidth=2, label='滤波估计')
        ax1.plot(smoothed_states[:, 0], smoothed_states[:, 1], 'r:',
                linewidth=2, label='平滑估计')
        ax1.set_xlabel('X位置')
        ax1.set_ylabel('Y位置')
        ax1.set_title('2D目标跟踪')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # X位置时间序列
        ax2 = axes[0, 1]
        time = np.arange(T) * dt
        ax2.plot(time, true_states[:, 0], 'b-', label='真实')
        ax2.scatter(time, observations[:, 0], c='gray', s=5, 
                   alpha=0.5, label='观测')
        ax2.plot(time, filtered_states[:, 0], 'g--', label='滤波')
        ax2.plot(time, smoothed_states[:, 0], 'r:', label='平滑')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('X位置')
        ax2.set_title('X坐标跟踪')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Y位置时间序列
        ax3 = axes[0, 2]
        ax3.plot(time, true_states[:, 1], 'b-', label='真实')
        ax3.scatter(time, observations[:, 1], c='gray', s=5,
                   alpha=0.5, label='观测')
        ax3.plot(time, filtered_states[:, 1], 'g--', label='滤波')
        ax3.plot(time, smoothed_states[:, 1], 'r:', label='平滑')
        ax3.set_xlabel('时间')
        ax3.set_ylabel('Y位置')
        ax3.set_title('Y坐标跟踪')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 速度估计
        ax4 = axes[1, 0]
        ax4.plot(time, true_states[:, 2], 'b-', label='真实Vx')
        ax4.plot(time, true_states[:, 3], 'b--', label='真实Vy')
        ax4.plot(time, filtered_states[:, 2], 'g-', label='滤波Vx')
        ax4.plot(time, filtered_states[:, 3], 'g--', label='滤波Vy')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('速度')
        ax4.set_title('速度估计')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 误差分析
        ax5 = axes[1, 1]
        filter_errors = np.linalg.norm(
            filtered_states[:, :2] - true_states[:, :2], axis=1
        )
        smooth_errors = np.linalg.norm(
            smoothed_states[:, :2] - true_states[:, :2], axis=1
        )
        obs_errors = np.linalg.norm(
            observations - true_states[:, :2], axis=1
        )
        
        ax5.plot(time, obs_errors, 'gray', alpha=0.5, label='观测误差')
        ax5.plot(time, filter_errors, 'g-', label='滤波误差')
        ax5.plot(time, smooth_errors, 'r-', label='平滑误差')
        ax5.set_xlabel('时间')
        ax5.set_ylabel('位置误差')
        ax5.set_title('估计误差')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 不确定性（协方差）
        ax6 = axes[1, 2]
        # 提取位置不确定性（协方差矩阵的迹）
        position_uncertainty = []
        for P in kf.P_history:
            position_uncertainty.append(np.sqrt(P[0, 0] + P[1, 1]))
        
        ax6.plot(time, position_uncertainty, 'b-', linewidth=2)
        ax6.set_xlabel('时间')
        ax6.set_ylabel('位置不确定性')
        ax6.set_title('估计不确定性（√tr(P)）')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('卡尔曼滤波器 - 目标跟踪', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. 滤波显著降低观测噪声")
    print("2. 平滑进一步改善估计")
    print("3. 可以估计未观测的速度")
    print("4. 提供不确定性估计")


def demonstrate_nonlinear_tracking(show_plot: bool = True) -> None:
    """
    演示非线性跟踪（EKF）
    """
    print("\n扩展卡尔曼滤波(EKF)演示 - 非线性跟踪")
    print("=" * 60)
    
    # 非线性系统：极坐标观测
    # 状态：[x, y, vx, vy]
    # 观测：[r, θ] (距离和角度)
    
    dt = 0.1
    
    # 状态转移函数（线性）
    def f(x):
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return A @ x
    
    # 观测函数（非线性：笛卡尔到极坐标）
    def h(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        theta = np.arctan2(x[1], x[0])
        return np.array([r, theta])
    
    # 雅可比矩阵
    def df_dx(x):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def dh_dx(x):
        r = np.sqrt(x[0]**2 + x[1]**2) + 1e-10
        return np.array([
            [x[0]/r, x[1]/r, 0, 0],
            [-x[1]/r**2, x[0]/r**2, 0, 0]
        ])
    
    # 创建EKF
    ekf = ExtendedKalmanFilter(4, 2, f, h, df_dx, dh_dx)
    
    # 设置噪声
    ekf.Q = 0.01 * np.eye(4)
    ekf.R = np.diag([0.1, 0.01])  # 距离和角度测量噪声
    
    # 初始状态
    ekf.x = np.array([10, 0, 0, 1])
    ekf.P = np.eye(4)
    
    # 生成数据
    T = 100
    true_states = []
    observations = []
    filtered_states = []
    
    true_state = np.array([10, 0, 0, 1])
    
    np.random.seed(42)
    for t in range(T):
        # 真实状态
        true_state = f(true_state) + np.random.multivariate_normal(
            np.zeros(4), ekf.Q
        )
        true_states.append(true_state)
        
        # 非线性观测
        obs = h(true_state) + np.random.multivariate_normal(
            np.zeros(2), ekf.R
        )
        observations.append(obs)
        
        # EKF滤波
        ekf.predict()
        ekf.update(obs)
        filtered_states.append(ekf.x.copy())
    
    true_states = np.array(true_states)
    observations = np.array(observations)
    filtered_states = np.array(filtered_states)
    
    # 转换极坐标观测回笛卡尔坐标
    obs_cartesian = np.zeros((T, 2))
    for t in range(T):
        r, theta = observations[t]
        obs_cartesian[t] = [r * np.cos(theta), r * np.sin(theta)]
    
    # 计算误差
    filter_error = np.mean(np.linalg.norm(
        filtered_states[:, :2] - true_states[:, :2], axis=1
    ))
    obs_error = np.mean(np.linalg.norm(
        obs_cartesian - true_states[:, :2], axis=1
    ))
    
    print(f"位置误差:")
    print(f"  观测(转换后): {obs_error:.3f}")
    print(f"  EKF估计: {filter_error:.3f}")
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 2D轨迹
        ax1 = axes[0, 0]
        ax1.plot(true_states[:, 0], true_states[:, 1], 'b-',
                linewidth=2, label='真实轨迹')
        ax1.scatter(obs_cartesian[:, 0], obs_cartesian[:, 1],
                   c='gray', s=10, alpha=0.5, label='观测(笛卡尔)')
        ax1.plot(filtered_states[:, 0], filtered_states[:, 1], 'g--',
                linewidth=2, label='EKF估计')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('非线性观测跟踪')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 极坐标观测
        ax2 = axes[0, 1]
        time = np.arange(T) * dt
        ax2.plot(time, observations[:, 0], 'r-', label='距离r')
        ax2.plot(time, observations[:, 1], 'b-', label='角度θ')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('观测值')
        ax2.set_title('极坐标观测')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 位置误差
        ax3 = axes[1, 0]
        errors = np.linalg.norm(
            filtered_states[:, :2] - true_states[:, :2], axis=1
        )
        ax3.plot(time, errors, 'g-', linewidth=2)
        ax3.set_xlabel('时间')
        ax3.set_ylabel('位置误差')
        ax3.set_title('EKF估计误差')
        ax3.grid(True, alpha=0.3)
        
        # 速度估计
        ax4 = axes[1, 1]
        ax4.plot(time, true_states[:, 2], 'b-', label='真实Vx')
        ax4.plot(time, true_states[:, 3], 'b--', label='真实Vy')
        ax4.plot(time, filtered_states[:, 2], 'g-', label='EKF Vx')
        ax4.plot(time, filtered_states[:, 3], 'g--', label='EKF Vy')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('速度')
        ax4.set_title('速度估计')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('扩展卡尔曼滤波(EKF) - 非线性观测', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n观察：")
    print("1. EKF处理非线性观测")
    print("2. 线性化引入近似误差")
    print("3. 仍然提供良好的跟踪性能")
    print("4. 适用于弱非线性系统")