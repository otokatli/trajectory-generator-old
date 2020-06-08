import matplotlib.pyplot as plt
from numpy import array, linspace
from trajectory_generator import Trajectory
import seaborn as sns

sns.set()

if __name__ == '__main__':
    q0 = array([0.0, 0.0, 0.0])
    qf = array([1.0, 0.0, 0.0])
    t0 = 0.0
    tf = 10.0

    fig, (ax_pos, ax_vel, ax_acc) = plt.subplots(3, 1)
    fig.tight_layout()
    t_vec = linspace(t0, tf, 1001)

    # Linear trajectory
    T_linear = Trajectory(t0, tf, q0[0], qf[0], n=1)
    x_linear = array([T_linear.q(t) for t in t_vec])
    xp_linear = array([T_linear.qp(t) for t in t_vec])
    xpp_linear = array([T_linear.qpp(t) for t in t_vec])
    ax_pos.plot(t_vec, x_linear, label='Linear')
    ax_vel.plot(t_vec, xp_linear)
    ax_acc.plot(t_vec, xpp_linear)
    print("Linear trajectory")
    print("Coefficients of the trajectory polynomial:", T_linear.a)

    # Cubic trajectory
    T_cubic = Trajectory(t0, tf, q0[0:2], qf[0:2], n=3)
    x_cubic = array([T_cubic.q(t) for t in t_vec])
    xp_cubic = array([T_cubic.qp(t) for t in t_vec])
    xpp_cubic = array([T_cubic.qpp(t) for t in t_vec])
    ax_pos.plot(t_vec, x_cubic, label='Cubic')
    ax_vel.plot(t_vec, xp_cubic)
    ax_acc.plot(t_vec, xpp_cubic)
    print("Cubic trajectory")
    print("Coefficients of the trajectory polynomial:", T_cubic.a)

    # Quintic trajectory
    T_quintic = Trajectory(t0, tf, q0, qf, n=5)
    x_quintic = array([T_quintic.q(t) for t in t_vec])
    xp_quintic = array([T_quintic.qp(t) for t in t_vec])
    xpp_quintic = array([T_quintic.qpp(t) for t in t_vec])
    ax_pos.plot(t_vec, x_quintic, label='Quintic')
    ax_vel.plot(t_vec, xp_quintic)
    ax_acc.plot(t_vec, xpp_quintic)
    print("Quintic trajectory")
    print("Coefficients of the trajectory polynomial:", T_quintic.a)

    ax_acc.set_xlabel('Time [s]')
    ax_pos.set_ylabel('Position [m]')
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_acc.set_ylabel('Acceleration [m/s^2]')
    ax_pos.legend()
    plt.savefig('trajectories.pdf')
