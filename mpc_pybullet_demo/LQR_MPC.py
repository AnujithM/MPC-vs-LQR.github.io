import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import sys
import pathlib
import pybullet as p
import time

# Params
TARGET_VEL = 1.0  # m/s
L = 0.3  # vehicle wheelbase [m]
T = 1.75  # Prediction Horizon [s]
DT = 0.2  # discretization step [s]


def get_state(robotId):
    robPos, robOrn = p.getBasePositionAndOrientation(robotId)
    linVel, angVel = p.getBaseVelocity(robotId)

    return np.array(
        [
            robPos[0],
            robPos[1],
            np.sqrt(linVel[0] ** 2 + linVel[1] ** 2),
            p.getEulerFromQuaternion(robOrn)[2],
        ]
    )


def set_ctrl(robotId, currVel, acceleration, steeringAngle):
    gearRatio = 1.0 / 21
    steering = [0, 2]
    wheels = [8, 15]
    maxForce = 50

    targetVelocity = currVel + acceleration * DT

    for wheel in wheels:
        p.setJointMotorControl2(
            robotId,
            wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=targetVelocity / gearRatio,
            force=maxForce,
        )

    for steer in steering:
        p.setJointMotorControl2(
            robotId, steer, p.POSITION_CONTROL, targetPosition=steeringAngle
        )


def plot_results(path, x_history, y_history):
    plt.style.use("ggplot")
    plt.figure()
    plt.title("LQR Tracking Results")

    plt.plot(
        path[0, :], path[1, :], c="tab:orange", marker=".", label="reference track"
    )
    plt.plot(
        x_history,
        y_history,
        c="tab:blue",
        marker=".",
        alpha=0.5,
        label="vehicle trajectory",
    )
    plt.axis("equal")
    plt.legend()
    plt.show()


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller."""
    # solve Ricatti equation
    X = solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.inv(R).dot(B.T.dot(X))

    return K


def compute_path_from_wp(x_wp, y_wp, step):
    """Compute a smooth path from waypoints."""
    path = np.zeros((3, 0))  # x, y, heading
    for i in range(len(x_wp) - 1):
        x_segment = np.linspace(x_wp[i], x_wp[i+1], int(np.hypot(x_wp[i+1]-x_wp[i], y_wp[i+1]-y_wp[i])/step))
        y_segment = np.linspace(y_wp[i], y_wp[i+1], int(np.hypot(x_wp[i+1]-x_wp[i], y_wp[i+1]-y_wp[i])/step))
        heading_segment = np.arctan2(y_wp[i+1] - y_wp[i], x_wp[i+1] - x_wp[i])
        heading_segment = np.full_like(x_segment, heading_segment)
        segment = np.vstack((x_segment, y_segment, heading_segment))
        path = np.hstack((path, segment))
    return path


def get_nn_idx(state, path):
    """
    Finds the index of the closest element

    Args:
        state (array-like): 1D array whose first two elements are x-pos and y-pos
        path (ndarray): 2D array of shape (2,N) of x,y points

    Returns:
        int: the index of the closest element
    """
    dx = state[0] - path[0, :]
    dy = state[1] - path[1, :]
    dist = np.hypot(dx, dy)
    nn_idx = np.argmin(dist)
    return nn_idx


def get_ref_trajectory(state, path, target_v, T, DT):
    """Generate reference trajectory."""
    K = int(T / DT)

    xref = np.zeros((4, K))
    ind = get_nn_idx(state, path)

    cdist = np.append(
        [0.0], np.cumsum(np.hypot(np.diff(path[0, :].T), np.diff(path[1, :]).T))
    )
    cdist = np.clip(cdist, cdist[0], cdist[-1])

    start_dist = cdist[ind]
    interp_points = [d * DT * target_v + start_dist for d in range(1, K + 1)]
    xref[0, :] = np.interp(interp_points, cdist, path[0, :])
    xref[1, :] = np.interp(interp_points, cdist, path[1, :])
    xref[2, :] = target_v
    xref[3, :] = np.interp(interp_points, cdist, path[2, :])

    # points where the vehicle is at the end of trajectory
    xref_cdist = np.interp(interp_points, cdist, cdist)
    stop_idx = np.where(xref_cdist == cdist[-1])
    xref[2, stop_idx] = 0.0

    # transform in ego frame
    dx = xref[0, :] - state[0]
    dy = xref[1, :] - state[1]
    xref[0, :] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])  # X
    xref[1, :] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])  # Y
    xref[3, :] = path[2, ind] - state[3]  # Theta

    def fix_angle_reference(angle_ref, angle_init):
        """
        Removes jumps greater than 2PI to smooth the heading

        Args:
            angle_ref (array-like):
            angle_init (float):

        Returns:
            array-like:
        """
        diff_angle = angle_ref - angle_init
        diff_angle = np.unwrap(diff_angle)
        return angle_init + diff_angle

    xref[3, :] = (xref[3, :] + np.pi) % (2.0 * np.pi) - np.pi
    xref[3, :] = fix_angle_reference(xref[3, :], xref[3, 0])

    return xref


def run_sim():
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=-90,
        cameraPitch=-45,
        cameraTargetPosition=[-0.1, -0.0, 0.65],
    )

    p.resetSimulation()

    p.setGravity(0, 0, -10)
    useRealTimeSim = 1

    p.setTimeStep(1.0 / 120.0)
    p.setRealTimeSimulation(useRealTimeSim)  # either this

    file_path = pathlib.Path(__file__).parent.resolve()
    plane = p.loadURDF(str(file_path) + "/racecar/plane.urdf")
    car = p.loadURDF(
        str(file_path) + "/racecar/f10_racecar/racecar_differential.urdf", [0, 0, 0.3]
    )

    for wheel in range(p.getNumJoints(car)):
        p.setJointMotorControl2(
            car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
        )
        p.getJointInfo(car, wheel)

    c = p.createConstraint(
        car,
        9,
        car,
        11,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(
        car,
        10,
        car,
        13,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        9,
        car,
        13,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        16,
        car,
        18,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(
        car,
        16,
        car,
        19,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        17,
        car,
        19,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        1,
        car,
        18,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    c = p.createConstraint(
        car,
        3,
        car,
        19,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    # Interpolated Path to follow given waypoints
    path = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, 3, -1, -6, -2, -2],
        0.05,
    )

    # Draw the gray road
    road_width = 0.17  # half of the total width of the road
    road_height = 0.01  # height of the road

    for i in range(path.shape[1] - 1):
        x1, y1 = path[0, i], path[1, i]
        x2, y2 = path[0, i + 1], path[1, i + 1]

        # Calculate the length of the road segment
        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Create a visual shape for the road segment
        road_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            halfExtents=[road_width, segment_length / 2 + 0.3, road_height]
        )

        # Calculate the position and orientation for the road segment
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        angle = np.arctan2(y2 - y1, x2 - x1)

        p.createMultiBody(
            baseVisualShapeIndex=road_shape,
            basePosition=[center_x, center_y, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, angle])
        )

    # Draw the reference trajectory in yellow
    for x_, y_ in zip(path[0, :], path[1, :]):
        p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.05], [1, 1, 0])

    # starting conditions
    control = np.zeros(2)

    # LQR Matrices
    Q = np.diag([40, 40, 10, 40])  # state error cost
    R = np.diag([30, 30])  # input cost

    # Linear system dynamics for a simple vehicle model
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    # Compute LQR gain
    K = lqr(A, B, Q, R)

    x_history = []
    y_history = []

    input("\033[92m Press Enter to continue... \033[0m")

    while 1:
        state = get_state(car)
        x_history.append(state[0])
        y_history.append(state[1])

        # track path in bullet
        p.addUserDebugLine(
            [state[0], state[1], 0], [state[0], state[1], 0.5], [1, 0, 0]
        )

        if np.sqrt((state[0] - path[0, -1]) ** 2 + (state[1] - path[1, -1]) ** 2) < 0.2:
            print("Success! Goal Reached")
            set_ctrl(car, 0, 0, 0)
            plot_results(path, x_history, y_history)
            input("Press Enter to continue...")
            p.disconnect()
            return

        # Get Reference_traj
        # NOTE: inputs are in world frame
        target = get_ref_trajectory(state, path, TARGET_VEL, T, DT)

        # for LQR base link frame is used:
        # so x, y, yaw are 0.0, but speed is the same
        ego_state = np.array([0.0, 0.0, state[2], 0.0])

        # Compute the control action using LQR
        control = -K.dot(ego_state - target[:, 0])

        set_ctrl(car, state[2], control[0], control[1])

        start = time.time()
        elapsed = time.time() - start
        print("LQR Computation Time: {:.4f}s".format(elapsed))

        if DT - elapsed > 0:
            time.sleep(DT - elapsed)


if __name__ == "__main__":
    run_sim()

