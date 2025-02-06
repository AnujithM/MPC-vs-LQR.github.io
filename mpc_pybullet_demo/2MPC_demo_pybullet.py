import numpy as np
import matplotlib.pyplot as plt

from cvxpy_mpc.utils import compute_path_from_wp, get_ref_trajectory
from cvxpy_mpc import MPC, VehicleModel
from scipy.linalg import solve_continuous_are

import sys
import time
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

def plot_results(path1, x_history1, y_history1, path2, x_history2, y_history2):
    plt.style.use("ggplot")
    plt.figure()
    plt.title("MPC vs LQR")

    plt.plot(
        path1[0, :], path1[1, :], c="tab:orange", marker=".", label="reference track 1"
    )
    plt.plot(
        x_history1,
        y_history1,
        c="tab:blue",
        marker=".",
        alpha=0.5,
        label="MPC",
    )
    plt.plot(
        path2[0, :], path2[1, :], c="tab:green", marker=".", label="reference track 2"
    )
    plt.plot(
        x_history2,
        y_history2,
        c="tab:red",
        marker=".",
        alpha=0.5,
        label="LQR",
    )
    plt.axis("equal")
    plt.legend()
    plt.show()

def create_vehicle(file_path, urdf_name, start_position):
    vehicle = p.loadURDF(str(file_path) + urdf_name, start_position)
    for wheel in range(p.getNumJoints(vehicle)):
        p.setJointMotorControl2(
            vehicle, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
        )
        p.getJointInfo(vehicle, wheel)

    constraints = [
        (9, 11, 1), (10, 13, -1), (9, 13, -1), (16, 18, 1),
        (16, 19, -1), (17, 19, -1), (1, 18, -1, 15), (3, 19, -1, 15)
    ]
    for c in constraints:
        parent, child, gear_ratio = c[:3]
        gear_aux_link = c[3] if len(c) > 3 else -1
        constraint_id = p.createConstraint(
            vehicle,
            parent,
            vehicle,
            child,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(constraint_id, gearRatio=gear_ratio, gearAuxLink=gear_aux_link, maxForce=10000)
    return vehicle

def disable_collision_between_vehicles(vehicle1, vehicle2):
    for link1 in range(-1, p.getNumJoints(vehicle1)):
        for link2 in range(-1, p.getNumJoints(vehicle2)):
            p.setCollisionFilterPair(vehicle1, vehicle2, link1, link2, enableCollision=0)

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller."""
    X = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R).dot(B.T.dot(X))
    return K

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
    
    car1 = create_vehicle(file_path, "/racecar/f10_racecar/racecar_differential.urdf", [0, 0, 0.3])
    car2 = create_vehicle(file_path, "/racecar/f10_racecar/racecar_differential1.urdf", [0, 1, 0.3])

    disable_collision_between_vehicles(car1, car2)

    # Interpolated Path to follow given waypoints for car1
    path1 = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, 3, -1, -6, -2, -2],
        0.05,
    )

    # Interpolated Path to follow given waypoints for car2, shifted by 1 in y
    path2 = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [1, 1, 3, 5, 4, 4, 0, -5, -1, -1],
        0.05,
    )

    # Draw the gray road for path1
    road_width = 0.17  # half of the total width of the road
    road_height = 0.01  # height of the road

    for i in range(path1.shape[1] - 1):
        x1, y1 = path1[0, i], path1[1, i]
        x2, y2 = path1[0, i + 1], path1[1, i + 1]

        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        road_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            halfExtents=[road_width, segment_length / 2 + 0.3, road_height]
        )

        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        angle = np.arctan2(y2 - y1, x2 - x1)

        p.createMultiBody(
            baseVisualShapeIndex=road_shape,
            basePosition=[center_x, center_y, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, angle])
        )

    for i in range(path2.shape[1] - 1):
        x1, y1 = path2[0, i], path2[1, i]
        x2, y2 = path2[0, i + 1], path2[1, i + 1]

        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        road_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            halfExtents=[road_width, segment_length / 2 + 0.3, road_height]
        )

        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        angle = np.arctan2(y2 - y1, x2 - x1)

        p.createMultiBody(
            baseVisualShapeIndex=road_shape,
            basePosition=[center_x, center_y, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, angle])
        )

    for x_, y_ in zip(path1[0, :], path1[1, :]):
        p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.05], [1, 1, 0])

    for x_, y_ in zip(path2[0, :], path2[1, :]):
        p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.05], [1, 1, 0])

    control1 = np.zeros(2)
    control2 = np.zeros(2)

    Q_mpc = [20, 20, 10, 20]
    Qf_mpc = [30, 30, 30, 30]
    R_mpc = [10, 10]
    P_mpc = [10, 10]

    mpc = MPC(VehicleModel(), T, DT, Q_mpc, Qf_mpc, R_mpc, P_mpc)

    Q_lqr = np.diag([40, 40, 10, 40])
    R_lqr = np.diag([30, 30])

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

    K = lqr(A, B, Q_lqr, R_lqr)

    x_history1 = []
    y_history1 = []
    x_history2 = []
    y_history2 = []

    input("\033[92m Press Enter to continue... \033[0m")

    while True:
        state1 = get_state(car1)
        state2 = get_state(car2)
        x_history1.append(state1[0])
        y_history1.append(state1[1])
        x_history2.append(state2[0])
        y_history2.append(state2[1])

        p.addUserDebugLine(
            [state1[0], state1[1], 0], [state1[0], state1[1], 0.5], [1, 0, 0]
        )
        p.addUserDebugLine(
            [state2[0], state2[1], 0], [state2[0], state2[1], 0.5], [0, 0, 1]
        )

        if np.sqrt((state1[0] - path1[0, -1]) ** 2 + (state1[1] - path1[1, -1]) ** 2) < 0.2:
            print("Success! Goal Reached for Car 1")
            set_ctrl(car1, 0, 0, 0)
            plot_results(path1, x_history1, y_history1, path2, x_history2, y_history2)
            input("Press Enter to continue...")
            p.disconnect()
            return

        if np.sqrt((state2[0] - path2[0, -1]) ** 2 + (state2[1] - path2[1, -1]) ** 2) < 0.2:
            print("Success! Goal Reached for Car 2")
            set_ctrl(car2, 0, 0, 0)
            plot_results(path1, x_history1, y_history1, path2, x_history2, y_history2)
            input("Press Enter to continue...")
            p.disconnect()
            return

        target1 = get_ref_trajectory(state1, path1, TARGET_VEL, T, DT)
        target2 = get_ref_trajectory(state2, path2, TARGET_VEL, T, DT)

        ego_state1 = np.array([0.0, 0.0, state1[2], 0.0])
        ego_state2 = np.array([0.0, 0.0, state2[2], 0.0])

        ego_state1[0] = ego_state1[0] + ego_state1[2] * np.cos(ego_state1[3]) * DT
        ego_state1[1] = ego_state1[1] + ego_state1[2] * np.sin(ego_state1[3]) * DT
        ego_state1[2] = ego_state1[2] + control1[0] * DT
        ego_state1[3] = ego_state1[3] + control1[0] * np.tan(control1[1]) / L * DT

        ego_state2[0] = ego_state2[0] + ego_state2[2] * np.cos(ego_state2[3]) * DT
        ego_state2[1] = ego_state2[1] + ego_state2[2] * np.sin(ego_state2[3]) * DT
        ego_state2[2] = ego_state2[2] + control2[0] * DT
        ego_state2[3] = ego_state2[3] + control2[0] * np.tan(control2[1]) / L * DT

        start = time.time()

        _, u_mpc1 = mpc.step(ego_state1, target1, control1, verbose=False)
        control1[0] = u_mpc1.value[0, 0]
        control1[1] = u_mpc1.value[1, 0]

        control2 = -K.dot(ego_state2 - target2[:, 0])

        elapsed = time.time() - start
        print("CVXPY Optimization Time: {:.4f}s".format(elapsed))

        set_ctrl(car1, state1[2], control1[0], control1[1])
        set_ctrl(car2, state2[2], control2[0], control2[1])

        if DT - elapsed > 0:
            time.sleep(DT - elapsed)


if __name__ == "__main__":
    run_sim()
