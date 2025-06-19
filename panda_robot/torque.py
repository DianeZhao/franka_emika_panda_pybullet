import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag

# cartesian_stiffness_target_array = np.array([300, 300, 300, 50, 50, 50])
cartesian_stiffness_target_array = np.array([30, 30, 30, 50, 50, 50])
cartesian_stiffness_target = np.diag(cartesian_stiffness_target_array)
cartesian_damping_target_array = 2 * np.sqrt(cartesian_stiffness_target_array)
cartesian_damping_target = np.diag(cartesian_damping_target_array)
nullspace_stiffness = 0.01
q_d_nullspace = None #can be read from yaml, if no, then read from initial robot state
        
        
        
def computed_torque_ftip(robot_model, r_d, q_d, pos, vel, q_d_nullspace):

    # ###########################################DYNAMIC COMPENSTTION#########################################
    #tau_inv_dyn = panda.calculate_inverse_dynamics(pos, vel, desired_acc)
    #####################################Cartesian Impedance Control#########################################
    # =============================================
    # 2. Cartesian Impedance Control
    # =============================================
    r,q=robot_model.link_pose_pin()
    error_pos = r - r_d
    # print(r)
    # print(r_d)
    #print(error_pos)
    #print("q",q,q_d)
    error_rot_base = compute_orientation_error(q, q_d, robot_model.ee_pose) 
    error_pos = error_pos.reshape(3, 1)
    #print("pos_e",error_pos)
    error_rot_base = error_rot_base.reshape(3, 1)
    error = np.vstack((error_pos, error_rot_base))
    #print("error",error)
    #rospy.logwarn(f"current error: {error}")#translation error not good, some 
    jacobian = robot_model.jacobian_pin(pos)
    #Q:WHY?
    #print("jacobian",jacobian)
    # 1. Compute end-effector spatial velocity
    velocity = jacobian @ vel  # shape (7,)
    #print("dthetalist", dthetalist) 
    #the 7th joint has large velocity when the gripper stays near to the target, why???
    #BECAUSE THEY ARE DEFINED AS POSITION! when they are initialized, typo
    velocity = velocity.reshape((6,1))
    # print("velocity",velocity)
    #rospy.logwarn(f"current velocity: {velocity}")
    # 2. Compute desired Cartesian wrench
    F_ee_des = -cartesian_stiffness_target @ error - cartesian_damping_target @ velocity  # shape (6,), the result is 6,6
    #print("F_ee_des", F_ee_des)

    jacobian_t = jacobian.T #7,6
    tau_task = jacobian_t @ F_ee_des
    #rospy.logwarn(f"current velocity norm: {np.linalg.norm(velocity[:3])}")
    #rospy.logwarn(f"tau_task: {tau_task}")
    #print("tau-task", tau_task)
    #TOO LARGER, WHICH IS WEIRD
    #THE ROBOT OSCIALLTES QUITE A LOT
    
    
    
    # =============================================
    # 2. Nullspace Control
    # =============================================
    jacobian_transpose_pinv = pseudo_inverse(jacobian_t, damped=True)
    #print(jacobian_transpose_pinv.shape) #6,7
    # Compute nullspace projection matrix
    I = np.eye(7)
    nullspace_projection = I - jacobian_t @ jacobian_transpose_pinv  # 7x7
    #print(nullspace_projection.shape) #(7,7)
    # PD control in nullspace (critically damped)
    q_d_nullspace = q_d_nullspace.reshape((7, 1))

    pos = np.array(pos)
    pos = pos.reshape((7, 1))
    vel = np.array(vel)
    vel = vel.reshape((7,1))

    vector = nullspace_stiffness * (q_d_nullspace - pos) - 2.0 * np.sqrt(nullspace_stiffness) * vel# 7,1
    
    #print(vector)
    # tau_nullspace = nullspace_projection @ (
    #     nullspace_stiffness * (q_d_nullspace - thetalist) -
    #     2.0 * np.sqrt(nullspace_stiffness) * dthetalist)#OTHERWISE the dim of tau will be 7x7
    tau_nullspace = nullspace_projection @ vector
    
    #print("tau-total", tau_nullspace+tau_task)
    #print("tau-nullspace", tau_nullspace)
    return tau_nullspace+tau_task
    
    
        


def compute_orientation_error(orientation: pin.Quaternion, orientation_d: pin.Quaternion, transform: pin.SE3):
    # Ensure shortest rotation (same hemisphere)
    # if orientation_d.coeffs().dot(orientation.coeffs()) < 0.0:
    #     orientation.coeffs()[:] *= -1.0  # Flip quaternion, but readonly
    if orientation_d.coeffs().dot(orientation.coeffs()) < 0.0:
        orientation = pin.Quaternion(-orientation.coeffs())

    # Compute the difference quaternion: q_err = inv(q) * q_des
    q_error = orientation.inverse() * orientation_d
    #rospy.logwarn(f"q error: {q_error}")
    # Take the imaginary part (x, y, z) of the error quaternion
    x,y,z,w = q_error.coeffs()
    error_rot = np.array([x,y,z])

    # Transform the error to the base frame: -R * error
    error_rot_base = -transform.rotation @ error_rot
    # This works as expected, because NumPy treats a (3,) array like a column vector when used with matrix multiplication (@). 
    # So if transform.rotation is a (3, 3) matrix, then: @ error_rot is valid
    # The result is a (3,) array (still flat)

    return error_rot_base

def pseudo_inverse(M, damped=True):
    """
    Compute the (optionally damped) Moore-Penrose pseudoinverse of matrix M.
    Equivalent to the Eigen C++ version using JacobiSVD.
    """
    lambda_ = 0.2 if damped else 0.0

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Compute damped inverse of singular values
    S_inv_damped = np.array([
        s / (s**2 + lambda_**2) if s > 1e-8 else 0.0 for s in S
    ])
    
    # Reconstruct pseudo-inverse
    S_inv_mat = np.diag(S_inv_damped)
    M_pinv = Vt.T @ S_inv_mat @ U.T
    
    return M_pinv


# =============================================
# 2. Kalman Filter Matrix Definitions
# =============================================

class KalmanFilter:
    def __init__(self, dt, pos, vel):
        # dt = 1.0/240.0  # Timestep matching simulation
        state_dim = 14   # 7 positions + 7 velocities
        measurement_dim = 7  # Only positions measured

        # State Transition Matrix (A)
        A = np.eye(state_dim)
        A[:7, 7:] = np.eye(7) * dt  # q_next = q_prev + v_prev*dt

        # Process Noise Covariance (Q)
        process_noise_pos = 1e-6  # Trust position dynamics
        process_noise_vel = 1e-5  # Less trust in velocity
        Q = block_diag(
            np.eye(7) * process_noise_pos,
            np.eye(7) * process_noise_vel
        )

        # Measurement Matrix (H) - Measures positions only
        H = np.hstack([np.eye(7), np.zeros((7, 7))])

        # Measurement Noise Covariance (R)
        sensor_noise = 1e-4  # From sensor characterization
        R = np.eye(7) * sensor_noise

        # Initial State (x0)
        # q_meas, v_meas = panda.get_position_and_velocity()
        x0 = np.concatenate([pos, vel])

        # Initial Covariance (P0)
        position_uncertainty = 0.1  # Initial position std dev (rad)
        velocity_uncertainty = 1.0   # Initial velocity std dev (rad/s)
        P0 = block_diag(
            np.eye(7) * position_uncertainty**2,
            np.eye(7) * velocity_uncertainty**2
        )
        
        self.A = A  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial error covariance
        

    def predict(self):
        """Predict step"""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        """Update step with measurement z"""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return self.x