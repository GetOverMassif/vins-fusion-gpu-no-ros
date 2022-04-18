import math
import numpy as np

"""
约定一些数据的格式：
    trans 平移变量 Translation [x,y,z]

    quat  四元数  Quaternion  [qw,qx,qy,qz]

    rotm  旋转矩阵 Rotation matrix [[r00,r01,r02],
                                   [r10,r11,r12],
                                   [r20,r21,r22]]

    axang 轴角    Axis-angle [x,y,z,angle(radian)]

    tform 齐次变换 Homogeneous Transformation [[r00,r01,r02,x],
                                              [r10,r11,r12,y],
                                              [r20,r21,r22,z],
                                              [  0,  0,  0,1],]

    eul   欧拉角   Euler Angles []
"""

def quat2rotm(q_):
    q0 = q_[0]
    q1 = q_[1]
    q2 = q_[2]
    q3 = q_[3]

    r11 = 2*(q0**2+q1**2)-1
    r12 = 2*(q1*q2-q0*q3)
    r13 = 2*(q1*q3+q0*q2)

    r21 = 2*(q1*q2+q0*q3)
    r22 = 2*(q0**2+q2**2)-1
    r23 = 2*(q2*q3-q0*q1)

    r31 = 2*(q1*q3-q0*q2)
    r32 = 2*(q2*q3+q0*q1)
    r33 = 2*(q0**2+q3**2)-1

    rotation = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    return rotation


def rotm2quat(r_):
    r = math.sqrt(1+np.trace(r_))
    s = 1/(2*r)
    w = r / 2
    x = (r_[2,1]-r_[1,2])*s
    y = (r_[0,2]-r_[2,0])*s
    z = (r_[1,0]-r_[0,1])*s
    quat = np.array([w,x,y,z])
    return quat

def rotm2axang(r_):
    r = r_
    u1 = r[2,1]-r[1,2]
    u2 = r[0,2]-r[2,0]
    u3 = r[1,0]-r[0,1]
    u_norm = (u1**2+u2**2+u3**2)**0.5
    u1 = u1/u_norm
    u2 = u2/u_norm
    u3 = u3/u_norm
    theta = math.acos((np.trace(r)-1)/2)
    axang = np.array([u1,u2,u3,theta])
    return axang


def axang2rotm(axang_):
    ux = axang_[0]
    uy = axang_[1]
    uz = axang_[2]
    theta = axang_[3]

    c = math.cos(theta)
    s = math.sin(theta)

    rotm = np.array([[   c+ux**2*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                     [uy*ux*(1-c)+uz*s,    c+uy**2*(1-c), uy*uz*(1-c)-ux*s],
                     [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s,    c+uz**2*(1-c)]])
    return rotm


def merge1(q1_,q2_,h_):
    h = h_
    q_w_ = (1.0-h)*q1_[0]+h*q2_[0]
    q_x_ = (1.0-h)*q1_[1]+h*q2_[1]
    q_y_ = (1.0-h)*q1_[2]+h*q2_[2]
    q_z_ = (1.0-h)*q1_[3]+h*q2_[3]
    norm = (q_w_**2+q_x_**2+q_y_**2+q_z_**2)**0.5
    q_w_ = q_w_ / norm
    q_x_ = q_x_ / norm
    q_y_ = q_y_ / norm
    q_z_ = q_z_ / norm
    q_interpolated = np.array([q_w_,q_x_,q_y_,q_z_])
    # pose_interpolated.printValue()
    return q_interpolated


def merge2(q1_,q2_,h_):
    h = h_
    r1 = quat2rotm(q1_)
    r2 = quat2rotm(q2_)
    r12 = (np.linalg.inv(r1)).dot(r2)
    axang12 = rotm2axang(r12)
    axang12_interpolated = axang12
    axang12_interpolated[3] *= h
    rotm = axang2rotm(axang12_interpolated)
    rotm_interpolated = rotm.dot(r1)
    quat_interpolated = rotm2quat(rotm_interpolated)
    return quat_interpolated


def merge3(q1_,q2_,h_):
    t = h_
    cos_theta = (q1_.transpose()).dot(q2_)
    theta = math.acos(cos_theta)
    print('theta',theta)
    qt = (math.sin((1-t)*theta)/math.sin(theta))*q1_+(math.sin(t*theta)/math.sin(theta))*q2_
    return qt