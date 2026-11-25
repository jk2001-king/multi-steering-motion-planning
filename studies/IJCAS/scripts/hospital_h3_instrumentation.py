"""Read-only tensor/contact serialization; not a control or verdict policy."""

import math

import numpy as np


def vector(data,size):
    if hasattr(data,"detach"):data=data.detach().cpu().numpy()
    elif hasattr(data,"numpy"):data=data.numpy()
    result=np.asarray(data,dtype=float).reshape(-1)
    if result.size!=size or not np.isfinite(result).all():raise ValueError("Invalid measured tensor vector")
    return result.tolist()


def joint_snapshot(view,time_s):
    if not math.isfinite(time_s):raise ValueError("Non-finite observer time")
    names=list(view.shared_metatype.dof_names);n=len(names)
    row={"time_s":float(time_s)}
    for key,getter in (("q_rad",view.get_dof_positions),("qd_rad_s",view.get_dof_velocities),
            ("projected_solver_joint_effort_nm",view.get_dof_projected_joint_forces),
            ("kp_tensor_units",view.get_dof_stiffnesses),("kd_tensor_units",view.get_dof_dampings),
            ("drive_force_limit_nm",view.get_dof_max_forces),("q_target_rad",view.get_dof_position_targets),
            ("qd_target_rad_s",view.get_dof_velocity_targets)):
        for name,value in zip(names,vector(getter(),n)):row[name+"__"+key]=value
    for key,value in zip(("base_x_m","base_y_m","base_z_m","base_qx","base_qy","base_qz","base_qw"),vector(view.get_root_transforms(),7)):
        row[key]=value
    return row


def contact_points(headers,data,time_s,decode_path,classify):
    """Keep original collider ordering; do NOT guess normal/impulse sign."""
    rows=[]
    for header in headers:
        collider0,collider1=str(decode_path(header.collider0)),str(decode_path(header.collider1))
        category=classify((collider0,collider1))
        if category not in ("wheel_threshold","wheel_support"):continue
        begin,count=int(header.contact_data_offset),int(header.num_contact_data)
        if count==0:continue  # CONTACT_LOST is not a force sample.
        if begin<0 or count<0 or begin+count>len(data):raise ValueError("Invalid contact-data range")
        for point in data[begin:begin+count]:
            row={"time_s":float(time_s),"category":category,"collider0":collider0,"collider1":collider1}
            for key in ("position","normal","impulse"):
                raw=getattr(point,key,None)
                values=vector(raw,3) if raw is not None else [None]*3
                for axis,value in zip("xyz",values):row[key+"_"+axis]=value
            separation=getattr(point,"separation",None)
            if separation is not None and not math.isfinite(float(separation)):raise ValueError("Invalid contact separation")
            row["separation_m"]=float(separation) if separation is not None else None
            row["complete_point_data"]=all(row[k] is not None for k in row if k.startswith(("position_","normal_","impulse_"))) and separation is not None
            rows.append(row)
    return rows
