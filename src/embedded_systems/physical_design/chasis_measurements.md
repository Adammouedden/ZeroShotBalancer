lowest z: 0.0  highest z: 0.1897
3 separate parts found
part 0: pos="-0.0045 -0.0003 0.0949"  size="0.0600 0.0582 0.0949"
part 1: pos="0.0205 -0.0258 0.0965"  size="0.0000 0.0000 0.0035"
part 2: pos="0.0120 0.0056 0.1502"  size="0.0028 0.0029 0.0002"

height (z)   width (x)   depth (y)
0.0057       0.0900      0.0497
0.0285       0.0835      0.0615
0.0569       0.0744      0.0615
0.0854       0.0703      0.0615
0.1138       0.0700      0.0615
0.1423       0.0700      0.0615
0.1708       0.1108      0.0480
0.1841       0.0284      0.0353

<mujoco model="balance">
    <!-- Calibrated robot axes: x = wheel to wheel, y = forward, z = up. -->
    <worldbody>
        <body name="robot_body" pos="0 0 0.00602">
            <joint name="robot_body_joint" type="free"/>

            <!-- Chassis mass incl. motors, battery and electronics estimate -->
            <inertial pos="-0.000659660 -0.000078962 0.065120368"
                mass="0.825160"
                fullinertia="0.00119093 0.00216205 0.00107316 0.00000015 0.00002156 0.00001663"/>

            <!-- Robot outline divides the chassis in different boxes -->
            <geom name="col_base_plate" type="box" size="0.060 0.01825 0.005" pos="0 0.00035 0.005" density="0" rgba="0.55 0.6 0.7 1"/>
            <geom name="col_posts" type="box" size="0.035 0.03075 0.038" pos="0 0.00035 0.052" density="0" rgba="0.55 0.6 0.7 0.5"/>
            <geom name="col_mid" type="box" size="0.06215 0.04445 0.02175" pos="0.00215 -0.00165 0.09025" density="0" rgba="0.4 0.7 0.6 1"/>
            <geom name="col_brim" type="cylinder" size="0.0601 0.0035" pos="0.0006 0.00075 0.1585" density="0" rgba="0.45 0.75 0.95 1"/>

            <!-- Wheels: CAD size 70 mm x 36 mm, 60 g each is an estimate -->
            <body name="l_wheel" pos="-0.084586471 -0.000100401 0.029508788">
                <inertial pos="0 0 0" mass="0.060" diaginertia="0.00003675 0.000024855 0.000024855"/>
                <joint name="torso_l_wheel" type="hinge" pos="0 0 0" axis="-1 0 0" damping="0.01"/>
                <geom name="l_wheel_geom" size="0.035 0.018" quat="0.707107 0 0.707107 0" type="cylinder" material="wheel_material"/>
            </body>
            <body name="r_wheel" pos="0.082502957 0.000323456 0.028975666">
                <inertial pos="0 0 0" mass="0.060" diaginertia="0.00003675 0.000024855 0.000024855"/>
                <joint name="torso_r_wheel" type="hinge" pos="0 0 0" axis="1 0 0" damping="0.01"/>
                <geom name="r_wheel_geom" size="0.035 0.018" quat="0.707107 0 0.707107 0" type="cylinder" material="wheel_material"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <!-- Motors -->
        <velocity name="motor_l_wheel" joint="torso_l_wheel" ctrllimited="true" ctrlrange="-20.6 20.6" kv="4.0" forcelimited="true" forcerange="-0.115 0.115"/>
        <velocity name="motor_r_wheel" joint="torso_r_wheel" ctrllimited="true" ctrlrange="-20.6 20.6" kv="4.0" forcelimited="true" forcerange="-0.115 0.115"/>
    </actuator>
</mujoco>