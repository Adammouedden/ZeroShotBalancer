# Report on Creating MJCF Files From Scratch

September 25, 2026

## The bigger picture

An MJCF file describes **what exists in a MuJoCo simulation and how it can move**. For our project, that means the ground, the robot's chassis and wheels, their masses and shapes, the wheel axles, the motors, and any simulated sensors. MuJoCo reads the XML file and **compiles** it into a model (`MjModel`). A simulation then keeps a changing state (`MjData`): positions, velocities, contacts, sensor readings, and controls. Our Gymnasium environment will reset and step that simulation, pass observations to a controller or RL policy, and calculate rewards and termination conditions. Those tasks do not all belong in MJCF.

One way to think of the bigger picture: **bodies form a tree; joints say what can move; geoms give bodies physical shapes; actuators apply control; sensors yield measurements.** The XML is the physical starting point for our simulation. [1–3]

| MJCF element | What it means | Example for our robot |
| --- | --- | --- |
| `<mujoco>` | Root of the model file | The entire robot scene |
| `<worldbody>` | Fixed world frame and root of the body tree | Ground plane and top-level robot body |
| `<body>` | A physical frame that may have mass and children | Chassis, left wheel, right wheel |
| `<joint>` / `<freejoint>` | Degrees of freedom between a body and its parent | Floating chassis; a rotating axle per wheel |
| `<geom>` | Shape used for collision, appearance, and often mass/inertia inference | Floor, chassis box, wheel cylinders |
| `<site>` | Named point/frame attached to a body | IMU mounting location |
| `<actuator>` | Controls that produce forces through joints or other transmissions | A motor on each wheel axle |
| `<sensor>` | Named measurements calculated during simulation | Gyroscope, accelerometer, wheel speeds |
| `<option>` | Simulation settings | Gravity and physics timestep |

## What is required to start?

MJCF is XML: elements have opening/closing tags or a self-closing `/>`, and attributes (e.g. `pos="0 0 0"`) live inside those tags. A small, **useful** simulation needs a `<mujoco>` root, a `<worldbody>`, and some bodies/geoms to simulate. A moving robot additionally needs joints, valid mass and inertia, and actuators if it must drive. Many XML sections are optional; you do not need defaults classes, materials, meshes, or a custom actuator in order to start seeing things happen. MuJoCo can infer a body's mass properties from its geoms when no `<inertial>` element is given; the model below assigns a mass to each robot geom. [1, 2]

Here is an illustrative starting point for our **two-wheeled, self-balancing robot**. The model compiles, but whether it balances depends on a controller and on realistic physical parameters.

```xml
<mujoco model="balancing_starter">
  <compiler angle="degree"/>]
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"
          friction="1 0.005 0.0001"/>

    <body name="chassis" pos="0 0 0.04">
      <freejoint name="root"/>
      <geom name="chassis_shape" type="box" pos="0 0 0.10"
            size="0.08 0.07 0.10" mass="0.8"/>
      <site name="imu" pos="0 0 0.12" size="0.005"/>

      <body name="left_wheel" pos="0 0.10 0">
        <joint name="left_axle" type="hinge" axis="0 1 0"
               damping="0.001"/>
        <geom name="left_tire" type="cylinder" size="0.04 0.015"
              euler="90 0 0" mass="0.1"
              friction="1 0.005 0.0001"/>
      </body>

      <body name="right_wheel" pos="0 -0.10 0">
        <joint name="right_axle" type="hinge" axis="0 1 0"
               damping="0.001"/>
        <geom name="right_tire" type="cylinder" size="0.04 0.015"
              euler="90 0 0" mass="0.1"
              friction="1 0.005 0.0001"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_drive" joint="left_axle" gear="1"
           ctrllimited="true" ctrlrange="-0.2 0.2"/>
    <motor name="right_drive" joint="right_axle" gear="1"
           ctrllimited="true" ctrlrange="-0.2 0.2"/>
  </actuator>

  <sensor>
    <gyro name="angular_rate" site="imu"/>
    <accelerometer name="linear_accel" site="imu"/>
    <jointvel name="left_speed" joint="left_axle"/>
    <jointvel name="right_speed" joint="right_axle"/>
  </sensor>
</mujoco>
```

### Read it from the outside in

1. **Scene:** `<worldbody>` contains a fixed floor and a chassis. The world is fixed; the chassis has a `<freejoint>`, so it can translate and rotate in 3D and can fall over. If we omitted that joint, the chassis would be welded to the world and could not balance or fall.
2. **Geometry and mass:** The chassis box has a mass of 0.8 kg. The wheel cylinders have a radius of 0.04 m, half-width of 0.015 m, and mass of 0.1 kg each. The box `size` values are *half-lengths*, not full dimensions. MuJoCo uses these geoms to infer inertial properties because there are no explicit `<inertial>` tags. [2]
3. **Wheel motion:** Each wheel is a child of the chassis and has its own hinge joint. Its local hinge axis is `0 1 0`, along the axle. `euler="90 0 0"` rotates the cylinder's long axis onto that direction. The two wheel bodies sit at opposite Y offsets from the chassis frame. The wheel radius and the chassis's initial height place the tires against the floor.
4. **Control:** Each `<motor>` references a *named joint*. With `gear="1"`, these are idealized direct-drive torque controls, limited here to −0.2 to +0.2 N·m. Those example limits need to be replaced with an actuator model grounded in our motor, gearing, and firmware. A motor does not create a wheel joint; it acts on a joint that already exists. [2]
5. **Measurements:** The `<site>` gives the IMU a mounting frame on the chassis. The gyro and accelerometer measure at that site; `jointvel` provides the axle speeds. The IMU outputs three values each, and the wheel-speed sensors one each. These measurements can feed the Gymnasium observation, after we decide what information the real robot will actually have. The accelerometer reports acceleration **including gravity** in its local frame; it is not a ready-made pitch-angle sensor. [2]

Positions are generally **local**: the chassis `pos` is relative to the world, its geom and IMU site positions are relative to the chassis, and wheel-body positions are relative to the chassis. The orientation of each wheel geom is local to its wheel. MuJoCo's convention is Z-up. Here distances are in meters, masses in kilograms, time in seconds, and the specified `euler` angle in degrees because of `<compiler angle="degree"/>`; compiled angles are represented in radians. [1, 2]

## How to build the real MJCF, in order

1. **Write down the physical frames and measurements.** Pick a forward direction, an axle direction, an upright direction, and a chassis origin (for example, the midpoint between the wheel centers). Measure wheel radius and spacing, chassis and wheel masses, center of mass, IMU position, and motor limits. Record which quantities are measured and which remain estimates.
2. **Start with the tree and primitive shapes.** Put the floor in `<worldbody>`. Give the chassis a free joint and a box or capsule geom. Add a wheel body with a hinge and cylinder geom on each side. Make the names unique and descriptive. Primitive shapes are enough to test dynamics before introducing visual meshes.
3. **Check frames and inertia.** Verify that the wheels touch the floor at the initial pose, both axle axes point the intended way, and the robot can tip. Inspect the robot from the front and side in the viewer. Start with geom-based mass inference; use explicit `<inertial>` values later if measured centers of mass and moments of inertia demand it. Adding `<inertial>` changes how inertia is specified: it supplies the body properties instead of relying on automatic inference from that body's geoms. [2]
4. **Attach motors and sensors.** Connect one motor to each named wheel joint. Begin with a simple, bounded torque input; only then consider more faithful motor behavior, actuator delay, encoders, or IMU noise. Keep the controller in our Gymnasium/Python code so we can test different policies against the same MJCF.
5. **Compile and run short physical checks.** Load the XML in MuJoCo to catch missing attributes, invalid references, or unrealistic inertia. Observe the robot with no control, then apply small equal wheel commands to check travel direction and opposite commands to check turning. Confirm contacts, wheel speeds, and whether the chassis tips as expected. Correct the model before starting RL training.

With MuJoCo's Python package installed, the standalone viewer can open a saved XML file with:

```text
python -m mujoco.viewer --mjcf=/path/to/balancing_starter.xml
```

Loading the file with `mujoco.MjModel.from_xml_path(...)` in Python also compiles it and reports errors. The example above was compiled and stepped with MuJoCo 3.14.0: it produced a model with **9 position coordinates, 8 velocity coordinates, 2 actuator controls, and 8 sensor values**. The position/velocity difference is expected because the chassis free joint represents orientation with a quaternion in `qpos`. A successful compile confirms valid model structure, **not** realistic robot behavior. [2, 3]

## What changes for our experiments?

Our pitch proposes testing generalization across friction, gravity, drag or push disturbances, payload shifts, and motor degradation. [4] The first MJCF should establish one nominal robot. Later, the Gymnasium environment can change permitted physical parameters at reset and run separate evaluation conditions. We should keep a record of the nominal values, the training ranges, and the held-out test ranges so the zero-shot claim is meaningful. Friction belongs to contacting geoms; masses and centers of mass belong to the robot's physical model; applied pushes and episode logic typically belong to simulation code. Aerodynamic effects require a deliberate MuJoCo fluid-force model or disturbance approximation, rather than a generic `drag` XML tag. [1, 2]

The XML describes a *simulated* sensor and actuator. It does not guarantee that the policy sees the same signals or experiences the same latency, motor saturation, or friction as the ESP32-based robot. For transfer to hardware, compare its observation list, update rate, axis conventions, control limits, and initial conditions with the firmware team's actual design. Calibrate and document the differences before treating simulation success as evidence of real-world generalization.

## A note on defaults and inheritance

The `<default>` / `class` / `childclass` example in the MuJoCo modeling guide is useful for understanding shared settings: an explicitly set attribute wins; otherwise an element uses the active class, which can be inherited through an ancestor body's `childclass`. The four colors in that example are correct. **The documentation explicitly says the color-only example does not compile**, because required geometry information is missing. We can introduce defaults after we have a working robot model, for example to share wheel-friction settings or visual colors. They shorten repeated XML; they do not supply the missing physical design. [1]

## References

1. MuJoCo Documentation, [Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html): kinematic tree, defaults, frames, actuators, and sensors.
2. MuJoCo Documentation, [XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html): element attributes and physical interpretation.
3. MuJoCo Documentation, [Python](https://mujoco.readthedocs.io/en/stable/python.html): loading XML and using the viewer.
4. *Sim-to-Real Generalization for an Embodied Balancing Robot*, team project pitch (provided with this report).
