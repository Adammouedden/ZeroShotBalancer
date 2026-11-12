from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
import xml.etree.ElementTree as ET

here = Path(__file__).parent
xml_folder = here / "../../machine_learning/training/robot_environments"
xml_in = xml_folder / "robot_calibrated.xml"
xml_out = xml_in

# Read the CSV
df = pd.read_csv(here / "robot_parts.csv")


# Warn about parts with no size
no_size = df[df[["sx_mm", "sy_mm", "sz_mm"]].isna().any(axis=1)]
for name in no_size["name"]:
    if not name.startswith(("Wheel", "Chassis")):
        print("warning: no size for", name, "(using point mass)")

# Shift the STL from CAD coordinates to the robot's coordinates
STL_OFFSET = np.array([4.527293, 0.323456, 0.0]) / 1000

# Chassis own inertia (kg) from its 3D model calculated using trimesh
chassis = trimesh.load(here / "Chassis.stl")
chassis.apply_scale(0.001)
chassis.apply_translation(STL_OFFSET)
chassis.density = 1 / chassis.volume
chassis_I = chassis.moment_inertia
print("chassis volume (mm^3):", round(chassis.volume * 1e9), " center (mm):", np.round(chassis.center_mass * 1000, 2))

#Calculating body mass and center of mass
mass=0
for i in range(len(df)):
    name=df["name"].iloc[i]
    if name.startswith("Wheel"):
        continue
    mass+=df["mass_g"].iloc[i]/1000

#Calculating center of mass. Using position x mass over our total mass for each coordinate
sx = sy = sz = 0
for i in range(len(df)):
    if df["name"].iloc[i].startswith("Wheel"):
        continue
    mass_i = df["mass_g"].iloc[i] / 1000
    sx += mass_i * df["x_mm"].iloc[i] / 1000
    sy += mass_i * df["y_mm"].iloc[i] / 1000
    sz += mass_i * df["z_mm"].iloc[i] / 1000

com = [sx / mass, sy / mass, sz / mass]

# Calculate the Inertia around the center of mass
rotation_x = rotation_y = rotation_z = 0
Ixy = Ixz = Iyz = 0

# Inertia of the robot body around its center of mass:
# for each part, mass x distance² + its own inertia
for i in range(len(df)):
    if df["name"].iloc[i].startswith("Wheel"):
        continue
    mass_i = df["mass_g"].iloc[i] / 1000
    dx = df["x_mm"].iloc[i] / 1000 - com[0]
    dy = df["y_mm"].iloc[i] / 1000 - com[1]
    dz = df["z_mm"].iloc[i] / 1000 - com[2]
    #Distance from the center of mass
    size_x = df["sx_mm"].fillna(0).iloc[i] / 1000
    size_y = df["sy_mm"].fillna(0).iloc[i] / 1000
    size_z = df["sz_mm"].fillna(0).iloc[i] /1000
    #parallel axis theorem to calculate the inertia of just the chasis not counting motors and battery
    rotation_x += mass_i * (dy**2 + dz**2)
    rotation_y += mass_i * (dx**2 + dz**2)
    rotation_z += mass_i * (dx**2 + dy**2)

    # Own inertia of each part the chassis uses its STL for the hollow frame and
    # the other parts use the calculations of a solid box
    if df["name"].iloc[i].startswith("Chassis"):
        rotation_x += mass_i * chassis_I[0, 0]
        rotation_y += mass_i * chassis_I[1, 1]
        rotation_z += mass_i * chassis_I[2, 2]
        Ixy += mass_i * chassis_I[0, 1]
        Ixz += mass_i * chassis_I[0, 2]
        Iyz += mass_i * chassis_I[1, 2]

    # Solid box: m * (size² + size²) / 12
    else:
        rotation_x += mass_i * (size_y**2 + size_z**2) / 12
        rotation_y += mass_i * (size_x**2 + size_z**2) / 12
        rotation_z += mass_i * (size_x**2 + size_y**2) / 12
    # Cross values from the distance
    Ixy -= mass_i * dx * dy
    Ixz -= mass_i * dx * dz
    Iyz -= mass_i * dy * dz


# Write the new mass, center of mass and inertia into the MuJoCo XML
parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
tree = ET.parse(xml_in, parser)
inertial = tree.getroot().find(".//body[@name='robot_body']/inertial")

inertial.set("pos", f"{com[0]:.9f} {com[1]:.9f} {com[2]:.9f}")
inertial.set("mass", f"{mass:.6f}")
inertial.set("fullinertia", f"{rotation_x:.8f} {rotation_y:.8f} {rotation_z:.8f} {Ixy:.8f} {Ixz:.8f} {Iyz:.8f}")

tree.write(xml_out, encoding="utf-8")
print("wrote", xml_out.resolve())



