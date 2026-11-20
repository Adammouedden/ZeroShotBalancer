'''
Generate an evaluation schedule for ZSB in the form of [start time s, target wheel speed, target yaw]

TO DO:

    1. Fix the problem related to m/s speed instructions and rad/s robot instructions | HANDLED
    2. Yaw is in degrees as well, but robot accepts rad/s | HANDLED
    3. Negative speeds are never used, resolve this
    4. The last sample is truncated, set the final one with respect to the prior | HANDLED
    5. Heading is resampled independently, so there are many near instantaenous 180 degree reversals
    6. MAX_distance is relative to (0, 0), we can make this dynamic to the agent's current position
    

        
ACTUATOR MODELING NOTES:
    From robot_calibrated.xml:
        Speed range: ±20.6 rad/s, approximately ±197 RPM, from 0.115 N·m

    For 12V / 330 RPM JGB37-520 motors with the 65 mm wheels:
        Motor max is 330 RPM, approximately 34.5 rad/s
        Maximum no-load speed: ~1.12 m/s
        Max torque estimation: 0.128 N·m
        
    Under load, at current 0.95 kg:
        Cruising speed 28.27 rad/s (~0.92 m/s)
        Real world acceleration 4.5 - 5.0 m/s^2
        Time to 95% Top Speed: ~0.35 to 0.45 s
        Distance to reach top speed: ~20 to 25 cm
'''

import click
import random
import math
import json

#@click.command()
@click.option('-a', '--area', help="Type in the maximum area of the testing boundary in meters squared", default=4)
@click.option('-ms', '--max-speed', help="Actuator Modeling: maximum speed for our acquired motor", default=1)
@click.option('-t', '--total-time', help="Total time (in seconds) for evaluation", default=60)
@click.option('-wheel', '--wheel-radius', help="", default=0.0325)
@click.option('-threshold', '--boundary-threshold', help="Threshold for how close the robot should get to the walls, in meters", default=0.5)
@click.option('-rev', '--allow-reversals', help="", default=False)
def varying_time_generator(area, total_time, max_speed, seed, wheel_radius=0.0325, boundary_threshold=0):
    random.seed(seed)
    
    MAX_DISTANCE = (math.sqrt(area) / 2) - boundary_threshold
    MINIMUM_DISTANCE_FACTOR = 0.10
    TIME_GENEROSITY_FACTOR = 1.10
    SETTLE_TIME = 2
    total_time_counter = SETTLE_TIME
    
    x_position, y_position = 0, 0
    
    schedule = [[0, 0, 0]]
    
    while total_time >= total_time_counter:
        
        # Sampling distance and time
        distance_sample = random.uniform(MINIMUM_DISTANCE_FACTOR * MAX_DISTANCE, MAX_DISTANCE) 

        maximum_time_for_distance = (MAX_DISTANCE / max_speed) * TIME_GENEROSITY_FACTOR # t = d/v, ensure that we never sample a time that would make traversing this distance impossible
        minimum_time_for_distance = (distance_sample / max_speed) * TIME_GENEROSITY_FACTOR # We can give more than the exact amount of time it takes to travel distance_sample meters
    
        time = random.uniform(minimum_time_for_distance, maximum_time_for_distance)
        
        # Computing speed and constraint sampling direction
        speed = distance_sample / time
        speed /= wheel_radius # v = radius * angular speed, turn speed from m/s to rad/s, since the agent expects a target of wheel speed 
        
        # x component
        distance_to_right_wall = MAX_DISTANCE - x_position
        distance_to_left_wall = x_position - -1 * MAX_DISTANCE
        
        if distance_to_right_wall < distance_sample: # Near the right wall so move left
            sign_x = -1
        elif distance_to_left_wall < distance_sample: # Near the left wall, so move right
            sign_x = 1
        else:
            sign_x = random.choice((-1, 1)) # Nearing neither left or right wall, so randomly choose direction
        
        # y component
        distance_to_top_wall = MAX_DISTANCE - y_position
        distance_to_bottom_wall = y_position - -1 * MAX_DISTANCE  
        
        if distance_to_top_wall < distance_sample: # Near the top wall, so move down
            sign_y = -1
        elif distance_to_bottom_wall < distance_sample: # Near the bottom wall, so move up
            sign_y = 1
        else:
            sign_y = random.choice((-1, 1)) # Nearing neither top or bottom wall, so randomly choose direction
        
        quadrant_angle = random.uniform(0, math.pi / 2)
        x_component_direction = sign_x * math.cos(quadrant_angle)
        y_component_direction = sign_y * math.sin(quadrant_angle)
        
        direction = math.atan2(y_component_direction, x_component_direction)
        # direction *= (180 / math.pi) # Keep direction in radians
        
        #Update position
        x_position += distance_sample * x_component_direction # x coordinate of position
        y_position += distance_sample * y_component_direction # y coordinate of position
        
        assert abs(x_position) <= MAX_DISTANCE and abs(y_position) <= MAX_DISTANCE
        
        #Final generated sample
        sample = [round(total_time_counter, 3), round(speed, 3), round(direction, 3)]
        schedule.append(sample)
        
        #Update time counter
        total_time_counter += time
                

    schedule.append([round(total_time_counter, 3), 0, round(direction, 3)]) 
    return schedule


if __name__ == "__main__":
    size = 25 # 5x5 meters example
    total_time = 60 #seconds
    max_speed = 1 # 1 m/s comes from actuator modeling
    seed = 0
    threshold = 0 # in meters

    schedule = varying_time_generator(area=size, total_time=total_time, max_speed=max_speed, seed=seed, boundary_threshold=threshold)
    print(*schedule, sep="\n")
    
    with open("src/machine_learning/evaluation/golden_eval/schedules/test.json", "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(json.dumps(row) for row in schedule) + "\n]")