'''
Generate an evaluation schedule for ZSB in the form of [start time s, target wheel speed, target yaw]
'''

import click
import random
import math

@click.option('-s', '--size', help="Type in the maximum area of the testing boundary in meters squared", default=4)
@click.option('-a', '--max-acceleration', help="Actuator Modeling: maximum acceleration for our acquired motor", default=1)
@click.option('-t', '--total-time', help="Total time (in seconds) for evaluation", default=60)
def generate(size, max_acceleration, total_time, max_speed, seed, boundary_threshold=1):
    '''
    
        Accept all parameters
        
        Target Speed Generation:
            Use the parameters to compute a normal distribution of possible values for target speed
            
            [start time s1, target wheel speed, target yaw]  s2 - s1 long
            [start time s2, target wheel speed, target yaw] 
            
            acceleration = (target wheel speed_1 - target_wheel_speed_2) / s2-s1
            
            
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


        Target Yaw Generation:
    '''



    random.seed(seed)
    total_time_counter = 0
    positions = []
    prev_sample = [0, 0, 0]
    schedule = [prev_sample]
    
    while total_time >= total_time_counter:
        
        time, speed, direction = prev_sample[0], prev_sample[1], prev_sample[2]
        
        distance_traveled_vector = speed * time # v = d/t -> d = v*t
        x_component_of_distance_traveled_vector = math.cos(direction) * distance_traveled_vector
        y_component_of_distance_traveled_vector = math.sin(direction) * distance_traveled_vector
        positions.append((x_component_of_distance_traveled_vector, y_component_of_distance_traveled_vector))
        
        x_component_of_current_distance_traveled = 0
        y_component_of_current_distance_traveled = 0
        
        for i in range(len(positions)):
            x_component_of_current_distance_traveled += positions[i][0]
            y_component_of_current_distance_traveled += positions[i][1]
            
        distance_constraints = check_boundary(x_component_of_current_distance_traveled, y_component_of_current_distance_traveled, size)
        for d in distance_constraints:
            if d > threshold:
                
        
                
        
        # Sampling
        time = random.randint(1, 6) * 0.5 # seconds between 0.5 and 3 seconds
        speed = random.uniform(-1 * max_speed, max_speed) # rads per second
        direction = random.uniform(-1 * 180, 180) # degrees    
        
        #print(f"Time: {total_time_counter}, Speed: {speed}, Direction: {direction}")
        sample = [total_time_counter, speed, direction]
        total_time_counter += time
        schedule.append(sample)
        
        prev_sample = sample


    print(schedule)

def check_boundary(x, y, area):
    max_x = math.sqrt(area) / 2
    min_x = -1 * max_x
    max_y = max_x
    min_y = min_x
    
    x_distance_to_right_boundary = abs(max_x - x)
    x_distance_to_left_boundary = abs(min_x - x)
    
    y_distance_to_top_boundary = abs(max_y - y)
    y_distance_to_bottom_boundary = abs(min_y - y)
    
    generation_constraints = []
    for distance in (x_distance_to_right_boundary, x_distance_to_left_boundary, y_distance_to_top_boundary, y_distance_to_bottom_boundary):
        generation_constraints.append(distance)
    
    return generation_constraints
        


if __name__ == "__main__":
    size = 25 # 5x5 meters
    max_acceleration = 5 # m/s^2
    total_time = 60 #seconds
    max_speed = 28 # rads/sec
    seed = 0


    generate(size, max_acceleration, total_time, max_speed, seed)