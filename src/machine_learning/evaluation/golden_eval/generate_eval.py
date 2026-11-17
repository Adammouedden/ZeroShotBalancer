'''
Generate an evaluation schedule for ZSB in the form of [start time s, target wheel speed, target yaw]
'''

import click
import random

@click.option('-s', '--size', help="Type in the maximum area of the testing boundary in meters squared", default=4)
@click.option('-a', '--max-acceleration', help="Actuator Modeling: maximum acceleration for our acquired motor", default=1)
@click.option('-t', '--total-time', help="Total time (in seconds) for evaluation", default=60)
def generate(size, max_acceleration, total_time, max_speed, seed):
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
    top_left = (-1 * (size ** 0.5)/2, (size ** 0.5)/2)
    top_right = ((size ** 0.5)/2, (size ** 0.5)/2)
    bottom_left = (-1 * (size ** 0.5)/2, -1 * (size ** 0.5)/2)
    bottom_right = ((size ** 0.5)/2, -1 * (size ** 0.5)/2)


    random.seed(seed)
    total_time_counter = 0
    prev_sample = [0, 0, 0]
    while total_time >= total_time_counter:
        
        prev_time = prev_sample[0]
        prev_speed = prev_sample[1]
        prev_direction = prev_sample[2]
        
        time = random.randint(1, 6) * 0.5 # seconds between 0.5 and 3 seconds
        total_time_counter += time
        delta_time = time - prev_time
        
        speed = random.uniform(-1 * max_speed, max_speed) # rads per second
        delta_speed = speed - prev_speed
        
        
        direction = random.uniform(-1 * 180, 180) # degrees
        delta_direction = (direction - prev_direction + 180) % 360 - 180 # keep it between -180 and 180
        
        #print(f"Time: {total_time_counter}, Speed: {speed}, Direction: {direction}")
        sample = [total_time_counter, speed, direction]
        print(sample)

        prev_sample = sample


size = 25 # 5x5 meters
max_acceleration = 5 # m/s^2
total_time = 60 #seconds
max_speed = 28 # rads/sec
seed = 0


generate(size, max_acceleration, total_time, max_speed, seed)