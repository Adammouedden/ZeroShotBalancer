'''
Generate an evaluation schedule for ZSB in the form of [start time s, target wheel speed, target yaw]
'''

import click

@click.option('-s', '--size', help="Type in the maximum area of the testing boundary in meters squared", default=4)
@click.option('-a', '--acceleration', help="Actuator Modeling: maximum acceleration for our acquired motor", default=1)
def generate(size, acceleration):
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
    
    pass

generate()