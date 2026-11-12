# Meeting 29/10

### Test Schedule Design
- Mix smooth and sharp segments.
- Include both gentle and extreme accelerations, all physically achievable by the robot.
- Limit the space: the Gymnasium environment is effectively infinite, so cap the distance.
- Real-world metric: measure survival time only. 

### Programmatic Generation
- Write a function that generates the schedule, so it can be regenerated quickly.
- Possibly use a decorator (a wrapper that adds information or arguments to another function).
- Build it modularly, so we can just regenerate it when we get more info from the embedded team.
- Define an acceptable acceleration range.
- Proposed structure: 30 s smooth → 30 s mixed → “golden mix” of the two.

### Acceleration Limits
- No real values yet, so always check against a max acceleration.
- Area is set to 2 × 2 m for now.
- If there’s time, check acceleration based on the area. A larger area means fewer time constraints before switching to the next row.
- Action item: find out how long the physical robot takes to go from 0 to 20 (units TBD).

### Tracking & Scoring
- Command a target speed and yaw while the robot maintains balance.
- Once the target is reached, score how well it’s held by averaging over that period.
- Find out what is causing large errors.
- Penalize the robot if it doesn’t match the target within 2 s.
- Risk: we could end up penalizing it for a target that isn’t physically reachable.
- Decide how to handle the case where the target isn’t reached in time.
- Make sure the robot can never exit the boundary.

### Time Interval vs. Acceleration
- Use a fixed time interval between max accelerations.
- With a fixed interval, the robot will only ever hit max acceleration, not a range of values.
- Alternative: get different accelerations by allowing more time per segment.


## Moving Forward:
- We have calculated max acceleration and max speed of the robot
- We can leverage this to create a normal distribution 
-  

From Gemini:
### Physical Characteristics & Motor Values
* **Total Robot Mass:** 945.2 g (0.9452 kg).
* **Wheel Diameter:** 65 mm (radius $r = 0.0325\text{ m}$).
* **Max Speed:**
  * **Linear:** $1.12\text{ m/s}$ ($4.04\text{ km/h}$) no-load; $\sim 0.92\text{ m/s}$ rated under load.
  * **Angular (Wheel):** $34.56\text{ rad/s}$ ($330\text{ RPM}$) no-load; $\sim 28.27\text{ rad/s}$ rated under load.
* **Acceleration & Spin Performance:**
  * **Max Linear Acceleration:** $7.66\text{ to }9.58\text{ m/s}^2$ (theoretical motor stall limit across 2 drive wheels); practically bounded to $\sim 4.5\text{ to }5.0\text{ m/s}^2$ due to tire friction on smooth surfaces.
  * **Time to Reach Max Speed:** $\sim 0.35\text{ to }0.45\text{ s}$ under load ($\sim 0.21\text{ s}$ free-spinning).
  * **Chassis Angular Velocity (Zero-Radius Turn):** $\sim 11.23\text{ to }14.97\text{ rad/s}$ ($\sim 644^\circ\text{/s to }858^\circ\text{/s}$, depending on track width $15\text{ to }20\text{ cm}$).
* **Torque Discrepancies & Estimations:**
  * Estimations discussed between calculations were $0.115\text{ N}\cdot\text{m}$ and $0.128\text{ N}\cdot\text{m}$.
  * Note from manufacturer spec table (1:30 ratio): rated torque is $3.5\text{ kg}\cdot\text{cm}$ ($0.343\text{ N}\cdot\text{m}$), and max stall torque is $5.0\text{ kg}\cdot\text{cm}$ ($0.490\text{ N}\cdot\text{m}$).
  * **Decision:** Take a frugal approach and use the lower conservative torque estimation ($0.115\text{ to }0.128\text{ N}\cdot\text{m}$) so evaluation schedules stay physically achievable.

---

### Evaluation Metric: Wheel Speed vs. Whole Body Speed
* **Open Discussion Question:** Determine whether to score/reward individual wheel speeds or the whole body speed.
  * *Observation:* Left and right wheel speeds are currently direct observation/action inputs.
  * *Body Speed Pros:* Tracking the actual body speed is acknowledged as more accurate.
  * *Body Speed Cons:* Potentially more convoluted/complicated to isolate without reward hacking.

---

### Normal Distribution Construction
* Since the robot accelerates to max speed in under $0.5\text{ s}$, fixed-time intervals (e.g., $2\text{ s}$ to $4\text{ s}$) do not choke or restrict the reachable acceleration distribution.
* Target speed and yaw generation will sample from a normal distribution bounded by physically achievable limits.
* **Decision:** Test both methods (fixed duration intervals vs. dynamically scaled durations based on speed delta).
* Incorporate target yaw/angular velocity into the distribution sampling alongside linear speed.

---

### Action Items & Next Steps
* **Milestone:** Finish the "Golden Evaluation" schedule generator by next Friday to resume model training.
* **PR Policy:** All constants and limits derived from LLM calculations must include the exact prompt trail in the PR to ensure reproducibility and transparency.

---

### Prompts Used to Generate Values (with [Amazon page](https://www.amazon.com/dp/B0CKP1J1X2?th=1) as context)
1. If I ordered this wheel, what is the speed range and acceleration I'll have available? 
2. Whats the max speed to reach that max speed under no load
3. 945.2g is the weight of the entire robot right now
4. convert speed to rad/s
5. Max angular velocity and acceleration
6. Is 0.128 N x m a max torque estimation?