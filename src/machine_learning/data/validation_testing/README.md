# Physical-property validation

The current work targets the two-wheel self-balancer. See the [self-balancer guide](self_balancer_tests/README.md) for the implemented boundary calculator, mutation generator and frozen-PPO evaluations. The [implementation plan](self_balancer_tests/IMPLEMENTATION_PLAN.md) records the design and deferred extensions.

The self-balancer runner reuses `inverted_pendulum_tests/pretrained_policy.py` for frozen-policy loading. Earlier pendulum experiments are separate work; their measured ranges do not transfer directly to the robot.
