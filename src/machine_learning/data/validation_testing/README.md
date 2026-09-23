# Data: Validation Testing

Validating the quality of MJCF environments will be critical

The first of these validation tests will be essentially smoke tests for any undefined behavior and ensuring that the values of our physical parameters remain with a realistic and useful range. These tests will include simply generating some sample files where we test individual parameters of fixed intervals, loading them onto the inverted_pendulum code, running them for a fixed amount of time, then recording results. 

These validation tests are tracked by this Jira ticket:
https://ucf-team-xx2ob1z2.atlassian.net/jira/software/projects/STR/boards/3/timeline?selectedIssue=STR-68