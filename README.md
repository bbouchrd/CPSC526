# CPSC526

To use, move directories under models/ into your .gazebo/models directory.

In the catkin directory: 
1. source devel/setup.sh - this is so that ros commands will look for necessary files in src/
2. roscore - starts a rosmaster instance
3. python run.py (comment and uncomment as necessary to switch to and from code for standing and sitting; sitting doesn't utilize the actor-critic model that's set up)
