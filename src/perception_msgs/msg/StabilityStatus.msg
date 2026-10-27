# Tip-over stability of the vehicle, from the IMU and the last velocity command.
std_msgs/Header header
float32 roll               # rad, estimated vehicle attitude
float32 pitch              # rad
float32 margin             # rad, tip-over margin from the measured specific force
float32 predicted_margin   # rad, margin expected for the last velocity command
float32 load_transfer      # lateral load transfer ratio, +1 when the left wheels lift
uint8 edge                 # support polygon edge closest to tipping
uint8 level                # CollisionWarning levels
float32 command_scale      # factor applied to the last command, 1 = unchanged
