# Graded collision warning for one tracked object.
uint8 LEVEL_NONE=0
uint8 LEVEL_INFO=1
uint8 LEVEL_WARN=2
uint8 LEVEL_CRITICAL=3

uint32 track_id
string class_name
uint8 level
float32 distance             # closest distance to the ego footprint (m)
float32 time_to_collision    # seconds; large value if not closing
geometry_msgs/Point closest_point  # closest point on the object, ego frame
