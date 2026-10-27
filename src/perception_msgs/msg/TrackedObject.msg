# A single tracked object in the reference frame given by the array header.
uint32 track_id
string class_name
float32 score            # detector / fusion confidence in [0, 1]
geometry_msgs/Pose pose  # center pose of the 3D box
geometry_msgs/Vector3 size      # box extents (x, y, z) in metres
geometry_msgs/Vector3 velocity  # linear velocity (m/s) in the reference frame
float32 age_seconds      # time since the track was created
