# The particles of a particle filter: one pose per particle and its weight (how likely the particle
# is, given everything measured so far). The weights add up to 1.
std_msgs/Header header
geometry_msgs/Pose[] poses
float64[] weights
