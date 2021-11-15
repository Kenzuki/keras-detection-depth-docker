import pyrealsense2 as rs


def calculate_distance(depth: rs.depth_frame, x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2

    width = x2 - x1
    height = y2 - y1

    depth_width = depth.get_width()
    depth_height = depth.get_height()

    middle = (x1 + width / 2, y1 + height / 2)

    if depth_width < middle[0] or depth_height < middle[1]:
        print('Bounding box size out of range for depth frame')
        return 0.0

    return depth.get_distance(int(middle[0]), int(middle[1]))
