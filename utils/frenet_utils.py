import numpy as np
from numba import njit

def get_closest_point_vectorized(point, array):
    """
    Find ID of the closest point from point to array.
    Using euclidian norm.
    Works in nd.
    :param point: np.array([x, y, z, ...])
    :param array: np.array([[x1, y1, z1, ...], [x2, y2, z2, ...], [x3, y3, z3, ...], ...])
    :return: id of the closest point
    """

    min_id = np.argmin(np.sum(np.square(array - point), 1))

    return min_id

def determine_side(a, b, p):
    """ Determines, if car is on right side of trajectory or on left side
    Arguments:
         a - point of trajectory, which is nearest to the car, geometry_msgs.msg/Point
         b - next trajectory point, geometry_msgs.msg/Point
         p - actual position of car, geometry_msgs.msg/Point

    Returns:
         1 if car is on left side of trajectory
         -1 if car is on right side of trajectory
         0 if car is on trajectory
    """
    side = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if side > 0:
        return -1
    elif side < 0:
        return 1
    else:
        return 0
    
def get_rotation_matrix_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

def find_center_of_arc(point, radius, direction):
    """
    :param point: Point on the arc. np.array([x, y])
    :param radius: Radius of arc. - -> arc is going to the right, + -> arc is going to the left.
    :param direction: direction which way the arc continues from the point (angle 0-2pi)
    :return: center: np.array([x, y])
    """
    R = get_rotation_matrix_2d(direction + np.pi / 2.0 * np.sign(radius))
    C = np.squeeze(point + (R @ np.array([[abs(radius)], [0.0]])).T)
    return C
    
def find_arc_end(start_point, radius, start_angle, arc_angle):
    # print(f"start angle: {start_angle}")
    angle = start_angle + arc_angle * np.sign(radius)
    # print(f"angle: {start_angle + np.pi/2.0 * np.sign(radius)}")
    C = find_center_of_arc(start_point, radius, start_angle + np.pi / 2.0 * np.sign(radius))
    arc_end_point = C + abs(radius) * np.array([np.cos(angle), np.sin(angle)])
    return arc_end_point     

# @njit(cache=True)
def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

# @njit(cache=True)
def find_curvature(point, trajectory):
    assert point in trajectory
    for i in range(trajectory.shape[0]):
        if np.all(trajectory[i, :] == point):
            point_index = i
    index_previous, index_next = (point_index - 1) % trajectory.shape[0], (point_index + 1) % trajectory.shape[0]
    point_previous, point_next = trajectory[index_previous, :], trajectory[index_next, :]
    (cx, cy), radius = define_circle(point_previous, point, point_next)
    curvature = 1 / radius
    sign = np.sign((cx - point_previous[0]) * (cy - point_next[1]) - (cx - point_next[0]) * (cy - point_previous[1]))
    # if sign > 0:
    #     tan_rad = np.arctan((cy - point[1]) / (cx - point[0])) + np.pi / 2
    # else:
    #     tan_rad = np.arctan((cy - point[1]) / (cx - point[0])) + 3 * np.pi / 2 
    return sign * curvature

def frenet_to_cartesian(pose, trajectory):
    """
    :param pose: [s, ey, eyaw]
    :return:
    """
    # trajectory ... s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    diff = trajectory[:, 0] - pose[0]

    # print(diff)

    segment_id = np.argmax(diff[diff <= 0])  # should be always id of the point that has smaller s than the point

    # print(segment_id)
    # print(trajectory[segment_id, :])

    if trajectory[segment_id, 4] == 0:
        # line
        yaw = np.mod(trajectory[segment_id, 3] + pose[2], 2.0 * np.pi)
        s_reminder = pose[0] - trajectory[segment_id, 0]
        R1 = get_rotation_matrix_2d(trajectory[segment_id, 3])
        R2 = get_rotation_matrix_2d(trajectory[segment_id, 3] + np.pi / 2.0 * np.sign(pose[1]))
        position = (trajectory[segment_id, 1:3] + (R1 @ np.array([[abs(s_reminder)], [0.0]])).T + (
                R2 @ np.array([[abs(pose[1])], [0.0]])).T).squeeze()
    else:
        # circle
        center = find_center_of_arc(trajectory[segment_id, 1:3],
                                            1.0 / trajectory[segment_id, 4],
                                            trajectory[segment_id, 3])

        s_reminder = pose[0] - trajectory[segment_id, 0]

        start_angle = np.mod(trajectory[segment_id, 3] - np.pi / 2.0 * np.sign(trajectory[segment_id, 4]), 2 * np.pi)
        arc_angle = s_reminder / abs(1.0 / trajectory[segment_id, 4])

        trajectory_point = find_arc_end(trajectory[segment_id, 1:3],
                                                1.0 / trajectory[segment_id, 4],
                                                start_angle,
                                                arc_angle)
        vector = trajectory_point - center

        position = trajectory_point + vector / np.linalg.norm(vector) * pose[1] * (-1) * np.sign(trajectory[segment_id, 4])
        yaw = np.mod(np.arctan2(vector[1], vector[0]) + np.pi / 2.0 * np.sign(trajectory[segment_id, 4]) + pose[2], 2 * np.pi)

    return np.array([position[0], position[1], yaw])


@njit(fastmath=True, cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,), dtype=np.float32)
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],), dtype=np.float32)
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=True, cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :] + np.float32(1e-4)
        end = trajectory[i + 1, :] + np.float32(1e-4)
        V = np.ascontiguousarray(end - start).astype(np.float32)

        a = np.dot(V, V)
        b = np.array(2.0, dtype=np.float32) * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - 2.0 * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = (end - start).astype(np.float32)

            a = np.dot(V, V)
            b = np.array(2.0, dtype=np.float32) * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - 2.0 * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(cache=True)
def simple_norm_axis1(vector):
    return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2)


@njit(cache=True)
def get_wp_xyv_with_interp(L, curr_pos, theta, waypoints, wpNum, interpScale):
    traj_distances = simple_norm_axis1(waypoints[:, :2] - curr_pos)
    nearest_idx = np.argmin(traj_distances)
    nearest_dist = traj_distances[nearest_idx]
    segment_end = nearest_idx
    # count = 0
    if wpNum < 100 and traj_distances[wpNum - 1] < L:
        segment_end = wpNum - 1
    #     # print(traj_distances[-1])
    else:
        while traj_distances[segment_end] < L:
            segment_end = (segment_end + 1) % wpNum
    #     count += 1
    #     if count > wpNum:
    #         segment_end = wpNum - 1
    #         break
    segment_begin = (segment_end - 1 + wpNum) % wpNum
    x_array = np.linspace(
        waypoints[segment_begin, 0], waypoints[segment_end, 0], interpScale
    )
    y_array = np.linspace(
        waypoints[segment_begin, 1], waypoints[segment_end, 1], interpScale
    )
    v_array = np.linspace(
        waypoints[segment_begin, 2], waypoints[segment_end, 2], interpScale
    )
    xy_interp = np.vstack((x_array, y_array)).T
    dist_interp = simple_norm_axis1(xy_interp - curr_pos) - L
    i_interp = np.argmin(np.abs(dist_interp))
    target_global = np.array((x_array[i_interp], y_array[i_interp]))
    new_L = np.linalg.norm(curr_pos - target_global)
    return (
        np.array((x_array[i_interp], y_array[i_interp], v_array[i_interp])),
        new_L,
        nearest_dist,
    )


@njit(fastmath=True, cache=True)
def cartesian_to_frenet(point: np.ndarray, trajectory: np.ndarray) -> np.ndarray:

    a, b, c, i = nearest_point(point, trajectory)
    ia, ib = (i - 1) % trajectory.shape[0], (i + 1) % trajectory.shape[0]

    # calculate s
    point_a, point_b = trajectory[ia, :], trajectory[ib, :]
    t = (point[0] - point_a[0]) * (point_b[0] - point_a[0]) + (
        point[1] - point_a[1]
    ) * (point_b[1] - point_a[1])
    t /= (point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2

    if i > 0:
        # to avoid issue when closest wp i indexed as 0
        diffs = trajectory[1:i, :] - trajectory[: i - 1, :]
        s = sum(np.sqrt(np.sum(diffs**2, -1)))
    else:
        s = 0.0

    s += t * np.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)

    d = distance_point_to_line(point, trajectory[ia], trajectory[ib])

    return np.array((s, d))


@njit(fastmath=False, cache=True)
def distance_point_to_line(
    point: np.ndarray, point_a: np.ndarray, point_b: np.ndarray
) -> float:
    """
    compute distance of point to line passing through a, b

    ref: http://paulbourke.net/geometry/pointlineplane/

    :param point: point from which we want to compute distance to line
    :param point_a: first point in line
    :param point_b: second point in line
    :return: distance
    """

    px, py = (
        point_b[0] - point_a[0],
        point_b[1] - point_a[1],
    )
    norm = px**2 + py**2

    u = ((point[0] - point_a[0]) * px + (point[1] - point_a[1]) * py) / norm
    u = min(max(u, 0), 1)
    sign = px * (point[1] - point_a[1]) - py * (point[0] - point_a[0]) > 0

    xl = point_a[0] + u * px
    yl = point_a[1] + u * py

    dx = xl - point[0]
    dy = yl - point[1]

    d = np.sqrt(dx**2 + dy**2)
    d = d if sign else -d

    return d
