import cv2
import gtsam
import random
import numpy as np
import supervision as sv

from typing import Any
from ultralytics import YOLO
# from robot_class import Robot
from shapely.geometry import Polygon
# from point_cloud_utils import transform_points_2D

import time
import matplotlib
matplotlib.use("Agg")  # ✅ correct

import matplotlib.pyplot as plt

class DetectionObject:
    
    def __init__(self, classes, boxes, frame_id):
        self.data = {"xyxyxyxy": boxes}
        self.class_id = classes
        self.frame_id = frame_id
        

def load_prior(output: list) -> list:
    """Load up the prior information. This is stored as a large array, what 
    we expect from the acoustic modem

    Args:
        output (list): the prior data as an array

    Returns:
        list: a list of DetectionObjects
    """

    
    output_reshaped = np.array(output).reshape(-1,10)

    # two empty dictionaries for data
    prior_detections_dict = {}
    prior_classes_dict = {}

    # loop over each row, which is a box
    for i in range(output_reshaped.shape[0]):

        # parse each row of the output message
        row = output_reshaped[i] # grab the row
        frame_id = row[0] # the first item is the frame_id
        class_id = row[1] # the next item is the class id
        box = row[2:] # the rest is the box coords in xy xy xy xy format
        
        # if we have not logged this frame before init it
        if frame_id not in prior_detections_dict:
            prior_detections_dict[frame_id] = []
            prior_classes_dict[frame_id] = []
            
        # log each box and each class
        prior_detections_dict[frame_id].append(box)
        prior_classes_dict[frame_id].append(class_id)
        
    # now we loop over all the frame ids and store the data as a detection object
    # the detection object mimmics super vision
    prior_detection_list = []
    for i in prior_detections_dict.keys():
        boxes = np.array(prior_detections_dict[i]).reshape(-1, 4, 2)
        classes = np.array(prior_classes_dict[i])
        prior_detection_list.append(DetectionObject(classes, boxes, i))

    return prior_detection_list


def transform_points_2D(points: np.array, pose: gtsam.Pose2) -> np.array:
    """transform a set of 2D points given a pose

    Args:
        points (np.array): point cloud to be transformed
        pose (gtsam.Pose2): transformation to be applied

    Returns:
        np.array: transformed point cloud
    """

    # check if there are actually any points
    if len(points) == 0:
        return np.empty_like(points, np.float32)

    # convert the pose to matrix format
    T = pose.matrix().astype(np.float32)

    # rotate and translate to the global frame
    return points.dot(T[:2, :2].T) + T[:2, 2]


def graph_match(source_feature_vector: np.array, 
                source_angles: list, 
                source_positions: list, 
                target_feature_vectors: list, 
                target_angles: list, 
                target_positions: list) -> tuple[np.array, np.array]:
    """This function searches for matches with the prior infomation. 
    The prior information is provided as a list of feature vectors, angles and positions. Each index in 
    these lists is a single graph or robot time step. This function loops over every time step in "target" 
    and compares them to the source_feature_vector. If there is enough similarity, the graphs are compared. 
    Graph comparison yeilds a floating point score, outside of this function we can take the argmin to 
    get the best match. 

    "source" is the current robot time step we want to find a loop closure for. 
    "target" is the whole time history of prior information. 

    Args:
        source_feature_vector (np.array): feature vector for the current timestep
        source_angles (list): the angles in the current timestep
        source_positions (list): the objects in the current timestep
        target_feature_vectors (list): the whole time history of prior infomation feature vectors
        target_angles (list): the whole time history of angles
        target_positions (list): the whole time history of objects

    Returns:
        tuple[np.array, np.array]: we return both a structureed array and the raw scores. 
    """
    
    # output container
    graph_errors = []

    # loop over all the possible places
    for i, candidate in enumerate(target_feature_vectors):
    
        # scene matches must have
        # the exact same number of edges
        # similar numbers of objects
        # the exact same number of ships
        if candidate[0] == source_feature_vector[0] and abs(candidate[1] - source_feature_vector[1]) <= 1 and candidate[2] == source_feature_vector[2]:

            # parse the candidate data
            candidate_angles = list(target_angles[i])
            candidate_positions = list(pixels_to_meters(np.zeros((490, 889, 3)), list(target_positions[i]), 30.0))
            
            
            # Assign each angle in angleS to the nearest neighrbor
            angle_association_error = 0
            for angle in source_angles:
                error_temp = abs(candidate_angles - angle)
                index_temp = np.argmin(error_temp)
                error_temp = error_temp[index_temp]
                candidate_angles.pop(index_temp)
                angle_association_error += error_temp
                
            # Assign each position to the nearest neighbor
            position_association_error = 0
            if len(candidate_positions) > 0: #TODO I'm not sure this line is required anymore
                for position in source_positions:
                    error_temp = np.array(candidate_positions) - np.array(position)
                    error_temp = np.sqrt(error_temp[:,0]**2 + error_temp[:,1]**2)
                    index_temp = np.argmin(error_temp)
                    error_temp = error_temp[index_temp]
                    candidate_positions.pop(index_temp)
                    position_association_error += error_temp
                    
                    if len(candidate_positions) == 0:
                        break

            graph_errors.append([angle_association_error,position_association_error,i])


    # normalize the error scores
    if len(graph_errors) > 0:
        graph_errors = np.array(graph_errors)
        graph_errors[:,0] /= np.max(graph_errors[:,0])
        graph_errors[:,1] /= np.max(graph_errors[:,1])
        graph_errors = np.nan_to_num(graph_errors, nan=0.0)
        scores = graph_errors[:,0] + graph_errors[:,1]
        
        return graph_errors, scores
    
    else:
        return None, None

def get_angles(detections: Any, sonar_image: np.array, sonar_range: float, target_class_id: int = 0) -> list[float]:
    """Compute line orientation angles (in radians) for detections of a given class.

    Args:
        detections (Any): Object containing:
            - data['xyxyxyxy']: ndarray of shape (N, 4, 2) with quadrilateral corners.
            - class_id: array-like of shape (N,) with class IDs.
        sonar_image (float): the cartisitan sonar image, needed to convert to meters
        sonar_range (float): the max range of the sonar
        target_class_id (int, optional): Class ID to filter for. Defaults to 0.

    Returns:
        list[float]: Angles in radians, normalized to the range [0, π/2].
    """

    angle_list = []
    
    for i, quad in enumerate(detections.data['xyxyxyxy']):
        if detections.class_id[i] == target_class_id:

            # convert the box to meteric meters
            box_metric = pixels_to_meters(sonar_image, quad, sonar_range)
            long_side = long_side_corners(box_metric)
            angle = angle_from_corners(long_side)
            angle_list.append(angle)

    return np.array(angle_list)

def get_positions(detections: Any, target_class_id: int = 1) -> list[float]:
    """Get the positions for the center of detections of a given class. 

    Args:
        detections (Any): Object containing:
            - data['xyxyxyxy']: ndarray of shape (N, 4, 2) with quadrilateral corners.
            - class_id: array-like of shape (N,) with class IDs.
        target_class_id (int, optional): Class ID to filter for. Defaults to 0.

    Returns:
        list[float]: Positions in pixels. 
    """
    
    position_list = []
    
    for i, quad in enumerate(detections.data['xyxyxyxy']):
        if detections.class_id[i] == target_class_id:
            rect = cv2.minAreaRect(quad.astype(np.float32))  # ((x_center, y_center), (w, h), angle)
            (x_center, y_center), (_, _), _ = rect
            
            position_list.append([x_center, y_center])
            
    return position_list

def get_center_lines_abc_from_detections(detections, target_class_id=0) -> list:
    """
    For each detection with target class, compute the line equation (a,b,c)
    of the center line along the longest edge of the minAreaRect.
    
    Returns:
        List of (a,b,c) tuples for ax + by + c = 0 line equation.
    """

    lines_abc = []

    for i, quad in enumerate(detections.data['xyxyxyxy']):
        if detections.class_id[i] == target_class_id:
            rect = cv2.minAreaRect(quad.astype(np.float32))
            (x_center, y_center), (width, height), angle = rect

            # Normalize so width is longer edge
            if width < height:
                width, height = height, width
                angle += 90.0

            if angle < 0:
                angle += 180.0

            # Convert angle to radians
            theta = np.radians(angle)

            # Calculate two points along the longest axis centered at (x_center, y_center)
            dx = (width / 2) * np.cos(theta)
            dy = (width / 2) * np.sin(theta)

            p1 = np.array([x_center - dx, y_center - dy])
            p2 = np.array([x_center + dx, y_center + dy])

            # Calculate line coefficients a, b, c for line ax + by + c = 0
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0] * p1[1] - p1[0] * p2[1]

            lines_abc.append((a, b, c))

    return lines_abc

def compute_intersections(all_lines: list) -> np.array:
    """Compute the intersections between lines. 

    Args:
        all_lines (list): a list of lines in slope intercept format, a,b,c. 

    Returns:
        np.array: the array of intersection points. 
    """
    
    intersections = []
    for i in range(len(all_lines)):
        a1, b1, c1 = all_lines[i]
        for j in range(i + 1, len(all_lines)):
            a2, b2, c2 = all_lines[j]

            # Build matrix A and vector b
            A = np.array([[a1, b1], [a2, b2]])
            B = np.array([-c1, -c2])

            # Check for non-singular matrix (i.e., not parallel)
            if np.abs(np.linalg.det(A)) > 1e-6:
                intersection = np.linalg.solve(A, B)
                intersections.append(intersection)

    return np.array(intersections)

def proccess_image(img: np.array, map_x: np.array, map_y: np.array, model: YOLO, yolo_conf: float) -> tuple:
    """Proccess a raw sonar image with YOLO. Here we reduce the image to the bounding boxes. 
    Depending on the class type, we retain that detection as an edge (angle) or object (position)

    Args:
        img (np.array): the raw sonar image
        map_x (np.array): the mapping to convert to cartesian
        map_y (np.array): the mapping to convert to cartesian
        model (YOLO): the yolo model
        yolo_conf (float): the yolo confidence

    Returns:
        detections (sv.detections): the raw yolo result.
        intersections (list): the interections of all the lines. 
        feature_vector (list): the counts of each object class. 
        angles (list): the angle of each edge feature
        positions (list): the position of each object feature
    """

    # convert the image to something for yolo
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # call the yolo model
    results = model.predict(img, conf = yolo_conf,verbose = False)
    detections = sv.Detections.from_ultralytics(results[0])
    lines = get_center_lines_abc_from_detections(detections)
    intersections = compute_intersections(lines)
    
    # generate a feature vector
    feature_vector = np.array([len(detections.class_id[detections.class_id == 0]), 
                              len(detections.class_id[detections.class_id == 1]),
                              len(detections.class_id[detections.class_id == 2])])
    angles = get_angles(detections, img, 30)
    positions = get_positions(detections)

    
    return detections, intersections, feature_vector, angles, positions

def proccess_detections(detections: sv.Detections, fake_image: np.array, max_range: float) -> tuple:
    """Parse a series of detections in to the correct format, like in proccess_image

    Args:
        detections (sv.Detections): the detection object
        fake_image (np.array): an image that is all black, just to keep the dims
        max_range (float): max sensor range 

    Returns:
        detections (sv.detections): the raw yolo result.
        intersections (list): the interections of all the lines. 
        feature_vector (list): the counts of each object class. 
        angles (list): the angle of each edge feature
        positions (list): the position of each object feature
    """

    lines = get_center_lines_abc_from_detections(detections)
    intersections = compute_intersections(lines)
    
    # generate a feature vector
    feature_vector = np.array([len(detections.class_id[detections.class_id == 0]), 
                              len(detections.class_id[detections.class_id == 1]),
                              len(detections.class_id[detections.class_id == 2])])
    angles = get_angles(detections, fake_image, max_range)
    positions = get_positions(detections)

    
    return detections, intersections, feature_vector, angles, positions
    
def draw_boxes(img: np.array, 
               detections: sv.Detections, 
               class_names: dict, 
               oriented_box_annotator: sv.OrientedBoxAnnotator, 
               label_annotator: sv.LabelAnnotator,
               map_x: np.array,
               map_y: np.array) -> np.array:
    """This function draws the detections on the provided image. 

    Args:
        img (np.array): _description_
        detections (sv.Detections): _description_
        class_names (dict): _description_
        oriented_box_annotator (sv.OrientedBoxAnnotator): _description_
        label_annotator (sv.LabelAnnotator): _description_
        map_x (np.array): _description_
        map_y (np.array): _description_

    Returns:
        np.array: _description_
    """
    
    # convert the image to something for yolo
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Generate labels
    labels = [
        f"{class_names.get(cls, 'Unknown')}: {conf:.2f}"
        for cls, conf in zip(detections.class_id, detections.confidence)
    ]
    
    # change the sonar image to somthing easier to visulize 
    img = cv2.applyColorMap(img, 2)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # draw the boxes from yolo
    annotated_frame = oriented_box_annotator.annotate(scene=img, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return img

def create_initial_graph(angles: np.array, 
                         intersections: np.array, 
                         objects: np.array, 
                         prior_noise: gtsam.noiseModel.Diagonal.Sigmas, 
                         edge_noise: gtsam.noiseModel.Diagonal.Sigmas, 
                         point_noise: gtsam.noiseModel.Diagonal.Sigmas) -> tuple[gtsam.NonlinearFactorGraph,  gtsam.Values]:
    """This function is a subroutine inside register_graphs. This creates an initial graph we want to match to. 

    Args:
        angles (np.array): the angle features
        intersections (np.array): the intersection features
        objects (np.array): the object features
        prior_noise (gtsam.noiseModel.Diagonal.Sigmas): noise model for pose
        edge_noise (gtsam.noiseModel.Diagonal.Sigmas): noise model for edges
        point_noise (gtsam.noiseModel.Diagonal.Sigmas): noise model for point features, objects and intersections

    Returns:
        tuple[gtsam.NonlinearFactorGraph,  gtsam.Values]: returns a graph with the intitial guesses
    """

    # first we need to define a graph structure
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # 1. add the target pose as 0,0,0
    graph.addPriorPose2(gtsam.symbol("x",0), gtsam.Pose2(0,0,0), prior_noise)
    initial.insert(gtsam.symbol("x",0), gtsam.Pose2(0,0,0))

    # 2. add each edge as a rotation only factor
    for i, angle_temp in enumerate(angles):
        factor_temp = gtsam.BetweenFactorPose2(gtsam.symbol("x",0),
                                            gtsam.symbol("e",i), 
                                            gtsam.Pose2(0,0,angle_temp),
                                            edge_noise)
        graph.add(factor_temp)
        initial.insert(gtsam.symbol("e",i),gtsam.Pose2(0,0,angle_temp))
        
    # 3. add each intersection point as a landmark with a range bearing factor
    for i, position_temp in enumerate(intersections):
        
        position_temp = gtsam.Point2(position_temp)
        bearing_temp = gtsam.Pose2(0,0,0).bearing(position_temp)
        range_temp = gtsam.Pose2(0,0,0).range(position_temp)
        factor_temp = gtsam.BearingRangeFactor2D(gtsam.symbol("x",0),
                                                gtsam.symbol("l",i),
                                                bearing_temp,
                                                range_temp,
                                                point_noise)
        graph.add(factor_temp)
        initial.insert(gtsam.symbol("l",i),position_temp)

    # 4. add each object as a landmark
    for i, position_temp in enumerate(objects):

        position_temp = gtsam.Point2(position_temp)
        bearing_temp = gtsam.Pose2(0,0,0).bearing(position_temp)
        range_temp = gtsam.Pose2(0,0,0).range(position_temp)
        factor_temp = gtsam.BearingRangeFactor2D(gtsam.symbol("x",0),
                                                gtsam.symbol("o",i),
                                                bearing_temp,
                                                range_temp,
                                                point_noise)
        graph.add(factor_temp)
        initial.insert(gtsam.symbol("o",i),position_temp)

    return graph, initial

def pixels_to_meters(img: np.array, coords: np.array, max_range: float) -> np.array:
    """Convert from pixel to meter space. 

    Args:
        img (np.array): the image, we need this to pull the dims of the image
        coords (np.array): the pixel coorindates we want to convert
        max_range (float): the max range in meters of the sensor

    Returns:
        np.array: coords, converted from pixel to meters
    """

    output = []

    for value in coords:
        # we need to shift the refernce frame of the points to be the bottom center of the image
        y_val = img.shape[0] - value[1]
        x_val = value[0] - img.shape[1] / 2

        # define how many meters per pixel
        meters_per_pixel =  max_range / img.shape[0]

        # convert to meters
        y_val *= meters_per_pixel
        x_val *= meters_per_pixel

        output.append([x_val, y_val])

    return np.array(output)

def generate_object_associations(size_1: int, size_2: int) -> list:
    """Generate object level associations, randomly. 

    Args:
        size_1 (int): the larger number of objects
        size_2 (int): the smaller number of objects

    Returns:
        list: the data associations
    """
    
    temp = list(range(size_1))
    random.shuffle(temp)

    temp = temp[:size_2]

    return temp

def generate_random_data_associations(size: int, allow_null: bool = False) -> list[int]:
    """
    Generate a random mapping of size elements where each element 
    can either point to a unique index or be unassigned (-1).
    
    Args:
        size (int): Number of elements.
        allow_null
        
    Returns:
        list[int]: Randomized mapping with some elements set to -1.
    """
    
    # TODO 
    # We need to find a way to limit how high the contents can count. But we still need to cover
    # for the case where I am trying to find associations for 10 objects matching to 5
    # and 5 objects matching to 10

    # Step 1: Create a shuffled list of indices
    temp = list(range(size))
    random.shuffle(temp)

    # Step 2: Decide how many elements to mark as unassigned
    num_to_replace = random.randint(0, size)

    if allow_null:

        # Step 3: Randomly pick unique indices to replace
        indices_to_null = random.sample(range(size), num_to_replace)

        # Step 4: Replace selected indices with -1
        for i in indices_to_null:
            temp[i] = -1

    return temp

def long_side_corners(quad):
    """
    Given 4 corners of a quadrilateral, return the two corners along the longest side,
    sorted by absolute x-coordinate (distance from x=0).

    Args:
        quad (np.ndarray): shape (4,2), corners in order

    Returns:
        np.ndarray: shape (2,2), two corners along the longest side sorted by |x|
    """
    quad = np.asarray(quad)
    
    # Define the four sides as pairs of indices
    sides = [(0,1), (1,2)]
    
    # Compute lengths of all sides
    '''side_lengths = [np.linalg.norm(quad[i]-quad[j]) for i,j in sides]
    
    # Find the longest side
    idx_max = np.argmax(side_lengths)
    corner_indices = sides[idx_max]
    corners = quad[list(corner_indices)]
    
    # Sort by |x| (distance from 0 along x-axis)
    corners = corners[np.argsort(np.abs(corners[:,0]))]
    '''

    side_lengths = [np.linalg.norm(quad[i]-quad[j]) for i,j in sides]
    if side_lengths[0] > side_lengths[1]:
        return quad[:2]
    else:
        return quad[1:3]


    # return corners

def angle_from_corners(quad: list) -> float:
    """Convert a quad to the orientation of that object. 

    Args:
        quad (list): the corners of the obejct

    Returns:
        float: the orientation of the object
    """
    
    # Encode first point as Pose2 with heading 0
    pose0 = gtsam.Pose2(quad[0][0], quad[0][1], 0.0)

    # Encode second point as Point2
    p1 = gtsam.Point2(quad[1][0], quad[1][1])

    # Compute bearing from pose0 to p1
    bearing = pose0.bearing(p1)  # returns Rot2
    bearing_rad = bearing.theta()
    
    return bearing_rad


def filter_points_to_image(points: np.array, img: np.array) -> np.array:
    """Remove any points outside of the image. 

    Args:
        points (np.array): the points we want to filter
        img (np.array): the image space we want to limit the points to. 

    Returns:
        np.array: the new points
    """

    if len(points) > 0:
        h, w = img.shape[:2]
        points = points[
            (points[:,0] >= 0) & (points[:,0] < w) &
            (points[:,1] >= 0) & (points[:,1] < h)
        ]

    return points

def register_graphs(angles_target: np.array, 
                    intersections_metric_target: np.array,
                    positions_metric_target: np.array,
                    angles_source: np.array, 
                    intersections_metric_source: np.array,
                    positions_metric_source: np.array,
                    max_iterations: int = 1000, 
                    max_error: float = .36) -> tuple[gtsam.Pose2, float]:
    """This function uses RANSAC to find a 3 DoF transform between a pair of graphs. 
    This loop will run until the error dips below max_error or the max number of iterations has been hit. 
    Inside the loop we match all angles_target to angles_source using random assoications, applying the same for 
    intersections and poistions. There are two poses here, the source and target, which are both set as 0,0,0. 
    By generating random associations between the scene graphs, we solve for the pose between graphs. 
    The error function is simply the error from solving the graph. 


    Args:
        angles_target (np.array): target angles, rotation only features
        intersections_metric_target (np.array): target intersetctions, these are the intersection points between the angles
        positions_metric_target (np.array): target positions, these are the object positions
        angles_source (np.array):  source angles, rotation only features
        intersections_metric_source (np.array): source intersetctions, these are the intersection points between the angles
        positions_metric_source (np.array): source positions, these are the object positions
        max_iterations (int, optional): the max ransac iterations. Defaults to 20.
        max_error (float, optional): max error for ransac. Defaults to .36.

    Returns:
        tuple[gtsam.Pose2, float]: we return the pose and the graph error with it. 
    """

    # Define some noise models for the graph
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1])  
    edge_noise = gtsam.noiseModel.Diagonal.Sigmas([1e6, 1e6, 0.1])
    point_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1])

    # initilize the target graph graph
    graph_init, initial_init = create_initial_graph(angles_target, 
                                                    intersections_metric_target,
                                                    positions_metric_target,
                                                    prior_noise,
                                                    edge_noise,
                                                    point_noise)

    # start iterations at 0 and error very high
    error = 1000000000000000000000000000
    iterations = 0

    # Ransac loop
    while error > max_error and iterations < max_iterations: 
        iterations += 1 # increment

        # copy the initial graph
        graph = gtsam.NonlinearFactorGraph(graph_init)
        initial = gtsam.Values(initial_init)

        # 1. add an intitial guess for the source pose, in this case 0,0,0 since we have no idea
        initial.insert(gtsam.symbol("x",1), gtsam.Pose2(0,0,0))

        # 2. add each edge as a rotation only factor
        angle_data_association = generate_random_data_associations(len(angles_source)) # generate random guesses for the data assosiaction
        for i, angle_temp in enumerate(angles_source):

            # a -1 for a data association indicates we are not making a match for this angle
            if angle_data_association[i] != -1:

                # define the rotation only factor
                factor_temp = gtsam.BetweenFactorPose2(gtsam.symbol("x",1),
                                                    gtsam.symbol("e",angle_data_association[i]), 
                                                    gtsam.Pose2(0,0,angle_temp),
                                                    edge_noise)
                graph.add(factor_temp)
                if initial.exists(gtsam.symbol("e",angle_data_association[i])) == False: 
                    initial.insert(gtsam.symbol("e",angle_data_association[i]),gtsam.Pose2(0,0,angle_temp))
                

        # 3. add each intersection point as a bearing/range factor
        intersection_data_association = generate_random_data_associations(len(intersections_metric_source))  # generate random guesses for the data assosiaction
        intersection_data_association = np.array(intersection_data_association) # make it an array
        intersection_data_association[intersection_data_association > len(intersections_metric_target) - 1] = -1 # any value over the max value in target is a -1
        for i, intersection_temp in enumerate(intersections_metric_source):

            # a -1 for a data association indicates we are not making a match for this intersection
            if intersection_data_association[i] != -1:

                # get the range and bearing to target
                intersection_temp = gtsam.Point2(intersection_temp)
                bearing_temp = gtsam.Pose2(0,0,0).bearing(intersection_temp)
                range_temp = gtsam.Pose2(0,0,0).range(intersection_temp)

                # define the factor 
                factor_temp = gtsam.BearingRangeFactor2D(gtsam.symbol("x",1),
                                                        gtsam.symbol("l",intersection_data_association[i]),
                                                        bearing_temp,
                                                        range_temp,
                                                        point_noise)
                graph.add(factor_temp)
                
        # 4. add each object as a bearing/range factor
        object_data_association = None
        if min(len(positions_metric_source),len(positions_metric_target)) >= 3:
            object_data_association = generate_object_associations(max(len(positions_metric_source),
                                                                    len(positions_metric_target))-1,
                                                                    random.randint(3, min(len(positions_metric_source),
                                                                    len(positions_metric_target))))
            

            for i, object_temp in enumerate(positions_metric_source):
                if i < len(object_data_association) and object_data_association[i] != -1:

                    # get the range and bearing to target
                    object_temp = gtsam.Point2(object_temp)
                    bearing_temp = gtsam.Pose2(0,0,0).bearing(object_temp)
                    range_temp = gtsam.Pose2(0,0,0).range(object_temp)

                    # define the factor 
                    factor_temp = gtsam.BearingRangeFactor2D(gtsam.symbol("x",1),
                                                            gtsam.symbol("o",object_data_association[i]),
                                                            bearing_temp,
                                                            range_temp,
                                                            point_noise)
                    graph.add(factor_temp)
                
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
        error = graph.error(result)
        
    pose_result = result.atPose2(gtsam.symbol("x",0))
    pose_result = pose_result.between(result.atPose2(gtsam.symbol("x",1)))

    if error < max_error:
        return pose_result, error, (angle_data_association, intersection_data_association, object_data_association)
    else:
        return None, error, (None, None, None)

def polygon_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes given as 4-corner polygons.

    Args:
        box1: numpy array of shape (4,2), corners [(x1,y1), ..., (x4,y4)]
        box2: numpy array of shape (4,2), corners [(x1,y1), ..., (x4,y4)]

    Returns:
        IoU value between 0 and 1
    """
    
    # convert boxes to polygons
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    # check this polygons
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0  # invalid polygons

    # find the IOU between them
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def scene_graph_iou(source_boxes: list, target_boxes: list) -> float:
    """Evaluate the iou score between a pair of scene graphs. In this context the graphs are
    just two sets of boxes that have been registered to one another. HIGHER score is better. 

    Args:
        source_boxes (list): list of numpy arrays, see polygon_iou
        target_boxes (list): list of numpy arrays, see polygon_iou

    Returns:
        float: the iou score between the graphs
    """

    score = 0
    for i in range(len(source_boxes)):
        for j in range(len(target_boxes)):
            score += polygon_iou(source_boxes[i], target_boxes[j])

    return score

def apply_transform_to_detections(detections: sv.Detections, max_range: float, img: np.array, pose: gtsam.Pose2 = gtsam.Pose2(0,0,0)) -> tuple[list,list]:
    """Apply a transform to some yolo detections. 

    Args:
        detections (sv.Detections): the detections we want to transform
        max_range (float): the max sensor range for conversion
        img (np.array): the image for conversion
        pose (gtsam.Pose2): the transform we want to apply

    Returns:
        tuple[list,list]: we retun the boxes in metric space and with a plotting trick
    """

    # containers for output
    boxes = []
    boxes_plot = []

    # loop over each detection, pull the four corners and apply the transform
    for detection in detections.data["xyxyxyxy"]:

        # convert the detection to meters and apply the transform
        box_metric = pixels_to_meters(img, detection, max_range)
        box_metric = transform_points_2D(box_metric, pose)
        boxes.append(box_metric) # log

        # this trick makes matplotlib draw a closed box, rather then missing one side
        box_metric = np.vstack([box_metric, box_metric[0]])
        boxes_plot.append(box_metric)

    return boxes, boxes_plot

'''def merge_robot_with_prior(loop_closures: list, prior_robot: Robot, robot: Robot, calib: gtsam.Pose2) -> gtsam.gtsam.Values:
    """Merge the prior information with the robot pose chain using the detected loop closures. 

    Args:
        loop_closures (list): in the format, robot_index, prior_robot_index, pose
        prior_robot (Robot): the prior robot
        robot (Robot): the current robot
        calib (gtsam.Pose2): the calibration matrix between INS and sonar for robot

    Returns:
        gtsam.gtsam.Values: a solved graph
    """

    # define empty graph and initial guess
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # define some noise models
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.02, 0.02, 0.01])  
    loop_closure_model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[0.01, 0.01, 0.001])

    # add the prior poses as prior factors
    for i, pose in enumerate(prior_robot.bruce_lidar_poses_2D):
        pose = pose.compose(gtsam.Pose2(0.629,0,np.radians(90+15)))
        graph.addPriorPose2(gtsam.symbol("p",i), pose, prior_noise)
        initial.insert(gtsam.symbol("p",i), pose)
        
    # add the loop closures
    for (i,j,pose) in loop_closures:
        target_pose = prior_robot.bruce_lidar_poses_2D[j]
        target_pose = target_pose.compose(gtsam.Pose2(0.629,0,np.radians(90+15)))
        source_pose = target_pose.compose(calib.inverse().compose(pose.compose(calib)))
        pose_between = source_pose.between(target_pose)
        factor = gtsam.BetweenFactorPose2(gtsam.symbol("x",i), gtsam.symbol("p",j), pose_between, loop_closure_model)
        graph.add(factor)
        initial.insert(gtsam.symbol("x",i), source_pose)
        
    # add the robot pose chain
    ref_frame = source_pose.compose(robot.sonar_poses_gtsam[i].between(robot.sonar_poses_gtsam[0]))
    for i, _ in enumerate(robot.sonar_poses_gtsam[:-1]):
        pose_i = ref_frame.compose(robot.sonar_poses_gtsam[i])
        pose_i_plus_1 = ref_frame.compose(robot.sonar_poses_gtsam[i+1])
        pose_between = pose_i.between(pose_i_plus_1)
        factor = gtsam.BetweenFactorPose2(gtsam.symbol("x",i), gtsam.symbol("x",i+1), pose_between, loop_closure_model)
        graph.add(factor)
        
        if initial.exists(gtsam.symbol("x",i)) == False:
            initial.insert(gtsam.symbol("x",i), pose_i)

    if initial.exists(gtsam.symbol("x",i+1)) == False:
        initial.insert(gtsam.symbol("x",i+1), pose_i_plus_1)
        
    # optimize the graph
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    return result'''

def merge_robot_with_prior_list(loop_closures: list, prior_robot: list, robot: list, calib: gtsam.Pose2) -> gtsam.gtsam.Values:
    """Merge the prior information with the robot pose chain using the detected loop closures. 

    Args:
        loop_closures (list): in the format, robot_index, prior_robot_index, pose
        prior_robot (list): the prior robot
        robot (list): the current robot
        calib (gtsam.Pose2): the calibration matrix between INS and sonar for robot

    Returns:
        gtsam.gtsam.Values: a solved graph
    """

    # define empty graph and initial guess
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # define some noise models
    #prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.02, 0.02, 0.01])  
    #loop_closure_model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[0.01, 0.01, 0.001])
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.002, 0.002, 0.001])  
    loop_closure_model = gtsam.noiseModel.Diagonal.Sigmas([0.002, 0.002, 0.001]) 
    odom_model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[0.9, 0.9, 0.09]) # odom_model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[0.01, 0.01, 0.001])

    # add the prior poses as prior factors
    for i, pose in enumerate(prior_robot):#
        pose = pose.compose(gtsam.Pose2(0.629,0,np.radians(90+15)))
        graph.addPriorPose2(gtsam.symbol("p",i), pose, prior_noise)
        initial.insert(gtsam.symbol("p",i), pose)
        
    # add the loop closures
    for (i,j,pose) in loop_closures:
        target_pose = prior_robot[j]
        target_pose = target_pose.compose(gtsam.Pose2(0.629,0,np.radians(90+15)))
        source_pose = target_pose.compose(calib.inverse().compose(pose.compose(calib)))
        pose_between = source_pose.between(target_pose)
        factor = gtsam.BetweenFactorPose2(gtsam.symbol("x",i), gtsam.symbol("p",j), pose_between, loop_closure_model)
        graph.add(factor)
        initial.insert(gtsam.symbol("x",i), source_pose)
        
    # add the robot pose chain
    ref_frame = source_pose.compose(robot[i].between(robot[0]))
    for i, _ in enumerate(robot[:-1]):
        pose_i = ref_frame.compose(robot[i])
        pose_i_plus_1 = ref_frame.compose(robot[i+1])
        pose_between = pose_i.between(pose_i_plus_1)
        factor = gtsam.BetweenFactorPose2(gtsam.symbol("x",i), gtsam.symbol("x",i+1), pose_between, odom_model)
        graph.add(factor)
        
        if initial.exists(gtsam.symbol("x",i)) == False:
            initial.insert(gtsam.symbol("x",i), pose_i)

    if initial.exists(gtsam.symbol("x",i+1)) == False:
        initial.insert(gtsam.symbol("x",i+1), pose_i_plus_1)
        
    # optimize the graph
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    return result

def search_for_loop_closure_scene_graph(source_step: int,
                                        img: np.array, 
                                        prior_detections_list: list,
                                        prior_feature_vectors: list, 
                                        prior_angles: list, 
                                        prior_positions: list, 
                                        live_map_x: np.array,
                                        live_map_y: np.array,
                                        prior_map_x: np.array,
                                        prior_map_y: np.array,
                                        model: YOLO,
                                        yolo_conf: float,
                                        class_names: dict, 
                                        oriented_box_annotator: sv.OrientedBoxAnnotator, 
                                        label_annotator: sv.LabelAnnotator,
                                        min_iou_score: float = 1.0) -> list:
    """This is the main function that completes the loop closures search. 
    This function will pre-proccess the query image and search for a loop closure with the prior information. 
    If a match is found using graph_match, then registration is attempted. We return a list of [robot index, prior index, transform].


    Args:
        source_step (int): the step of the robot
        img (np.array): the current sonar image
        prior_detections_list (list): the prior robot detections
        prior_feature_vectors (list): the feature vectors for the prior robot
        prior_angles (list): the angle features for the prior robot
        prior_positions (list): the object features for the prior robot
        map_x (np.array): the remapping from polar to cartisitan for the sonar image
        map_y (np.array): the remapping from polar to cartisitan for the sonar image
        model (YOLO): the yolo model
        yolo_conf (float): the minimum yolo confidence 
        class_names (dict): the class names for yolo
        oriented_box_annotator (sv.OrientedBoxAnnotator): the annotator
        label_annotator (sv.LabelAnnotator): the annotator
        min_iou_score (float): the minimum iou score for outlier rejection

    Returns:
        list: the loop closure if any, in the format [robot index, prior index, transform]
    """

    start_time = time.time()
    # proccess the image with the yolo model
    (detections_source, 
    intersections_source, 
    feature_vector_source, 
    angles_source, 
    positions_source) = proccess_image(img, 
                                        live_map_x, 
                                        live_map_y, 
                                        model, 
                                        yolo_conf)
    total_time = time.time() - start_time

    with open("yolo_time.txt", "a") as file:
        file.write(f"{total_time}\n")


    # now annotate the image for sanity 
    annotated_img_source = draw_boxes(img, 
                                    detections_source, 
                                    class_names, 
                                    oriented_box_annotator, 
                                    label_annotator, 
                                    live_map_x, 
                                    live_map_y)
    
    # remove any intersections outside of the image bounds
    intersections_source = filter_points_to_image(intersections_source, annotated_img_source)
    
    # generate some possible matches
    # here first check that we actually have enough information to do anything
    # minimum two edges or an edge and at least 3 objects
    if feature_vector_source[0] >= 2 or (feature_vector_source[0] >= 1 and feature_vector_source[1] >= 3):\
        
        # find a match for the source graph
        start_time = time.time()
        graph_errors, scores = graph_match(feature_vector_source, 
                                            angles_source, 
                                            pixels_to_meters(annotated_img_source, positions_source, 30.0), 
                                            prior_feature_vectors, 
                                            prior_angles, 
                                            prior_positions)
        total_time = time.time() - start_time

        with open("match_time.txt", "a") as file:
            file.write(f"{total_time}\n")

        if graph_errors is not None:

            # what index are we matching with???
            match_index = int(graph_errors[np.argmin(scores)][2])

            # proccess the image with the yolo model
            '''(detections_target, 
            _, 
            _, 
            _, 
            _) = proccess_image(img_target,
                                prior_map_x, 
                                prior_map_y, 
                                model,
                                yolo_conf)'''

            (detections_target, 
            intersections_target, 
            feature_vector_target, 
            angles_target, 
            positions_target) = proccess_detections(prior_detections_list[match_index],
                                                    np.zeros((490, 256)),
                                                    30.0)
                        
            # perform the same check as before, we need a minimum number of items in view to attempt registration
            if feature_vector_target[0] >= 2 or (feature_vector_target[0] >= 1 and feature_vector_target[1] >= 3):

                # now annotate the image for sanity 
                '''annotated_img_target = draw_boxes(img_target,
                                                detections_target,
                                                class_names,
                                                oriented_box_annotator,
                                                label_annotator,
                                                prior_map_x, 
                                                prior_map_y)'''
                
                # remove any intersections outside of the image bounds
                intersections_target = filter_points_to_image(intersections_target, np.zeros((490, 889, 3)))

                # convert the intersection points to meters
                intersections_metric_source = pixels_to_meters(annotated_img_source, intersections_source, 30.0)
                intersections_metric_target = pixels_to_meters(np.zeros((490, 889, 3)), intersections_target, 30.0)

                # convert the centers of object to meters
                positions_metric_source = pixels_to_meters(annotated_img_source, positions_source, 30.0)
                positions_metric_target = pixels_to_meters(np.zeros((490, 889, 3)), positions_target, 30.0)

                # check if what we are about to do makes any sense.
                # if we have two edges and they don't intersect, this is not a fully constrained opitmization problem
                if feature_vector_target[0] >=2 and len(intersections_metric_target) == 0:
                    return None, annotated_img_source
                    
                if feature_vector_source[0] >=2 and len(intersections_metric_source) == 0:
                    return None, annotated_img_source
                
                # apply our registration method
                start_time = time.time()
                pose_result, reg_error, _ = register_graphs(angles_target,
                                                        intersections_metric_target,
                                                        positions_metric_target, 
                                                        angles_source, 
                                                        intersections_metric_source, 
                                                        positions_metric_source,
                                                        max_iterations = 1000,
                                                        max_error=0.36)
                total_time = time.time() - start_time
                with open("reg_time.txt", "a") as file:
                    file.write(f"{total_time}\n")
                
                print(reg_error)
                
                # check if the pose result is not none and the error is low enough
                if pose_result is not None and reg_error < 0.10:
                    
                    # convert boxes to a metric space so we can compare them
                    target_boxes, _ = apply_transform_to_detections(detections_target, 
                                                                                    30, 
                                                                                    np.zeros((490, 889, 3)))
                    source_boxes, _ = apply_transform_to_detections(detections_source, 
                                                                                    30, 
                                                                                    annotated_img_source,
                                                                                    pose_result)
                    
                    # check the IOU score to see if it is good or bad
                    iou_score = scene_graph_iou(source_boxes, target_boxes)

                    '''for box in target_boxes:
                        plt.plot(box[:,0],box[:,1],c="red")

                    for box in source_boxes:
                        plt.plot(box[:,0],box[:,1],c="blue")

                    plt.axis("equal")
                    plt.savefig("/home/jake/Desktop/workspace/src/boat_slam/animate/" + str(source_step) + ".png")
                    plt.clf()
                    plt.close()

                    print(target_boxes)'''

                    print("IOU: ", iou_score)

                    if iou_score < min_iou_score:
                        return None, annotated_img_source
                    else:
                        return [source_step, match_index, pose_result], annotated_img_source
                    
    return None, annotated_img_source


'''def search_for_loop_closure_scene_graph_temp(source_step: int,
                                        img: np.array, 
                                        prior_robot: Robot,
                                        prior_feature_vectors: list, 
                                        prior_angles: list, 
                                        prior_positions: list, 
                                        map_x: np.array,
                                        map_y: np.array,
                                        model: YOLO,
                                        yolo_conf: float,
                                        class_names: dict, 
                                        oriented_box_annotator: sv.OrientedBoxAnnotator, 
                                        label_annotator: sv.LabelAnnotator,
                                        min_iou_score: float = 0.5) -> list:
    """This is the main function that completes the loop closures search. 
    This function will pre-proccess the query image and search for a loop closure with the prior information. 
    If a match is found using graph_match, then registration is attempted. We return a list of [robot index, prior index, transform].


    Args:
        source_step (int): the step of the robot
        img (np.array): the current sonar image
        prior_robot (Robot): the prior robot
        prior_feature_vectors (list): the feature vectors for the prior robot
        prior_angles (list): the angle features for the prior robot
        prior_positions (list): the object features for the prior robot
        map_x (np.array): the remapping from polar to cartisitan for the sonar image
        map_y (np.array): the remapping from polar to cartisitan for the sonar image
        model (YOLO): the yolo model
        yolo_conf (float): the minimum yolo confidence 
        class_names (dict): the class names for yolo
        oriented_box_annotator (sv.OrientedBoxAnnotator): the annotator
        label_annotator (sv.LabelAnnotator): the annotator
        min_iou_score (float): the minimum iou score for outlier rejection

    Returns:
        list: the loop closure if any, in the format [robot index, prior index, transform]
    """

    # proccess the image with the yolo model
    (detections_source, 
    intersections_source, 
    feature_vector_source, 
    angles_source, 
    positions_source) = proccess_image(img, 
                                        map_x, 
                                        map_y, 
                                        model, 
                                        yolo_conf)

    # now annotate the image for sanity 
    annotated_img_source = draw_boxes(img, 
                                    detections_source, 
                                    class_names, 
                                    oriented_box_annotator, 
                                    label_annotator, 
                                    map_x, 
                                    map_y)
    
    # remove any intersections outside of the image bounds
    intersections_source = filter_points_to_image(intersections_source, annotated_img_source)
                    
'''

            

