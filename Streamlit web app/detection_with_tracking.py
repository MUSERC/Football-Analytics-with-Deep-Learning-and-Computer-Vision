# Import libraries
import numpy as np
import pandas as pd
import streamlit as st

import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import os
import json
import yaml
import time
from byte_tracker import ByteTracker

def get_labels_dics():
    # Get tactical map keypoints positions dictionary
    json_path = "../pitch map labels position.json"
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)

    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "../config pitch dataset.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    classes_names_dic = classes_names_dic['names']

    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "../config players dataset.yaml"
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    return keypoints_map_pos, classes_names_dic, labels_dic

def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

    colors_dic = {
        team1_name:[team1_p_color_rgb, team1_gk_color_rgb],
        team2_name:[team2_p_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name]+colors_dic[team2_name] # Define color list to be used for detected player team prediction
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Converting color_list to L*a*b* space
    return colors_dic, color_list_lab

def generate_file_name():
    list_video_files = os.listdir('./outputs/')
    idx = 0
    while True:
        idx +=1
        output_file_name = f'detect_{idx}'
        if output_file_name+'.mp4' not in list_video_files:
            break
    return output_file_name

def get_player_color_palette(frame_rgb, bbox, num_pal_colors):
    """Extract color palette for a player bounding box"""
    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
    center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
    center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
    center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
    center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
    center_filter = obj_img[center_filter_y1:center_filter_y2, 
                    center_filter_x1:center_filter_x2]
    obj_pil_img = Image.fromarray(np.uint8(center_filter))
    reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)
    palette = reduced.getpalette()
    palette = [palette[3*n:3*n+3] for n in range(256)]
    color_count = [(n, palette[m]) for n,m in reduced.getcolors()]
    RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(
                        by = 'cnt', ascending = False).iloc[0:num_pal_colors,:]
    palette = list(RGB_df.RGB)
    return palette

def predict_player_team(palette, color_list_lab, colors_dic):
    """Predict player team based on color palette"""
    palette_distance = []
    palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]
    
    for color in palette_lab:
        distance_list = []
        for c in color_list_lab:
            distance = skimage.color.deltaE_cie76(color, c)
            distance_list.append(distance)
        palette_distance.append(distance_list)
    
    vote_list = []
    nbr_team_colors = len(list(colors_dic.values())[0])
    for dist_list in palette_distance:
        team_idx = dist_list.index(min(dist_list))//nbr_team_colors
        vote_list.append(team_idx)
    
    return max(vote_list, key=vote_list.count)

def detect_with_tracking(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
            hyper_params, ball_track_hyperparams, plot_hyperparams, num_pal_colors, colors_dic, color_list_lab):

    show_k = plot_hyperparams[0]
    show_pal = plot_hyperparams[1]
    show_b = plot_hyperparams[2]
    show_p = plot_hyperparams[3]

    p_conf = hyper_params[0]
    k_conf = hyper_params[1]
    k_d_tol = hyper_params[2]

    nbr_frames_no_ball_thresh = ball_track_hyperparams[0]
    ball_track_dist_thresh = ball_track_hyperparams[1]
    max_track_length = ball_track_hyperparams[2]

    if (output_file_name is not None) and (len(output_file_name)==0):
        output_file_name = generate_file_name()

    # Initialize ByteTracker for player tracking
    player_tracker = ByteTracker(frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    
    # Player tracking history: {track_id: {'positions': [], 'team': team_name, 'color': color}}
    player_tracks = {}

    # Read tactical map image
    tac_map = cv2.imread('../tactical map.jpg')
    tac_width = tac_map.shape[0]
    tac_height = tac_map.shape[1]
    
    # Create output video writer
    if save_output:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + tac_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + tac_height
        output = cv2.VideoWriter(f'./outputs/{output_file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    # Create progress bar
    tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st_prog_bar = st.progress(0, text='Detection starting.')

    keypoints_map_pos, classes_names_dic, labels_dic = get_labels_dics()

    # Set variable to record the time when we processed last frame 
    prev_frame_time = 0
    # Set variable to record the time at which we processed current frame 
    new_frame_time = 0
    
    # Store the ball track history
    ball_track_history = {'src':[], 'dst':[]}
    nbr_frames_no_ball = 0

    # Loop over input video frames
    for frame_nbr in range(1, tot_nbr_frames+1):

        # Update progress bar
        percent_complete = int(frame_nbr/(tot_nbr_frames)*100)
        st_prog_bar.progress(percent_complete, text=f"Detection in progress ({percent_complete}%)")

        # Read a frame from the video
        success, frame = cap.read()

        # Reset tactical map image for each new frame
        tac_map_copy = tac_map.copy()

        if nbr_frames_no_ball>nbr_frames_no_ball_thresh:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

        if success:

            #################### Part 1 ####################
            # Object Detection & Coordinate Transformation #
            ################################################

            # Run YOLOv8 players inference on the frame
            results_players = model_players(frame, conf=p_conf)
            # Run YOLOv8 field keypoints inference on the frame
            results_keypoints = model_keypoints(frame, conf=k_conf)
            
            ## Extract detections information
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()                          
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()                        
            labels_p = list(results_players[0].boxes.cls.cpu().numpy())                     
            confs_p = list(results_players[0].boxes.conf.cpu().numpy())                     
            
            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()                        
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()                      
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())                   

            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extract detected field keypoints coordinates on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])

            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                if frame_nbr > 1:
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)
                        update_homography = coor_error > k_d_tol
                    else:
                        update_homography = True
                else:
                    update_homography = True

                if update_homography:
                    homog, mask = cv2.findHomography(detected_labels_src_pts, detected_labels_dst_pts)

            if 'homog' in locals():
                detected_labels_prev = detected_labels.copy()
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()

                #################### Part 2 #####################
                # Player Tracking with ByteTrack               #
                #################################################
                
                # Prepare detections for ByteTracker (only players - label 0)
                player_detections = []
                player_indices = []
                for i, label in enumerate(labels_p):
                    if int(label) == 0:  # Player detection
                        x1, y1, x2, y2 = bboxes_p[i]
                        w = x2 - x1
                        h = y2 - y1
                        # Format: [x1, y1, w, h, confidence]
                        player_detections.append([x1, y1, w, h, confs_p[i]])
                        player_indices.append(i)
                
                if len(player_detections) > 0:
                    player_detections = np.array(player_detections)
                    # Update tracker
                    tracked_objects = player_tracker.update(player_detections)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process tracked players
                    for track in tracked_objects:
                        track_id = track.track_id
                        bbox = track.tlbr  # [x1, y1, x2, y2]
                        score = track.score
                        
                        # Calculate player position for tactical map (center bottom of bbox)
                        player_center_x = (bbox[0] + bbox[2]) / 2
                        player_bottom_y = bbox[3]
                        player_pos_src = np.array([player_center_x, player_bottom_y])
                        
                        # Transform to tactical map coordinates
                        pt = np.append(player_pos_src, np.array([1]), axis=0)
                        dest_point = np.matmul(homog, np.transpose(pt))
                        dest_point = dest_point/dest_point[2]
                        player_pos_dst = np.transpose(dest_point)[:2]
                        
                        # Initialize or update player track data
                        if track_id not in player_tracks:
                            # Extract color palette for new track
                            palette = get_player_color_palette(frame_rgb, bbox, num_pal_colors)
                            team_prediction = predict_player_team(palette, color_list_lab, colors_dic)
                            team_name = list(colors_dic.keys())[team_prediction]
                            team_color = colors_dic[team_name][0]
                            
                            player_tracks[track_id] = {
                                'positions': [],
                                'team': team_name,
                                'color': team_color,
                                'palette': palette,
                                'last_seen': frame_nbr
                            }
                        
                        # Update track position and last seen frame
                        player_tracks[track_id]['positions'].append({
                            'frame': frame_nbr,
                            'src_pos': player_pos_src,
                            'dst_pos': player_pos_dst,
                            'bbox': bbox,
                            'score': score
                        })
                        player_tracks[track_id]['last_seen'] = frame_nbr
                        
                        # Keep only recent positions (last 30 frames for trails)
                        if len(player_tracks[track_id]['positions']) > 30:
                            player_tracks[track_id]['positions'].pop(0)

                # Handle ball detection and tracking
                bboxes_p_c_2 = bboxes_p_c[[i==2 for i in labels_p],:]
                detected_ball_src_pos = bboxes_p_c_2[0,:2] if bboxes_p_c_2.shape[0]>0 else None

                if detected_ball_src_pos is None:
                    nbr_frames_no_ball+=1
                else: 
                    nbr_frames_no_ball=0

                # Transform ball coordinates
                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point/dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)

                    # Track ball history
                    if show_b:
                        if len(ball_track_history['src'])>0 :
                            if np.linalg.norm(detected_ball_src_pos-ball_track_history['src'][-1])<ball_track_dist_thresh:
                                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                            else:
                                ball_track_history['src']=[(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                                ball_track_history['dst']=[(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                        else:
                            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                
                if len(ball_track_history) > max_track_length:
                    ball_track_history['src'].pop(0)
                    ball_track_history['dst'].pop(0)

            #################### Part 3 #####################
            # Updated Frame & Tactical Map With Annotations #
            #################################################
            
            ball_color_bgr = (0,0,255)
            palette_box_size = 10
            annotated_frame = frame

            # Annotate tracked players
            for track_id, track_data in player_tracks.items():
                if track_data['last_seen'] == frame_nbr:  # Only show currently tracked players
                    recent_pos = track_data['positions'][-1]
                    bbox = recent_pos['bbox']
                    score = recent_pos['score']
                    team_name = track_data['team']
                    color_rgb = track_data['color']
                    color_bgr = color_rgb[::-1]
                    palette = track_data['palette']
                    
                    if show_p:
                        # Draw bounding box
                        annotated_frame = cv2.rectangle(annotated_frame, 
                                                      (int(bbox[0]), int(bbox[1])),
                                                      (int(bbox[2]), int(bbox[3])), 
                                                      color_bgr, 2)
                        
                        # Draw track ID and team name
                        label_text = f"ID:{track_id} {team_name} {score:.2f}"
                        annotated_frame = cv2.putText(annotated_frame, label_text,
                                    (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color_bgr, 2)
                    
                    # Show color palette
                    if show_pal:
                        for k, c in enumerate(palette):
                            c_bgr = c[::-1]
                            annotated_frame = cv2.rectangle(annotated_frame, 
                                                          (int(bbox[2])+3, int(bbox[1])+k*palette_box_size),
                                                          (int(bbox[2])+palette_box_size, int(bbox[1])+(palette_box_size)*(k+1)),
                                                          c_bgr, -1)
                    
                    # Add player position on tactical map
                    dst_pos = recent_pos['dst_pos']
                    tac_map_copy = cv2.circle(tac_map_copy, (int(dst_pos[0]), int(dst_pos[1])),
                                            radius=8, color=color_bgr, thickness=-1)
                    tac_map_copy = cv2.circle(tac_map_copy, (int(dst_pos[0]), int(dst_pos[1])),
                                            radius=8, color=(0,0,0), thickness=1)
                    
                    # Draw player ID on tactical map
                    tac_map_copy = cv2.putText(tac_map_copy, str(track_id),
                                             (int(dst_pos[0])-5, int(dst_pos[1])+3), 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    
                    # Draw player trail on tactical map (last 10 positions)
                    if len(track_data['positions']) > 1:
                        trail_points = []
                        for pos_data in track_data['positions'][-10:]:
                            trail_points.append((int(pos_data['dst_pos'][0]), int(pos_data['dst_pos'][1])))
                        
                        if len(trail_points) > 1:
                            points = np.array(trail_points, dtype=np.int32).reshape((-1, 1, 2))
                            tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, 
                                                       color=color_bgr, thickness=2)

            # Annotate other detections (referees, ball)
            for i in range(bboxes_p.shape[0]):
                if labels_p[i] != 0:  # Not a player
                    conf = confs_p[i]
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,0]), int(bboxes_p[i,1])),
                                                  (int(bboxes_p[i,2]), int(bboxes_p[i,3])), (255,255,255), 1)
                    annotated_frame = cv2.putText(annotated_frame, labels_dic[labels_p[i]] + f" {conf:.2f}",
                                (int(bboxes_p[i,0]), int(bboxes_p[i,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255), 2)

                    # Add ball position on tactical map
                    if detected_ball_src_pos is not None and 'homog' in locals():
                        tac_map_copy = cv2.circle(tac_map_copy, (int(detected_ball_dst_pos[0]), 
                                                  int(detected_ball_dst_pos[1])), radius=5, 
                                                  color=ball_color_bgr, thickness=3)

            # Show keypoints if enabled
            if show_k:
                for i in range(bboxes_k.shape[0]):
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i,0]), int(bboxes_k[i,1])),
                                                  (int(bboxes_k[i,2]), int(bboxes_k[i,3])), (0,0,0), 1)

            # Plot ball tracks
            if len(ball_track_history['dst'])>0:
                points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
                tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2)
            
            # Combine annotated frame and tactical map in one image
            border_color = [255,255,255]
            annotated_frame=cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10,
                                              cv2.BORDER_CONSTANT, value=border_color)
            tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT,
                                            value=border_color)      
            tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0]))
            final_img = cv2.hconcat((annotated_frame, tac_map_copy))
            
            ## Add info annotations
            cv2.putText(final_img, "Enhanced Player Tracking", (1270,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            cv2.putText(final_img, f"Active Tracks: {len([t for t in player_tracks.values() if t['last_seen'] == frame_nbr])}", 
                       (1270,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(final_img, "FPS: " + str(int(fps)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
            # Display the annotated frame
            stframe.image(final_img, channels="BGR")
            
            if save_output:
                output.write(cv2.resize(final_img, (width, height)))

    # Clean up expired tracks (not seen for more than 30 frames)
    current_frame = frame_nbr
    expired_tracks = [track_id for track_id, track_data in player_tracks.items() 
                     if current_frame - track_data['last_seen'] > 30]
    for track_id in expired_tracks:
        del player_tracks[track_id]

    # Remove progress bar and return        
    st_prog_bar.empty()
    return True
