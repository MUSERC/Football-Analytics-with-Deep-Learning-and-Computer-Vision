import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import lap

@dataclass
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    shared_kalman = None
    
    def __init__(self, tlwh, score, track_id=None):
        # Wait time for initialization
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.mean = None
        
        self.state = TrackState.New
        self.frame_id = 0
        self.start_frame = 0
        self.track_id = track_id
        
    @property
    def end_frame(self):
        return self.frame_id
    
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy().astype(np.float32)
        ret = self.mean[:4].copy().astype(np.float32)
        ret[2:] = ret[2:] * ret[:2]  # convert xywh to tlwh
        return ret.astype(np.float32)
    
    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy().astype(np.float32)
        ret[2:] = ret[:2] + ret[2:]
        return ret.astype(np.float32)
    
    @property
    def xywh(self):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = self.tlwh.copy().astype(np.float32)
        ret[:2] += ret[2:] / 2
        return ret.astype(np.float32)
    
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :param new_track:
        :param frame_id:
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh.astype(np.float32)
        self._tlwh = new_tlwh
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
    
    def activate(self, frame_id, assigned_id=None):
        """Start a new tracklet"""
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        self.state = TrackState.Tracked
        if self.track_id is None:
            if assigned_id is not None:
                self.track_id = assigned_id
            else:
                self.track_id = self.next_id()
        self.tracklet_len = 0
        self.is_activated = True
    
    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track.tlwh.astype(np.float32)
        self.score = new_track.score
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
    
    def mark_lost(self):
        self.state = TrackState.Lost
    
    def mark_removed(self):
        self.state = TrackState.Removed
    
    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

# Initialize count
STrack._count = 0

class ByteTracker:
    def __init__(self, frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8, max_players=22):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.max_players = max_players
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        # Enhanced fixed player ID pool (1-22)
        self.player_id_pool = set(range(1, max_players + 1))  # Available IDs
        self.reserved_ids = {}  # ID -> last_seen_frame mapping
        self.id_to_track = {}  # ID -> STrack mapping for active tracks
        self.lost_id_positions = {}  # Track last known positions for recovery
        self.id_position_history = {}  # Position history for map-based recovery
        
        # Enhanced matching parameters
        self.position_match_thresh = 80.0  # Pixels threshold for position-based matching
        self.max_lost_frames = 120  # Maximum frames to keep lost track for recovery
        
        self.frame_id = 0
    
    def get_available_id(self):
        """Get next available ID from the fixed pool"""
        if self.player_id_pool:
            return self.player_id_pool.pop()
        return None
    
    def release_id(self, track_id):
        """Release ID back to the pool if track is permanently lost"""
        if track_id in self.reserved_ids:
            # Store last known position before releasing
            if track_id in self.id_to_track:
                track = self.id_to_track[track_id]
                if hasattr(track, '_tlwh'):
                    center_x = track._tlwh[0] + track._tlwh[2] / 2
                    center_y = track._tlwh[1] + track._tlwh[3] / 2
                    self.lost_id_positions[track_id] = {
                        'position': (center_x, center_y),
                        'frame_lost': self.frame_id,
                        'bbox': track._tlwh.copy()
                    }
                del self.id_to_track[track_id]
            
            # Only release if track has been lost for a very long time
            if self.frame_id - self.reserved_ids[track_id] > self.max_lost_frames:
                del self.reserved_ids[track_id]
                self.player_id_pool.add(track_id)
                # Clean up position history
                if track_id in self.lost_id_positions:
                    del self.lost_id_positions[track_id]
    
    def reserve_id(self, track_id):
        """Reserve ID for potentially lost track"""
        self.reserved_ids[track_id] = self.frame_id
    
    def try_reuse_id(self, new_track, existing_tracks=None):
        """Try to reuse ID from lost tracks based on position similarity"""
        if not hasattr(new_track, '_tlwh'):
            return None
            
        new_center_x = new_track._tlwh[0] + new_track._tlwh[2] / 2
        new_center_y = new_track._tlwh[1] + new_track._tlwh[3] / 2
        new_pos = (new_center_x, new_center_y)
        
        best_match_id = None
        best_distance = float('inf')
        
        # First check recently reserved IDs based on position
        for track_id, last_seen in list(self.reserved_ids.items()):
            # Only consider recently lost tracks
            frames_since_lost = self.frame_id - last_seen
            if frames_since_lost > self.track_buffer:
                continue
                
            # Check if we have position data for this ID
            if track_id in self.lost_id_positions:
                lost_pos = self.lost_id_positions[track_id]['position']
                distance = np.sqrt((new_pos[0] - lost_pos[0])**2 + (new_pos[1] - lost_pos[1])**2)
                
                if distance < self.position_match_thresh and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
        
        # If no good match found in reserved IDs, check lost positions
        if best_match_id is None:
            expired_ids = []
            for track_id, lost_data in list(self.lost_id_positions.items()):
                frames_since_lost = self.frame_id - lost_data['frame_lost']
                
                # Remove expired lost tracks
                if frames_since_lost > self.max_lost_frames:
                    expired_ids.append(track_id)
                    continue
                
                # Skip if ID is still reserved (already checked above)
                if track_id in self.reserved_ids:
                    continue
                    
                # Calculate position distance
                lost_pos = lost_data['position']
                distance = np.sqrt((new_pos[0] - lost_pos[0])**2 + (new_pos[1] - lost_pos[1])**2)
                
                if distance < self.position_match_thresh and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
            
            # Clean up expired IDs
            for expired_id in expired_ids:
                if expired_id in self.lost_id_positions:
                    del self.lost_id_positions[expired_id]
        
        # If found a match, clean up and reserve
        if best_match_id is not None:
            if best_match_id in self.lost_id_positions:
                del self.lost_id_positions[best_match_id]
            if best_match_id in self.player_id_pool:
                self.player_id_pool.discard(best_match_id)
            return best_match_id
        
        return None
    
    def update_id_position_history(self, track_id, position):
        """Update position history for an ID"""
        if track_id not in self.id_position_history:
            self.id_position_history[track_id] = []
        
        self.id_position_history[track_id].append({
            'frame': self.frame_id,
            'position': position
        })
        
        # Keep only recent history (last 30 frames for trails)
        if len(self.id_position_history[track_id]) > 30:
            self.id_position_history[track_id].pop(0)
    
    def get_id_preview_info(self):
        """Get information about ID availability for preview"""
        available_ids = sorted(list(self.player_id_pool))
        active_ids = sorted(list(self.id_to_track.keys()))
        lost_recoverable = sorted(list(self.lost_id_positions.keys()))
        
        return {
            'available': available_ids,
            'active': active_ids,
            'lost_recoverable': lost_recoverable,
            'total_capacity': self.max_players,
            'usage': f"{len(active_ids)}/{self.max_players}"
        }
    
    def update(self, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if len(output_results):
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            
            # Filter out low confidence detections
            remain_inds = scores > self.track_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
        else:
            dets = np.empty((0, 4))
            scores_keep = np.empty((0,))
        
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(tlwh, s) for (tlwh, s) in zip(dets, scores_keep)]
        else:
            detections = []
        
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        dists = matching_cost(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists = matching_cost(r_tracked_stracks, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                # Reserve the ID when track is lost
                self.reserve_id(track.track_id)
                lost_stracks.append(track)
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching_cost(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        """ Step 4: Init new stracks with fixed ID pool"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            
            # Try to reuse ID from recently lost tracks
            reuse_id = self.try_reuse_id(track, self.lost_stracks + self.removed_stracks)
            
            if reuse_id is not None:
                # Reuse existing ID
                track.activate(self.frame_id, assigned_id=reuse_id)
                if reuse_id in self.reserved_ids:
                    del self.reserved_ids[reuse_id]
            else:
                # Get new ID from pool
                new_id = self.get_available_id()
                if new_id is not None:
                    track.activate(self.frame_id, assigned_id=new_id)
                else:
                    # No available IDs, skip this detection
                    continue
            
            self.id_to_track[track.track_id] = track
            activated_starcks.append(track)
        
        """ Step 5: Update state and manage IDs"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.track_buffer:
                track.mark_removed()
                # Release ID after extended loss period
                self.release_id(track.track_id)
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def bbox_ious_numpy(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes using pure NumPy
    boxes1: (N, 4) array of [x1, y1, x2, y2]
    boxes2: (M, 4) array of [x1, y1, x2, y2]
    Returns: (N, M) array of IoU values
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    
    boxes1 = np.array(boxes1, dtype=np.float32)
    boxes2 = np.array(boxes2, dtype=np.float32)
    
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Expand dimensions for broadcasting
    boxes1 = np.expand_dims(boxes1, 1)  # (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, 0)  # (1, M, 4)
    area1 = np.expand_dims(area1, 1)    # (N, 1)
    area2 = np.expand_dims(area2, 0)    # (1, M)
    
    # Calculate intersection
    inter_x1 = np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
    inter_y1 = np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
    inter_x2 = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
    inter_y2 = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
    
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    # Calculate IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    return iou.astype(np.float32)

def matching_cost(tracks, detections):
    if (len(tracks) == 0) or (len(detections) == 0):
        return np.empty((len(tracks), len(detections)), dtype=np.float32)
    
    track_boxes = np.array([t.tlbr for t in tracks], dtype=np.float32)
    det_boxes = np.array([d.tlbr for d in detections], dtype=np.float32)
    
    ious = bbox_ious_numpy(track_boxes, det_boxes)
    cost_matrix = 1 - ious
    
    return cost_matrix

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

class BallTouchDetector:
    """
    Detects ball touches for specific player IDs based on tactical map distances
    """
    def __init__(self, touch_threshold_cm=30, tactical_map_size=(890, 570)):
        """
        Initialize ball touch detector
        
        Args:
            touch_threshold_cm: Distance threshold in centimeters for ball touch
            tactical_map_size: Size of tactical map in pixels (width, height)
        """
        self.touch_threshold_cm = touch_threshold_cm
        self.tactical_map_size = tactical_map_size
        
        # Football field dimensions in meters (FIFA standard)
        self.field_length_m = 105.0  # meters
        self.field_width_m = 68.0    # meters
        
        # Calculate pixels per meter for tactical map
        self.pixels_per_meter_x = tactical_map_size[0] / self.field_length_m
        self.pixels_per_meter_y = tactical_map_size[1] / self.field_width_m
        
        # Convert touch threshold from cm to pixels
        self.touch_threshold_pixels = (self.touch_threshold_cm / 100.0) * \
                                    ((self.pixels_per_meter_x + self.pixels_per_meter_y) / 2)
        
        # Track ball touches for each player
        self.player_touches = {}  # player_id -> touch_count
        self.player_ball_states = {}  # player_id -> {'in_contact': bool, 'last_contact_frame': int}
        self.monitored_players = set()  # Set of player IDs to monitor
        
    def add_monitored_player(self, player_id):
        """Add player ID to monitoring list"""
        self.monitored_players.add(player_id)
        if player_id not in self.player_touches:
            self.player_touches[player_id] = 0
            self.player_ball_states[player_id] = {'in_contact': False, 'last_contact_frame': 0}
    
    def remove_monitored_player(self, player_id):
        """Remove player ID from monitoring list"""
        self.monitored_players.discard(player_id)
    
    def calculate_distance_pixels(self, pos1, pos2):
        """Calculate Euclidean distance between two positions in pixels"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update_ball_touches(self, player_positions, ball_position, frame_id):
        """
        Update ball touches for monitored players
        
        Args:
            player_positions: dict {player_id: (x, y)} positions on tactical map
            ball_position: (x, y) ball position on tactical map
            frame_id: current frame number
        """
        if ball_position is None:
            return
        
        for player_id in self.monitored_players:
            if player_id in player_positions:
                player_pos = player_positions[player_id]
                distance = self.calculate_distance_pixels(player_pos, ball_position)
                
                player_state = self.player_ball_states[player_id]
                
                # Check if player is within touch threshold
                is_in_contact = distance <= self.touch_threshold_pixels
                
                # Count touch when player comes into contact
                if is_in_contact and not player_state['in_contact']:
                    self.player_touches[player_id] += 1
                    player_state['last_contact_frame'] = frame_id
                
                # Update contact state
                player_state['in_contact'] = is_in_contact
    
    def get_touch_count(self, player_id):
        """Get touch count for specific player"""
        return self.player_touches.get(player_id, 0)
    
    def get_all_touches(self):
        """Get touch counts for all monitored players"""
        return {pid: self.player_touches.get(pid, 0) for pid in self.monitored_players}
    
    def reset_touches(self, player_id=None):
        """Reset touch counts (for specific player or all players)"""
        if player_id is not None:
            if player_id in self.player_touches:
                self.player_touches[player_id] = 0
                self.player_ball_states[player_id]['in_contact'] = False
        else:
            for pid in self.monitored_players:
                self.player_touches[pid] = 0
                self.player_ball_states[pid]['in_contact'] = False
    
    def get_touch_info(self):
        """Get comprehensive touch information"""
        touch_info = {}
        for player_id in self.monitored_players:
            touch_info[player_id] = {
                'touches': self.get_touch_count(player_id),
                'in_contact': self.player_ball_states[player_id]['in_contact'],
                'last_contact_frame': self.player_ball_states[player_id]['last_contact_frame']
            }
        return touch_info
