import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import lap
from cython_bbox import bbox_overlaps as bbox_ious

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
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2:] = ret[2:] * ret[:2]  # convert xywh to tlwh
        return ret
    
    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    @property
    def xywh(self):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret
    
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :param new_track:
        :param frame_id:
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
    
    def activate(self, frame_id):
        """Start a new tracklet"""
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        self.state = TrackState.Tracked
        if self.track_id is None:
            self.track_id = self.next_id()
        self.tracklet_len = 0
        self.is_activated = True
    
    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track.tlwh
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
    def __init__(self, frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
    
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
        
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)
        
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.track_buffer:
                track.mark_removed()
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

def matching_cost(tracks, detections):
    if (len(tracks) == 0) or (len(detections) == 0):
        return np.empty((len(tracks), len(detections)), dtype=np.float32)
    
    track_boxes = np.array([t.tlbr for t in tracks])
    det_boxes = np.array([d.tlbr for d in detections])
    
    ious = bbox_ious(track_boxes, det_boxes)
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
