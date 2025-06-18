import os
import numpy as np
import random
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from dtaidistance import dtw
from pose_format import Pose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

# Configuration
WORDS_FOLDER = "E:/ISH/pose_data/CISLR/CISLR_v1.5-a_videos_poses/poses"
SENTENCES_FOLDER = "E:/ISH/pose_data/sports_mp4_pose"
hand_weight=3
# Define arm-related landmarks
# Upper body/arm keypoints typically include:
# 11-12: shoulders
# 13-14: elbows
# 15-16: wrists
# 17-22: hands/fingers (if available)
POSE_CAMERA_OFFSET = 0
POSE_WORLD_OFFSET = 33
FACE_OFFSET         = 66
LEFT_HAND_OFFSET    = 33 + 468  # = 534
RIGHT_HAND_OFFSET   = LEFT_HAND_OFFSET + 21  # = 555

# Local indices in each block for arm and wrist
CAMERA_ARM_LANDMARKS = [11, 12, 13, 14, 15, 16]
CAMERA_ARM_WORLD_LANDMARKS = [11, 12, 13, 14, 15, 16]
LEFT_HAND_LANDMARKS = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + 21))
RIGHT_HAND_LANDMARKS = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + 21))

def load_pose(file_path):
    """Load pose data from file"""
    with open(file_path, "rb") as f:
        pose = Pose.read(f.read())
    return pose

def extract_features(pose, feature_type="arm"):
    """
    Extract arm and hand features from a unified 576-point landmark array.

    Parameters:
      pose: object with attributes:
        - body.data: np array of shape (frames, 1, 576, 3)
        - body.confidence: np array of shape (frames, 576)
      feature_type: "arm", "left_hand", "right_hand" or "all"

    Returns:
      features: dict of sliced landmark coordinates
      confidences: dict of corresponding confidence scores
    """
    # flatten the 1-length second axis
    data = pose.body.data[:, 0, :, :]        # (frames, 576, 3)
    conf = pose.body.confidence[:,0,:]               # (frames, 576)

    features = {}
    confidences = {}

    # ARM features (camera-space)
    if feature_type in ["arm", "all"]:
        # camera-space shoulders/elbows/wrists
        cam_idxs = [i + POSE_CAMERA_OFFSET for i in CAMERA_ARM_LANDMARKS]
        features["camera_arm"] = data[:, cam_idxs, :]
        confidences["camera_arm"] = conf[:, cam_idxs]

        # world-space shoulders/elbows/wrists
        world_idxs = [i + POSE_WORLD_OFFSET for i in CAMERA_ARM_WORLD_LANDMARKS]
        features["world_arm"] = data[:, world_idxs, :]
        confidences["world_arm"] = conf[:, world_idxs]

    # LEFT hand model
    if feature_type in ["left_hand", "all","arm"]:
        lh_idxs = LEFT_HAND_LANDMARKS
        features["left_hand"] = data[:, lh_idxs, :]
        confidences["left_hand"] = conf[:, lh_idxs]

    # RIGHT hand model
    if feature_type in ["right_hand", "all","arm"]:
        rh_idxs = RIGHT_HAND_LANDMARKS
        features["right_hand"] = data[:, rh_idxs, :]
        confidences["right_hand"] = conf[:, rh_idxs]

    return features, confidences

def normalize_pose_data(pose_data, confidence=None):
    """Normalize pose data using hip coordinates"""
    # Use hip landmarks (23, 24) for normalization if available
    if pose_data.shape[1] > 24:  # Check if we have the hip landmarks
        hips = pose_data[:, [23, 24]]  # Left and right hip
        hip_center = np.mean(hips, axis=1, keepdims=True)
        
        # Center using hip coordinates
        centered_data = pose_data - hip_center
        
        # Scale using the distance between shoulders (appropriate for arm movements)
        if pose_data.shape[1] > 12:  # Ensure we have shoulder landmarks
            shoulder_dist = np.linalg.norm(pose_data[:, 11] - pose_data[:, 12], axis=1)
            scale = shoulder_dist.reshape(-1, 1, 1) + 1e-8
            normalized_data = centered_data / scale
        else:
            # Fallback to standard deviation scaling if shoulder landmarks aren't available
            scale = np.std(np.linalg.norm(centered_data, axis=2), axis=1, keepdims=True)
            normalized_data = centered_data / (scale[:, :, np.newaxis] + 1e-8)
    else:
        # Fallback to using mean of all landmarks as center
        center = np.mean(pose_data, axis=1, keepdims=True)
        centered_data = pose_data - center
        
        # Scale using standard deviation
        scale = np.std(np.linalg.norm(centered_data, axis=2), axis=1, keepdims=True)
        normalized_data = centered_data / (scale[:, :, np.newaxis] + 1e-8)
    
    return normalized_data

def compute_dtw_distance(query, candidate, confidence_q=None, confidence_c=None, dimension_weights=None):
    """
    Compute DTW distance optimized for arm landmarks with confidence weighting.
    
    Parameters:
      query: numpy array of shape (frames, landmarks, coordinates) representing the query sequence.
      candidate: numpy array of shape (frames, landmarks, coordinates) representing the candidate sequence.
      confidence_q: Optional numpy array of shape (frames, landmarks) containing confidence scores for the query.
      confidence_c: Optional numpy array of shape (frames, landmarks) containing confidence scores for the candidate.
      dimension_weights: Optional list/tuple of weights for each coordinate dimension. 
                         Default: [0.4, 0.4, 0.2] for X, Y, and Z, emphasizing X and Y over Z.
    
    Returns:
      The average normalized DTW distance across landmarks. If no landmarks could be computed, returns float('inf').
      
    Note:
      This function uses dtw.distance_fast to compute the DTW distance on 1D time series data for each coordinate.
    """
    
    # Get shapes for query and candidate
    num_frames_q, num_landmarks_q, num_coords_q = query.shape
    num_frames_c, num_landmarks_c, num_coords_c = candidate.shape
    
    # Ensure same number of landmarks and coordinates
    num_landmarks = min(num_landmarks_q, num_landmarks_c)
    num_coords = min(num_coords_q, num_coords_c)
    
    # Apply confidence weighting if available
    if confidence_q is not None and confidence_c is not None:
        query = query * confidence_q[:, :, np.newaxis]
        candidate = candidate * confidence_c[:, :, np.newaxis]
    
    # Default weights: weight X and Y more than Z if not provided
    if dimension_weights is None:
        dimension_weights = [0.33, 0.33, 0.33]
    
    landmark_distances = []
    
    for landmark_idx in range(num_landmarks):
        # Extract trajectory for the current landmark
        query_landmark = query[:, landmark_idx, :num_coords]
        candidate_landmark = candidate[:, landmark_idx, :num_coords]
        
        try:
            distance_sum = 0
            # Compute DTW distance for each coordinate dimension separately
            for dim in range(min(3, num_coords)):
                # Extract the 1D time series for the current dimension
                q_dim = np.ascontiguousarray(query_landmark[:, dim], dtype=np.double)
                c_dim = np.ascontiguousarray(candidate_landmark[:, dim], dtype=np.double)
                
                # Compute DTW distance for this dimension
                dim_distance = dtw.distance_fast(q_dim, c_dim)
                distance_sum += dimension_weights[dim] * dim_distance
                
            # Normalize by the number of frames in the query sequence
            landmark_distances.append(distance_sum / num_frames_q)
        except Exception as e:
            continue

    return np.mean(landmark_distances) if landmark_distances else float('inf')


def compute_cosine_similarity(query, candidate, confidence_q=None, confidence_c=None):
    """
    Compute cosine similarity comparing each landmark separately, then
    multiply each landmark's similarity by its average confidence.
    
    Parameters:
      query: numpy array, shape (frames, landmarks, coords)
      candidate: numpy array, shape (frames, landmarks, coords)
      confidence_q: optional, shape (frames, landmarks)
      confidence_c: optional, shape (frames, landmarks)
      
    Returns:
      Weighted average cosine similarity across all landmarks.
      (Higher = more similar.)
    """
    num_frames_q, num_landmarks_q, num_coords_q = query.shape
    num_frames_c, num_landmarks_c, num_coords_c = candidate.shape
    
    num_landmarks = min(num_landmarks_q, num_landmarks_c)
    num_coords    = min(num_coords_q,   num_coords_c)
    
    landmark_scores = []
    
    # weights for X, Y, Z
    dim_w = np.array([0.333, 0.333, 0.333])
    
    for i in range(num_landmarks):
        q_lm = query[:, i, :num_coords]
        c_lm = candidate[:, i, :num_coords]
        
        # use overlapping frames
        n = min(len(q_lm), len(c_lm))
        q_lm = q_lm[:n]
        c_lm = c_lm[:n]
        
        # compute per-dimension cosine similarity
        sim_dims = []
        for d in range(min(3, num_coords)):
            try:
                sim = 1 - cosine(q_lm[:, d], c_lm[:, d])
            except Exception:
                sim = 0.0
            if np.isnan(sim):
                sim = 0.0
            sim_dims.append(sim)
        sim_dims = np.array(sim_dims)
        
        # weighted sum over dims
        base_sim = np.dot(dim_w[:len(sim_dims)], sim_dims)
        
        # now compute average confidence for this landmark (if provided)
        if confidence_q is not None and confidence_c is not None:
            cq = confidence_q[:n, i].mean()
            cc = confidence_c[:n, i].mean()
            avg_conf = 0.5 * (cq + cc)
        else:
            avg_conf = 1.0
        
        # scale similarity by confidence
        landmark_scores.append(base_sim*avg_conf)
    
    # final average
    if not landmark_scores:
        return 0.0
    return float(np.mean(landmark_scores))

def compute_correlation(query, candidate, confidence_q=None, confidence_c=None):
    """
    Compute correlation comparing each landmark separately with confidence weighting
    Optimized for arm movements
    """
    num_frames_q, num_landmarks_q, num_coords_q = query.shape
    num_frames_c, num_landmarks_c, num_coords_c = candidate.shape
    
    # Ensure same number of landmarks and coordinates
    num_landmarks = min(num_landmarks_q, num_landmarks_c)
    num_coords = min(num_coords_q, num_coords_c)
    
    # Apply confidence weighting if available
    if confidence_q is not None and confidence_c is not None:
        query = query * confidence_q[:, :, np.newaxis]
        candidate = candidate * confidence_c[:, :, np.newaxis]
    
    # Calculate correlation for each landmark
    landmark_correlations = []
    
    for landmark_idx in range(num_landmarks):
        # Extract trajectory for this landmark
        query_landmark = query[:, landmark_idx, :num_coords]
        candidate_landmark = candidate[:, landmark_idx, :num_coords]
        
        # Skip landmarks with no movement
        if np.std(query_landmark) < 1e-6 or np.std(candidate_landmark) < 1e-6:
            continue
        
        try:
            # Compute correlation for each dimension separately
            dimension_weights = [0.4, 0.4, 0.2]  # Weight X,Y more than Z
            correlation_sum = 0
            
            for dim in range(min(3, num_coords)):
                q_dim = query_landmark[:, dim]
                c_dim = candidate_landmark[:, dim]
                
                if len(q_dim) >= 2 and len(c_dim) >= 2:
                    # Calculate correlation for this dimension
                    correlation = np.corrcoef(q_dim, c_dim[:len(q_dim)])[0, 1]
                    
                    if not np.isnan(correlation):
                        correlation_sum += dimension_weights[dim] * correlation
            
            landmark_correlations.append(correlation_sum)
        except Exception as e:
            continue
    
    # Return average correlation across all landmarks
    return np.mean(landmark_correlations) if landmark_correlations else 0

def compute_similarity(query, candidate, confidence_q, confidence_c, method):
    """Compute similarity between query and candidate features"""
    if method == "dtw":
        return compute_dtw_distance(query, candidate, confidence_q, confidence_c)
    elif method == "cosine":
        return compute_cosine_similarity(query, candidate, confidence_q, confidence_c)
    elif method == "euclidean":
        return compute_euclidean_distance(query, candidate, confidence_q, confidence_c)
    elif method == "correlation":
        return compute_correlation(query, candidate, confidence_q, confidence_c)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def sliding_window_match(query_features, query_confidences, sentence_features, sentence_confidences, 
                        similarity_method, window_step=1, movement_threshold=0.01, context_frames=5):
    """Find best match with automatic query trimming"""
    # 1. Calculate hand movement for each frame
    def get_movement(features):
        movement = []
        for i in range(1, len(features)):
            movement.append(np.linalg.norm(features[i] - features[i-1]))
        return np.array(movement + [0])  # Pad last frame
    
    # Combine left/right hand movement
    lh_mov = get_movement(query_features.get('left_hand', []))
    rh_mov = get_movement(query_features.get('right_hand', []))
    combined_mov = np.maximum(lh_mov, rh_mov)
    
    # 2. Find dynamic segments with context
    smoothed_mov = np.convolve(combined_mov, np.ones(3)/3, 'same')  # Smooth with 3-frame window
    active_frames = smoothed_mov > movement_threshold
    
    # Find first/last active frames with context
    start_idx = max(0, np.argmax(active_frames) - context_frames)
    end_idx = min(len(active_frames) - 1, len(active_frames) - np.argmax(active_frames[::-1]) + context_frames)
    
    # 3. Trim query features (keep at least 5 frames)
    min_query_length = 5
    if (end_idx - start_idx) < min_query_length:
        start_idx = 0
        end_idx = len(combined_mov)
    else:
        # Ensure we don't trim too aggressively
        start_idx = max(0, start_idx - context_frames)
        end_idx = min(len(combined_mov), end_idx + context_frames)
    
    # Create trimmed query
    trimmed_query = {
        k: v[start_idx:end_idx] for k, v in query_features.items()
    }
    trimmed_conf = {
        k: v[start_idx:end_idx] for k, v in query_confidences.items()
    }
    
    # Original sliding window logic using trimmed query
    primary_feature = "right_hand"
    window_size = len(trimmed_query[primary_feature])
    
    # Handle case where sentence is shorter than query
    if len(sentence_features[primary_feature]) < window_size:
        if similarity_method in ["cosine", "correlation"]:
            return 0, -1  # Higher is better
        else:
            return 0, float('inf')  # Lower is better
    
    # Initialize best score based on similarity method
    if similarity_method in ["cosine", "correlation"]:
        best_score = -1  # Higher is better
        is_higher_better = True
    else:  # DTW, Euclidean
        best_score = float('inf')  # Lower is better
        is_higher_better = False
    
    best_start = 0
    
    # Optimization: Increase window step for longer sequences
    adaptive_window_step = window_step
    if len(sentence_features[primary_feature]) > 5 * window_size:
        adaptive_window_step = max(window_step, window_size // 10)
    
    # Slide window through sentence features
    for start in range(0, len(sentence_features[primary_feature]) - window_size + 1, adaptive_window_step):
        total_score = 0
        feature_count = 0
        
        # Compute similarity for each feature type
        for feat_type in query_features:
            if feat_type in sentence_features:
                query_data = query_features[feat_type]
                query_conf = query_confidences[feat_type]
                window_data = sentence_features[feat_type][start:start+window_size]
                window_conf = sentence_confidences[feat_type][start:start+window_size]
                
                # Ensure same number of frames
                min_frames = min(len(query_data), len(window_data))
                query_data = query_data[:min_frames]
                query_conf = query_conf[:min_frames]
                window_data = window_data[:min_frames]
                window_conf = window_conf[:min_frames]
                
                if feat_type=="left_hand" or feat_type=="right_hand":
                    score = compute_similarity(query_data, window_data, query_conf, window_conf, similarity_method)*hand_weight
                    feature_count+=hand_weight
                else:
                    score = compute_similarity(query_data, window_data, query_conf, window_conf, similarity_method)
                total_score += score
                feature_count += 1
        
        # Calculate average score across all feature types
        if feature_count > 0:
            avg_score = total_score / feature_count
            
            # Update best score
            if (is_higher_better and avg_score > best_score) or (not is_higher_better and avg_score < best_score):
                best_score = avg_score
                best_start = start
                
                # Optimization: Early stopping if we find an extremely good match
                if (is_higher_better and avg_score > 0.95) or (not is_higher_better and avg_score < 0.05):
                    break
    
    # Refine search around best match if using adaptive step
    if adaptive_window_step > 1:
        # Define search range around best match
        refine_start = max(0, best_start - adaptive_window_step)
        refine_end = min(len(sentence_features[primary_feature]) - window_size, best_start + adaptive_window_step)
        
        # Fine-grained search
        for start in range(refine_start, refine_end + 1, 1):
            total_score = 0
            feature_count = 0
            
            # Compute similarity for each feature type
            for feat_type in query_features:
                if feat_type in sentence_features:
                    query_data = query_features[feat_type]
                    query_conf = query_confidences[feat_type]
                    window_data = sentence_features[feat_type][start:start+window_size]
                    window_conf = sentence_confidences[feat_type][start:start+window_size]
                    
                    # Ensure same number of frames
                    min_frames = min(len(query_data), len(window_data))
                    query_data = query_data[:min_frames]
                    query_conf = query_conf[:min_frames]
                    window_data = window_data[:min_frames]
                    window_conf = window_conf[:min_frames]
                    
                    score = compute_similarity(query_data, window_data, query_conf, window_conf, similarity_method)
                    total_score += score
                    feature_count += 1
            
            # Calculate average score
            if feature_count > 0:
                avg_score = total_score / feature_count
                
                # Update best score
                if (is_higher_better and avg_score > best_score) or (not is_higher_better and avg_score < best_score):
                    best_score = avg_score
                    best_start = start
    
    return best_start, best_score

def detect_sign_word(word_path, sentence_path, similarity_method="dtw", feature_type="all"):
    """Detect if a sign word is present in a sentence with focus on arm features"""
    # Load pose data
    word_pose = load_pose(word_path)
    sentence_pose = load_pose(sentence_path)
    
    # Extract features and confidences with focus on arms
    word_features, word_confidences = extract_features(word_pose, feature_type)
    sentence_features, sentence_confidences = extract_features(sentence_pose, feature_type)
    
    # Normalize features using hip-based normalization
    for feat_type in word_features:
        word_features[feat_type] = normalize_pose_data(word_features[feat_type], word_confidences[feat_type])
    
    for feat_type in sentence_features:
        sentence_features[feat_type] = normalize_pose_data(sentence_features[feat_type], sentence_confidences[feat_type])
    
    # Find best match
    start_idx, score = sliding_window_match(
        word_features, word_confidences,
        sentence_features, sentence_confidences,
        similarity_method
    )
    
    # Define thresholds based on similarity method - adjusted for arm focus
    # These may need tuning based on your specific data
    thresholds = {
        "cosine": 0.65,       # Higher is better, more strict for arm-only
        "dtw": 0.45,          # Lower is better, more permissive for arm-only
        "euclidean": 0.25,    # Lower is better, more permissive for arm-only
        "correlation": 0.65   # Higher is better, more strict for arm-only
    }
    threshold = thresholds[similarity_method]
    
    # Determine if word is present
    if similarity_method in ["cosine", "correlation"]:
        is_present = score > threshold
    else:  # DTW, Euclidean
        is_present = score < threshold
    
    results = {}
    for feat_type in word_features:
        if feat_type in sentence_features:
            results[feat_type] = (start_idx, score)
    
    return is_present, results, score, threshold

def evaluate_mapping(mapping_csv_path, similarity_method="dtw", feature_type="all"):
    """Evaluate word detection over a CSV mapping"""
    import pandas as pd
    
    df = pd.read_csv(mapping_csv_path)
    total_cases = len(df)
    correct_predictions = 0
    
    for idx, row in df.iterrows():
        video_uid, gloss = row['folder'], row['word_id']
        query_pose_path = os.path.join(WORDS_FOLDER, f"{gloss}.pose")
        candidate_pose_path = os.path.join(SENTENCES_FOLDER, f"{video_uid}.pose")
        
        try:
            is_present, results, combined_score, threshold = detect_sign_word(
                query_pose_path, candidate_pose_path, similarity_method, "all"
            )
            
            # Ground truth is assumed to be True
            ground_truth = True
            
            if is_present == ground_truth:
                correct_predictions += 1
            
            print(f"Video: {video_uid}, Word: {gloss}, Score: {combined_score:.4f}, Threshold: {threshold:.4f}, Predicted: {is_present}")
        
        except Exception as e:
            print(f"Error processing {video_uid} with {gloss}: {e}")
    
    accuracy = correct_predictions / total_cases if total_cases > 0 else 0
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    return accuracy

def evaluate_batch_retrieval(
    mapping_csv_path, similarity_method="dtw", feature_type="all", num_words=10
):
    """
    Evaluate retrieval accuracy using unique word-sentence pairs (1:1 mapping).
    Implements:
      4) explicit exception handling when scoring,
      6) random unused candidate selection instead of always picking the first,
      7) correct sort order for distance vs similarity metrics.
    """
    df = pd.read_csv(mapping_csv_path)
    num_words = int(input("Enter a number").strip())
    # 1) Get all unique word_ids from the CSV
    all_word_ids = df['word_id'].unique().tolist()

    # 2) Randomly sample up to num_words of them
    selected_word_ids = random.sample(all_word_ids, min(num_words, len(all_word_ids)))

    # -- Now build your 1:1 pairs as before --
    used_sentences = set()
    word_sentence_pairs = []
    shuffled_words = random.sample(selected_word_ids, len(selected_word_ids))
    
    for word_id in shuffled_words:
        candidates = df.loc[df['word_id'] == word_id, 'folder'].unique()
        # filter out already used sentences
        available = [c for c in candidates if c not in used_sentences]
        if not available:
            continue
        # randomly pick one unused sentence
        selected_sentence = random.choice(available)
        word_sentence_pairs.append((word_id, selected_sentence))
        used_sentences.add(selected_sentence)
    
    candidate_sentences = [s for _, s in word_sentence_pairs]
    
    # Initialize counters
    top5 = top10 = top15 = top20 = 0
    total = len(word_sentence_pairs)
    
    # Determine sort order based on metric
    higher_is_better = similarity_method in ("cosine", "correlation")
    
    for word_id, true_sentence in word_sentence_pairs:
        query_path = os.path.join(WORDS_FOLDER, f"{word_id}.pose")
        if not os.path.exists(query_path):
            continue
        
        sentence_scores = []
        for sentence in candidate_sentences:
            candidate_path = os.path.join(SENTENCES_FOLDER, f"{sentence}.pose")
            if not os.path.exists(candidate_path):
                continue
            print(word_id+"--->"+sentence)
            try:
                result = detect_sign_word(
                    query_path, candidate_path, similarity_method, feature_type
                )
                # robustly extract the 3rd element as score
                if isinstance(result, tuple) or isinstance(result, list):
                    if len(result) >= 3:
                        score = result[2]
                    else:
                        # fallback: last item
                        score = result[-1]
                else:
                    score = float(result)
    
            except Exception as e:
                print(f"⚠️ Error scoring {word_id} vs {sentence}: {e}")
                score = -np.inf if higher_is_better else np.inf
            
            # optionally apply penalty here...
            sentence_scores.append((sentence, score))
        
        # sort correctly
        sentence_scores.sort(key=lambda x: x[1], reverse=higher_is_better)
        
        ranked = [s for s, _ in sentence_scores]
        if true_sentence in ranked[:5]:
            top5 += 1
        if true_sentence in ranked[:10]:
            top10 += 1
        if true_sentence in ranked[:15]:
            top15 += 1
        if true_sentence in ranked[:20]:
            top20 += 1
    
    # Compute accuracies
    accuracy5  = top5  / total if total else 0.0
    accuracy10 = top10 / total if total else 0.0
    accuracy15 = top15 / total if total else 0.0
    accuracy20 = top20 / total if total else 0.0
    
    print(f"\nUnique Pair Evaluation ({total} pairs):")
    print(f" Top-5  : {accuracy5*100:.2f}% ({top5}/{total})")
    print(f" Top-10 : {accuracy10*100:.2f}% ({top10}/{total})")
    print(f" Top-15 : {accuracy15*100:.2f}% ({top15}/{total})")
    print(f" Top-20 : {accuracy20*100:.2f}% ({top20}/{total})")
    
    return accuracy5, accuracy10, accuracy15, accuracy20, total, word_sentence_pairs

def set_axes_equal(ax):
    """
    Make 3D axes have equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range) / 2.0

    x_middle = sum(x_limits) / 2.0
    y_middle = sum(y_limits) / 2.0
    z_middle = sum(z_limits) / 2.0

    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

ARM_CONNECTIONS = [
    (0, 2),  # left shoulder to left elbow
    (2, 4),  # left elbow to left wrist
    (1, 3),  # right shoulder to right elbow
    (3, 5),  # right elbow to right wrist
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_middle = sum(x_limits) / 2.0
    y_middle = sum(y_limits) / 2.0
    z_middle = sum(z_limits) / 2.0
    max_range = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0])
    ) / 2.0
    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

def visualize_arm_hand_connected(word_pose, sentence_pose, start_frame, similarity_method):
    """
    Visualize arms and hands with skeletal connections.
    """
    
    # Extract all features
    word_feat, _ = extract_features(word_pose, 'all')
    sent_feat, _ = extract_features(sentence_pose, 'all')

    win_size = word_feat['camera_arm'].shape[0]
    sent_window = {k: v[start_frame:start_frame+win_size] for k, v in sent_feat.items()}
    
    frame_idxs = [30, win_size // 2, win_size - 30]
    labels = ['Start', 'Middle', 'End']

    fig = plt.figure(figsize=(18, 10))
    for row, (label_prefix, features) in enumerate([('Word', word_feat), ('Sentence', sent_window)]):
        for col, idx in enumerate(frame_idxs):
            ax = fig.add_subplot(2, 3, row*3 + col + 1, projection='3d')
            ax.set_title(f"{label_prefix} – {labels[col]}", pad=10, fontsize=12)
            
            # Plot arms
            arm_coords = features['camera_arm'][idx]
            for a, b in ARM_CONNECTIONS:
                x = [arm_coords[a, 0], arm_coords[b, 0]]
                y = [arm_coords[a, 1], arm_coords[b, 1]]
                z = [arm_coords[a, 2], arm_coords[b, 2]]
                ax.plot(x, y, z, linewidth=3)

            # Plot left hand
            lh_coords = features['left_hand'][idx]
            for a, b in HAND_CONNECTIONS:
                x = [lh_coords[a, 0], lh_coords[b, 0]]
                y = [lh_coords[a, 1], lh_coords[b, 1]]
                z = [lh_coords[a, 2], lh_coords[b, 2]]
                ax.plot(x, y, z, linewidth=2)

            # Plot right hand
            rh_coords = features['right_hand'][idx]
            for a, b in HAND_CONNECTIONS:
                x = [rh_coords[a, 0], rh_coords[b, 0]]
                y = [rh_coords[a, 1], rh_coords[b, 1]]
                z = [rh_coords[a, 2], rh_coords[b, 2]]
                ax.plot(x, y, z, linewidth=2)

            set_axes_equal(ax)
            ax.axis('off')

    #plt.suptitle(f"Arm & Hand Matching ({similarity_method}) at frame {start_frame}", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()

# def visualize_results(word_pose, sentence_pose, start_frame, similarity_method):
#     """
#     Visualize the matching results with 6 separate plots:
#     - 3 plots for word pose (start, middle, end frames)
#     - 3 plots for sentence pose (start, middle, end frames)
#     """
#     # Extract the relevant pose data
#     word_data = word_pose.body.data[:, 0, :, :]  # (frames, landmarks, 3)
#     sentence_data = sentence_pose.body.data[start_frame:start_frame+word_data.shape[0], 0, :, :]
    
#     # Create figure with 2 rows (word and sentence) and 3 columns (start, middle, end)
#     fig = plt.figure(figsize=(18, 10))
    
#     # Get frame indices (start, middle, end)
#     num_frames = word_data.shape[0]
#     frame_indices = [0, num_frames // 2, num_frames - 1]
#     frame_labels = ['Start Frame', 'Middle Frame', 'End Frame']
    
#     # Plot each frame for word pose (top row)
#     for i, frame_idx in enumerate(frame_indices):
#         if frame_idx < word_data.shape[0]:
#             ax = fig.add_subplot(2, 3, i+1, projection='3d')
#             plot_pose_frame(ax, word_data[frame_idx], 'blue')
#             ax.set_title(f"Word Pose - {frame_labels[i]}")
    
#     # Plot each frame for sentence pose (bottom row)
#     for i, frame_idx in enumerate(frame_indices):
#         if frame_idx < sentence_data.shape[0]:
#             ax = fig.add_subplot(2, 3, i+4, projection='3d')
#             plot_pose_frame(ax, sentence_data[frame_idx], 'red')
#             ax.set_title(f"Sentence Pose - {frame_labels[i]} (from frame {start_frame + frame_idx})")
    
#     # Add title
#     plt.suptitle(f"Pose Matching using {similarity_method} - Match starts at frame {start_frame}", fontsize=16)
    
#     # Adjust layout
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()

# def plot_pose_frame(ax, pose_frame, color):
#     """
#     Plot a single frame of pose data with connections between body parts.
#     """
#     # Extract x, y, z coordinates
#     x = pose_frame[:, 0]
#     y = pose_frame[:, 1]
#     z = pose_frame[:, 2]
    
#     # Plot the points with smaller size
#     ax.scatter(x, y, z, color=color, s=10, alpha=0.7)
    
#     # Connect key points with lines
#     # Arms (shoulders, elbows, wrists)
#     connect_points(ax, pose_frame, [11, 13, 15], color)  # Left arm
#     connect_points(ax, pose_frame, [12, 14, 16], color)  # Right arm
    
#     # Torso
#     connect_points(ax, pose_frame, [11, 12], color)  # Shoulders
#     connect_points(ax, pose_frame, [11, 23], color)  # Left shoulder to hip
#     connect_points(ax, pose_frame, [12, 24], color)  # Right shoulder to hip
#     connect_points(ax, pose_frame, [23, 24], color)  # Hips
    
#     # Legs
#     connect_points(ax, pose_frame, [23, 25, 27, 31], color)  # Left leg
#     connect_points(ax, pose_frame, [24, 26, 28, 32], color)  # Right leg
    
#     # Face as a blob (if face landmarks are present)
#     if pose_frame.shape[0] > 468:
#         # Use landmarks 0-10 as face landmarks (adjust based on your model)
#         face_center = np.mean(pose_frame[:10], axis=0)
#         ax.scatter(face_center[0], face_center[1], face_center[2], color=color, s=100, alpha=0.7)
    
#     # Hands and fingers (if present in the data)
#     if pose_frame.shape[0] >= 489:  # Left hand is present
#         # Connect wrist to hand
#         if 15 < pose_frame.shape[0] and 468 < pose_frame.shape[0]:
#             ax.plot([pose_frame[15, 0], pose_frame[468, 0]], 
#                     [pose_frame[15, 1], pose_frame[468, 1]], 
#                     [pose_frame[15, 2], pose_frame[468, 2]], color=color, linewidth=1)
        
#         # Left hand fingers (MediaPipe format)
#         left_hand_indices = list(range(468, 489))
#         plot_hand_connections(ax, pose_frame[left_hand_indices], color)
    
#     if pose_frame.shape[0] >= 510:  # Right hand is present
#         # Connect wrist to hand
#         if 16 < pose_frame.shape[0] and 489 < pose_frame.shape[0]:
#             ax.plot([pose_frame[16, 0], pose_frame[489, 0]], 
#                     [pose_frame[16, 1], pose_frame[489, 1]], 
#                     [pose_frame[16, 2], pose_frame[489, 2]], color=color, linewidth=1)
        
#         # Right hand fingers (MediaPipe format)
#         right_hand_indices = list(range(489, 510))
#         plot_hand_connections(ax, pose_frame[right_hand_indices], color)
    
#     # Set axis properties
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
    
#     # Make the plot more visually consistent
#     max_range = np.max([np.ptp(x), np.ptp(y), np.ptp(z)])
#     if max_range > 0:
#         mid_x = (np.max(x) + np.min(x)) / 2
#         mid_y = (np.max(y) + np.min(y)) / 2
#         mid_z = (np.max(z) + np.min(z)) / 2
#         ax.set_xlim(mid_x - max_range/1.8, mid_x + max_range/1.8)
#         ax.set_ylim(mid_y - max_range/1.8, mid_y + max_range/1.8)
#         ax.set_zlim(mid_z - max_range/1.8, mid_z + max_range/1.8)

# def connect_points(ax, pose_frame, indices, color):
#     """
#     Connect points in the pose frame with lines.
#     """
#     for i in range(len(indices) - 1):
#         ax.plot([pose_frame[indices[i], 0], pose_frame[indices[i+1], 0]], 
#                 [pose_frame[indices[i], 1], pose_frame[indices[i+1], 1]], 
#                 [pose_frame[indices[i], 2], pose_frame[indices[i+1], 2]], 
#                 color=color, linewidth=1.5)

# def plot_hand_connections(ax, hand_landmarks, color):
#     """
#     Plot connections between hand landmarks (MediaPipe format).
    
#     MediaPipe hand has 21 landmarks:
#     - 0: Wrist
#     - 1-4: Thumb
#     - 5-8: Index finger
#     - 9-12: Middle finger
#     - 13-16: Ring finger
#     - 17-20: Pinky
#     """
#     if len(hand_landmarks) == 21:
#         # Connect palm (0) to finger bases (1, 5, 9, 13, 17)
#         for finger_base in [1, 5, 9, 13, 17]:
#             ax.plot([hand_landmarks[0, 0], hand_landmarks[finger_base, 0]], 
#                     [hand_landmarks[0, 1], hand_landmarks[finger_base, 1]], 
#                     [hand_landmarks[0, 2], hand_landmarks[finger_base, 2]], 
#                     color=color, linewidth=1)
        
#         # Connect each finger joint
#         for finger in [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], 
#                     [13, 14, 15, 16], [17, 18, 19, 20]]:
#             connect_points(ax, hand_landmarks, finger, color)
            
#             # Make fingertips more prominent
#             ax.scatter(hand_landmarks[finger[-1], 0], 
#                     hand_landmarks[finger[-1], 1], 
#                     hand_landmarks[finger[-1], 2], 
#                     color=color, s=30, edgecolor='black')


def save_results_to_file(query_word, candidate_video, is_present, start_idx, score, threshold, method, feature_type):
    """Save detection results to a file"""
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    
    output_file = os.path.join(result_dir, f"detection_{query_word}_{candidate_video}.txt")
    
    with open(output_file, "w") as f:
        f.write(f"Sign Language Word Detection Results\n")
        f.write(f"==================================\n\n")
        f.write(f"Query word: {query_word}\n")
        f.write(f"Candidate video: {candidate_video}\n")
        f.write(f"Similarity method: {method}\n")
        f.write(f"Feature type: {feature_type}\n\n")
        f.write(f"Start frame: {start_idx}\n")
        f.write(f"Similarity score: {score:.4f}\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        f.write(f"Detection result: {'PRESENT' if is_present else 'NOT PRESENT'}\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    print("Sign Language Word Detection - Arm Features Focus")
    print("===============================================")
    
    # Prompt user to choose similarity method
    print("\nSelect similarity method:")
    print("1. DTW (Dynamic Time Warping)")
    print("2. Cosine Similarity")
    print("3. Euclidean Distance")
    print("4. Correlation")
    
    method_choice = input("Enter your choice (1-4): ").strip()
    method_map = {
        "1": "dtw",
        "2": "cosine",
        "3": "euclidean",
        "4": "correlation"
    }
    similarity_method = method_map.get(method_choice, "dtw")
    
    # Set feature type to "arm" by default for arm-focused detection
    feature_type = "arm"
    print(f"\nUsing {feature_type} features for detection")
    
    # Choose mode
    print("\nSelect mode:")
    print("1. Single query")
    print("2. Evaluation over CSV mapping")
    print("3. Threshold tuning with negative examples")
    print("4. Top 5/10:")
    
    mode = input("Enter your choice (1-4): ").strip()
    
    if mode == '2':
        mapping_csv_path = "E:/ISH/updated_main.csv"
        print(f"\nEvaluating using {similarity_method} similarity and {feature_type} features...\n")
        evaluate_mapping(mapping_csv_path, similarity_method, feature_type)

    elif mode == '4':
        mapping_csv_path = "E:/ISH/updated_main.csv"
        try:
            # Get top accuracy metrics
            avg_top5, avg_top10 = evaluate_batch_retrieval(
                mapping_csv_path,
                similarity_method="dtw",
                feature_type="arm"
            )

        except Exception as e:
            print(f"\nError during evaluation: {str(e)}")

    elif mode == '3':
        print("\nThreshold tuning with negative examples...")
        # Specify known non-matching word-sentence pairs
        word_paths = []
        sentence_paths = []
        
        # Get word paths from user
        num_words = int(input("Enter number of word paths: "))
        for i in range(num_words):
            word_gloss = input(f"Enter word gloss {i+1}: ")
            word_paths.append(os.path.join(WORDS_FOLDER, f"{word_gloss}.pose"))
        
        # Get sentence paths from user
        num_sentences = int(input("Enter number of sentence paths: "))
        for i in range(num_sentences):
            sentence_uid = input(f"Enter sentence UID {i+1}: ")
            sentence_paths.append(os.path.join(SENTENCES_FOLDER, f"{sentence_uid}.pose"))
        
        #evaluate_negative_examples(word_paths, sentence_paths, similarity_method, feature_type)
    
    else:  # mode == '1'
        query_word_gloss = input("Enter the query word gloss: ").strip()
        candidate_video_uid = input("Enter the candidate sentence video UID: ").strip()
        
        query_pose_path = os.path.join(WORDS_FOLDER, f"{query_word_gloss}.pose")
        candidate_pose_path = os.path.join(SENTENCES_FOLDER, f"{candidate_video_uid}.pose")
        
        try:
            is_present, results, combined_score, threshold = detect_sign_word(
                query_pose_path, candidate_pose_path, similarity_method, feature_type
            )
            
            print(f"\nResults for '{query_word_gloss}' in '{candidate_video_uid}':")
            
            for feat_type, (start_idx, score) in results.items():
                print(f"{feat_type.capitalize()}: Start frame = {start_idx}, Score = {score:.4f}")
            
            print(f"\nCombined score: {combined_score:.4f} (Threshold: {threshold:.4f})")
            print(f"Prediction: The word is {'PRESENT' if is_present else 'NOT PRESENT'} in the sentence.")
            
            # Ask if user wants to visualize or save results
            visualize = input("\nVisualize results? (y/n): ").strip().lower() == 'y'
            save = input("Save results to file? (y/n): ").strip().lower() == 'y'
            
            if visualize:
                word_pose = load_pose(query_pose_path)
                sentence_pose = load_pose(candidate_pose_path)
                start_frame = next(iter(results.values()))[0]  # Get start frame from first result
                visualize_arm_hand_connected(word_pose, sentence_pose, start_frame,similarity_method)
            
            if save:
                start_frame = next(iter(results.values()))[0]  # Get start frame from first result
                save_results_to_file(
                    query_word_gloss, 
                    candidate_video_uid, 
                    is_present, 
                    start_frame, 
                    combined_score, 
                    threshold,
                    similarity_method,
                    feature_type
                )
        
        except Exception as e:
            print(f"Error processing files:\n{e}")
    
    print("\nSign Language Word Detection completed.")