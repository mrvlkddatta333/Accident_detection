# ---------------------------
# Frame Sequence for ConvLSTM
# ---------------------------
from config import *
def extract_sequence(cap, current_frame_idx, num_frames=12, size=(64, 64)):
    frame_buffer = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx - num_frames + 1)

    for _ in range(num_frames):
        success, frame = cap.read()
        if not success:
            # Use a black RGB frame if reading fails
            frame = np.zeros((size[0], size[1], 3), dtype=np.float32)
        else:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # Normalize to [0, 1]
        frame_buffer.append(frame)

    # cap.release()
    return np.array(frame_buffer).reshape(1, num_frames, size[0], size[1], 3)

# ---------------------------
# Save Full-Frame Accident Clip (3 seconds: 1.5s before + after)
# ---------------------------
def save_fullframe_clip(video_path, pre_buffer_frames, start_frame_idx, accident_type, fps, width, height):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_sec = int(fps * 1.5)
    start = max(0, start_frame_idx - half_sec)
    end = min(total_frames, start_frame_idx + half_sec)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(clip_output_dir, f"Accident_{accident_type}_{timestamp}_f{start_frame_idx}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in pre_buffer_frames:
        out.write(f)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx + 1)
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    # cap.release()
    print(f"Saved full-frame accident clip: {out_path}")
    return out_path  # Optional: return path for alerts


# ---------------------------
# Optional: Save Cropped Object Clip (per track)
# ---------------------------
def save_cropped_clip_by_track(video_path, start_frame, bbox, track_id, label, fps=30, clip_len_sec=3):
    cap = cv2.VideoCapture(video_path)
    half_clip = int((fps * clip_len_sec) // 2)
    start = max(0, start_frame - half_clip)
    end = start_frame + half_clip

    x1, y1, x2, y2 = map(int, bbox)
    out_path = os.path.join(clip_output_dir, f"ID_{track_id}_{label}_frame{start_frame}.mp4")

    out = None
    current = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start <= current <= end:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (224, 224))

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, fps, (224, 224))

            out.write(crop)

        current += 1
        if current > end:
            break

    cap.release()
    if out:
        out.release()
        print("Cropped clip saved:", out_path)
        return out_path
