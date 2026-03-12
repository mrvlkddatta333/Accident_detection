from config import *
from alerts import *
from utils import *
import threading
import time
from dotenv import load_dotenv

load_dotenv()

def run_pipeline_with_ui(video_source, output_path=None, conf_threshold=0.4, 
                         alert_severities=["Critical", "High"], 
                         enable_email=True, enable_sms=True,
                         frame_callback=None, stats_callback=None,
                         stop_flag_func=None, session_id=None, user_id=None, clips_folder=None):
    """
    Pipeline with UI callback support for real-time updates
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return {"error": "Cannot open video source"}
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live = isinstance(video_source, int) or video_source == 'webcam' or (isinstance(video_source, str) and video_source.startswith('rtsp'))

    # Set output path to user-specific folder if not live
    if not output_path and not is_live:
        if clips_folder:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(clips_folder, f"output_{ts}.mp4")
        else:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"output_{ts}.mp4"
        
    out = None
    if not is_live and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    buffer_duration = 1.5
    frame_buffer = deque(maxlen=int(fps * buffer_duration))
    prediction_buffer = deque(maxlen=3)
    accident_saved_ids = set()
    accident_event_sent = False
    accident_free_counter = 0
    frame_idx = 0
    accident_type, severity = None, None
    
    stats = {"total_detections": 0, "critical_alerts": 0, "fps": 0}

    while cap.isOpened():
        # Check stop flag
        if stop_flag_func and stop_flag_func():
            break
            
        success, frame = cap.read()
        if not success:
            break

        start_time = time.time()

        # COCO detection
        coco_result = coco_model.predict(frame, imgsz=416, conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in coco_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())
            if cls in [0, 2, 3, 5, 7]:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, conf, cls))
        tracks = tracker.update_tracks(detections, frame=frame)

        # Accident detection
        acc_result = accident_model.predict(frame, imgsz=416, conf=conf_threshold, verbose=False)[0]
        accident_detected = any(int(box.cls[0].cpu().item()) == 1 for box in acc_result.boxes)

        # Draw accident boxes
        for box in acc_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            label = "Accident" if cls_id == 1 else "Non-Accident"
            color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ConvLSTM prediction
        if accident_detected and frame_idx > 9 and not accident_event_sent:
            seq = extract_sequence(cap, frame_idx)
            pred = convlstm_model.predict(seq, verbose=0)
            accident_type = labels[np.argmax(pred)]
            severity = severity_levels.get(accident_type, "Unknown")
            prediction_buffer.append(accident_type)
            
            if len(prediction_buffer) == 3 and len(set(prediction_buffer)) == 1:
                if severity in alert_severities:
                    stats["total_detections"] += 1
                    if severity == "Critical":
                        stats["critical_alerts"] += 1
                    
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        if track_id in accident_saved_ids:
                            continue

                        l, t, r, b = map(int, track.to_ltrb())
                        class_name = "object"
                        if hasattr(track, "det_class") and track.det_class is not None:
                            try:
                                class_name = coco_classes[track.det_class]
                            except IndexError:
                                pass

                        clip_path = save_fullframe_clip(
                            video_path=video_source,
                            pre_buffer_frames=list(frame_buffer),
                            start_frame_idx=frame_idx,
                            accident_type=accident_type,
                            fps=fps,
                            width=width,
                            height=height,
                            output_folder=clips_folder,
                            session_id=session_id
                        )
                        
                        def alert_thread():
                            if enable_email:
                                send_email_alert(
                                    subject=f"🚨 {accident_type} ({severity}) Detected",
                                    body=f"Accident Type: {accident_type}\nSeverity: {severity}\nObject: {class_name} (ID {track_id})\nFrame: {frame_idx}",
                                    to_email=os.getenv("ALERT_RECIPIENT_EMAIL"),
                                    video_path=clip_path
                                )
                            if enable_sms:
                                send_sms_alert(
                                    track_id=track_id,
                                    frame_idx=frame_idx,
                                    accident_type=accident_type,
                                    severity=severity,
                                    class_name=class_name
                                )

                        threading.Thread(target=alert_thread).start()
                        accident_saved_ids.add(track_id)
                        accident_event_sent = True

        # Draw tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            class_name = "object"
            if hasattr(track, "det_class") and track.det_class is not None:
                try:
                    class_name = coco_classes[track.det_class]
                except IndexError:
                    pass

            label = f"ID {track_id} - {class_name}"
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status text
        status_text = f"No Accident - {conf:.2f}" if not accident_detected else f"Accident - {accident_type} ({severity}) - {conf:.2f}"
        status_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        # Logging
        if accident_detected and accident_type is not None:
            accident_free_counter = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                bbox = f"{l},{t},{r},{b}"
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_idx, track_id, accident_type or "unknown", severity, bbox, session_id])
        else:
            accident_free_counter += 1
            if accident_free_counter > 100:
                accident_event_sent = False

        frame_buffer.append(frame.copy())
        if out:
            out.write(frame)
        
        # Calculate FPS
        stats["fps"] = int(1.0 / (time.time() - start_time))
        
        # Callbacks for UI
        if frame_callback:
            frame_callback(frame)
        if stats_callback:
            stats_callback(stats)
        
        frame_idx += 1

        # Don't show window for web UI
        # if is_live:
        #     cv2.imshow("Real-Time Dashboard", frame)
        #     if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        #         break

    cap.release()
    if out:
        out.release()
    # Don't destroy windows for web UI
    # if is_live:
    #     cv2.destroyAllWindows()

    return stats


def run_full_pipeline(video_path, output_path=None):
    """Original pipeline function for backward compatibility"""
    return run_pipeline_with_ui(video_path, output_path)
