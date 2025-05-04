from config import *
from alerts import *
from utils import *
import threading

def run_full_pipeline(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video source!")
        return
    
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live = isinstance(video_path, int)


    if not output_path:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"output_{ts}.mp4"
        
    out = None
    if not is_live and output_path:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    
    buffer_duration = 1.5  # seconds
    frame_buffer = deque(maxlen=int(fps * buffer_duration))
    prediction_buffer = deque(maxlen=3)
    accident_saved_ids = set()
    accident_event_sent = False
    accident_free_counter = 0
    frame_idx = 0
    accident_type, severity = None, None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLOv8 COCO detection
        coco_result = coco_model.predict(frame, imgsz=416, conf=0.4, verbose=False)[0]
        detections = []
        for box in coco_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())
            if cls in [0, 2, 3, 5, 7]:  # person, car, motorcycle, bus, truck
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, conf, cls))
        tracks = tracker.update_tracks(detections, frame=frame)

        # Accident detection
        acc_result = accident_model.predict(frame, imgsz=416, conf=0.4, verbose=False)[0]
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

        # ConvLSTM Prediction
        
        if accident_detected and frame_idx > 9 and not accident_event_sent:
            seq = extract_sequence(cap, frame_idx)
            pred = convlstm_model.predict(seq)
            accident_type = labels[np.argmax(pred)]
            severity = severity_levels.get(accident_type, "Unknown")
            prediction_buffer.append(accident_type)
            
            # Check if 3 consecutive predictions are the same
            if len(prediction_buffer) == 3 and len(set(prediction_buffer)) == 1:

                if severity in ["High", "Critical",]:  # Only send for serious types
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        if track_id in accident_saved_ids:
                            continue  # Skip duplicate alert

                        l, t, r, b = map(int, track.to_ltrb())
                        class_name = "object"
                        if hasattr(track, "det_class") and track.det_class is not None:
                            try:
                                class_name = coco_classes[track.det_class]
                            except IndexError:
                                pass

                        clip_path = save_fullframe_clip(
                            video_path=video_path,
                            pre_buffer_frames=list(frame_buffer),
                            start_frame_idx=frame_idx,
                            accident_type=accident_type,
                            fps=fps,
                            width=width,
                            height=height
                        )
                        def alert_thread():
                            # print("Email Sent!")
                            # print("SMS Sent!")
                            send_email_alert(
                                subject=f"🚨 {accident_type} ({severity}) Detected",
                                body=f"Accident Type: {accident_type}\nSeverity: {severity}\nObject: {class_name} (ID {track_id})\nFrame: {frame_idx}",
                                to_email="diarysilk68@gmail.com",
                                video_path=clip_path
                            )

                            send_sms_alert(
                                track_id=track_id,
                                frame_idx=frame_idx,
                                accident_type=accident_type,
                                severity=severity,
                                class_name=class_name
                            )

                        threading.Thread(target=alert_thread).start()
                        accident_saved_ids.add(track_id)
                        accident_event_sent = True  # Stop further alerts for this event

        # Draw Tracks
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
            # if accident_detected and accident_type:
            #     label += f" - {accident_type}"
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status Text
        status_text = f"No Accident - {conf:.2f}" if not accident_detected else f"Accident - {accident_type} ({severity}) - {conf:.2f}"
        status_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        # Logging
        if accident_detected and accident_type is not None:
            accident_free_counter = 0  # reset countdown
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                bbox = f"{l},{t},{r},{b}"
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_idx, track_id, accident_type or "unknown", severity, bbox])
        else:
            accident_free_counter += 1
            if accident_free_counter > 100:
                accident_event_sent = False  # Re-arm for next incident

        # Write frame
        frame_buffer.append(frame.copy())
        if out:
            out.write(frame)
        
        frame_idx += 1

        # Real-time display if live input
        if is_live:
            cv2.imshow("Real-Time Dashboard", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    # Cleanup after loop
    cap.release()
    if out:
        out.release()
    if is_live:
        cv2.destroyAllWindows()

    print("Full pipeline completed......................!")