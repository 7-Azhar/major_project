def recommend(telemetry, video, mode):
    """Observation rules; proposals never issue a flight command directly."""
    if video.get("age_s") is None or video["age_s"] > 1:
        return {"action": None, "reason": "Video unavailable; use pilot observation"}
    if telemetry.get("battery_percent") is not None and telemetry["battery_percent"] < 25:
        return {"action": None, "reason": "Low battery: pilot should assess return or landing"}
    people = [d for d in video.get("detections", []) if d["label"] == "person"]
    if people:
        return {"action": "hold", "reason": f"{len(people)} possible person(s): hold for operator verification",
                "source_frame": video.get("detection_sequence"), "mode": mode}
    return {"action": None, "reason": "Continue observation; no verified rescue lead"}
