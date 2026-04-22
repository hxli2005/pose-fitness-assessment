class RepCounter:
    """
    A robust repetition counter based on key-angle peak detection and hysteresis logic.
    State Machine approach for smooth and jitter-free count increments.

    Hysteresis works by defining an "entry" threshold (e.g. knee angle < 100 deg starts a squat)
    and an "exit" threshold (e.g. knee angle > 140 deg finishes a squat).
    """
    def __init__(self, action_type="squat"):
        self.action_type = action_type.lower()
        self.count = 0
        self.in_rep = False      # Is currently inside the active phase of movement
        self.current_rep = []    # Store angles for the ongoing rep to evaluate its quality later

        # Defaults for action types
        if self.action_type == "squat":
            self.track_angle_keys = ['left_knee', 'right_knee']
            self.entry_threshold = 100 # Degree below which a squat is considered "started"
            self.exit_threshold = 140  # Degree above which a squat is considered "finished" (standing up)
            self.is_entry_below = True # Because for a squat, angle DECREASES as you go down

        elif self.action_type == "pushup":
            self.track_angle_keys = ['left_elbow', 'right_elbow']
            self.entry_threshold = 100
            self.exit_threshold = 150
            self.is_entry_below = True # Arm bends down

        elif self.action_type == "crunch":
            self.track_angle_keys = ['left_hip', 'right_hip'] # Tracking torso flex
            self.entry_threshold = 70  # Actually crunches are about compressing torso
            self.exit_threshold = 120
            self.is_entry_below = True
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

    def update(self, angles_dict) -> dict:
        """
        Takes the dictionary of current angles and updates the state machine.
        Returns a dict indicating if a rep just completed, and its trajectory.
        """
        result = {
            "rep_finished": False,
            "count": self.count,
            "trajectory": None
        }

        # Average the tracked angles (e.g., both knees if visible)
        vals = []
        for key in self.track_angle_keys:
            val = angles_dict.get(key)
            if val is not None:
                vals.append(val)

        if not vals:
            return result # No visible keypoints to track currently

        current_val = sum(vals) / len(vals)

        # Logic transitions
        if not self.in_rep:
            # Check if we should START a rep
            started = (current_val < self.entry_threshold) if self.is_entry_below else (current_val > self.entry_threshold)
            if started:
                self.in_rep = True
                self.current_rep = [angles_dict] # Start accumulating states for the new rep
        else:
            # Check if we should FINISH a rep
            self.current_rep.append(angles_dict)

            finished = (current_val > self.exit_threshold) if self.is_entry_below else (current_val < self.exit_threshold)
            if finished:
                self.in_rep = False
                self.count += 1
                result["rep_finished"] = True
                result["count"] = self.count
                result["trajectory"] = self.current_rep # Return full rep history for scorer
                self.current_rep = []

        return result
