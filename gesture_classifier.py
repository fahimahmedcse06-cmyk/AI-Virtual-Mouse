# =============================================================================
# gesture_classifier.py — Optimized Gesture Classification
# =============================================================================

import time
import math
import config


class GestureClassifier:

    def __init__(self):
        self._left_latched = False
        self._right_latched = False
        self._last_lclick_time = 0.0
        self._last_rclick_time = 0.0
        print("✅ Optimized GestureClassifier ready")

    def classify(self, landmarks):
        if landmarks is None:
            self._left_latched = False
            self._right_latched = False
            return "NONE"

        now = time.monotonic()

        left_dist = self._dist(
            landmarks, config.THUMB_TIP, config.INDEX_TIP
        )
        right_dist = self._dist(
            landmarks, config.THUMB_TIP, config.MID_TIP
        )

        middle_up = (
            landmarks[config.MID_TIP].y <
            landmarks[config.MID_PIP].y
        )

        index_up = (
            landmarks[config.INDEX_TIP].y <
            landmarks[config.INDEX_PIP].y
        )

        # Hysteresis: require a wider release distance to prevent jitter.
        left_release = config.LCLICK_THRESHOLD * 1.35
        right_release = config.RCLICK_THRESHOLD * 1.35

        left_pose = not middle_up and left_dist < config.LCLICK_THRESHOLD

        right_pose = (
            not index_up
            and left_dist >= config.LCLICK_THRESHOLD
            and right_dist < config.RCLICK_THRESHOLD
        )

        # Re-arm only after the pinch is clearly released.
        if middle_up or left_dist >= left_release:
            self._left_latched = False

        if (
            index_up
            or left_dist < config.LCLICK_THRESHOLD
            or right_dist >= right_release
        ):
            self._right_latched = False

        # FIST has highest priority.
        if self._is_fist(landmarks):
            return "FIST"

        # One pinch = one click.
        if left_pose and not self._left_latched:
            self._left_latched = True
            self._last_lclick_time = now
            return "LEFT_CLICK"

        if right_pose and not self._right_latched:
            self._right_latched = True
            self._last_rclick_time = now
            return "RIGHT_CLICK"

        if self._is_scrolling(landmarks):
            return "SCROLL"

        if self._is_pointing(landmarks):
            return "POINT"

        return "NONE"

    # =========================================================================
    # GESTURE DETECTORS
    # =========================================================================

    def _is_pointing(self, lm):
        return (
            lm[config.INDEX_TIP].y < lm[config.INDEX_PIP].y
            and lm[config.MID_TIP].y > lm[config.MID_PIP].y
            and lm[config.RING_TIP].y > lm[config.RING_PIP].y
        )

    def _is_scrolling(self, lm):
        return (
            lm[config.INDEX_TIP].y < lm[config.INDEX_PIP].y
            and lm[config.MID_TIP].y < lm[config.MID_PIP].y
            and lm[config.RING_TIP].y > lm[config.RING_PIP].y
            and lm[config.PINKY_TIP].y > lm[config.PINKY_PIP].y
        )

    def _is_fist(self, lm):
        return (
            lm[config.INDEX_TIP].y > lm[config.INDEX_PIP].y
            and lm[config.MID_TIP].y > lm[config.MID_PIP].y
            and lm[config.RING_TIP].y > lm[config.RING_PIP].y
            and lm[config.PINKY_TIP].y > lm[config.PINKY_PIP].y
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _dist(lm, a, b):
        dx = lm[a].x - lm[b].x
        dy = lm[a].y - lm[b].y
        return math.hypot(dx, dy)

    def get_debug_info(self, lm):
        if lm is None:
            return {}

        return {
            "index_up": lm[config.INDEX_TIP].y < lm[config.INDEX_PIP].y,
            "middle_up": lm[config.MID_TIP].y < lm[config.MID_PIP].y,
            "L-dist": round(self._dist(lm, config.THUMB_TIP, config.INDEX_TIP), 3),
            "R-dist": round(self._dist(lm, config.THUMB_TIP, config.MID_TIP), 3),
            "L-thresh": config.LCLICK_THRESHOLD,
            "R-thresh": config.RCLICK_THRESHOLD,
        }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import cv2
    from hand_detector import HandDetector

    print("=" * 58)
    print("  GESTURE TEST")
    print("  Pinch once = one click; release before clicking again.")
    print("  Press Q to quit.")
    print("=" * 58)

    detector = HandDetector()
    classifier = GestureClassifier()
    prev_time = time.monotonic()

    COLOURS = {
        "POINT": (0, 220, 0),
        "SCROLL": (0, 200, 255),
        "LEFT_CLICK": (0, 165, 255),
        "RIGHT_CLICK": (255, 100, 50),
        "FIST": (0, 0, 220),
        "NONE": (150, 150, 150),
    }

    try:
        while True:
            landmarks, frame, hand_found = detector.get_landmarks(draw=True)

            curr_time = time.monotonic()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            h, w = frame.shape[:2]
            gesture = classifier.classify(landmarks)
            colour = COLOURS.get(gesture, config.HUD_WHITE)

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.HUD_YELLOW, 2
            )

            lbl_w = cv2.getTextSize(
                gesture, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3
            )[0][0]

            cv2.putText(
                frame, gesture, ((w - lbl_w) // 2, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3
            )

            if hand_found:
                dbg = classifier.get_debug_info(landmarks)

                ld_col = (
                    config.HUD_RED
                    if dbg["L-dist"] < dbg["L-thresh"]
                    else config.HUD_WHITE
                )
                rd_col = (
                    config.HUD_RED
                    if dbg["R-dist"] < dbg["R-thresh"]
                    else config.HUD_WHITE
                )

                panel = [
                    (f"index_up  : {dbg['index_up']}", config.HUD_WHITE),
                    (f"middle_up : {dbg['middle_up']}", config.HUD_WHITE),
                    (f"L-dist    : {dbg['L-dist']}", ld_col),
                    (f"L-thresh  : {dbg['L-thresh']}", config.HUD_WHITE),
                    (f"R-dist    : {dbg['R-dist']}", rd_col),
                    (f"R-thresh  : {dbg['R-thresh']}", config.HUD_WHITE),
                ]

                for i, (line, col) in enumerate(panel):
                    cv2.putText(
                        frame, line, (w - 270, 30 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1
                    )
            else:
                cv2.putText(
                    frame, "No hand detected", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.HUD_RED, 2
                )

            cv2.putText(
                frame, "Q to quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.HUD_WHITE, 1
            )

            cv2.imshow(config.WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.release()
        cv2.destroyAllWindows()
