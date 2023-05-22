import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import (
    _normalized_to_pixel_coordinates as denormalize_coordinates,
)


def get_face_mesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(
                lm.x, lm.y, frame_width, frame_height
            )
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(
    landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h
):
    left_ear, left_lm_coordinates = get_ear(
        landmarks, left_eye_idxs, image_w, image_h
    )
    right_ear, right_lm_coordinates = get_ear(
        landmarks, right_eye_idxs, image_w, image_h
    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(
    frame, left_lm_coordinates, right_lm_coordinates, color
):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    # frame = cv2.flip(frame, 1)
    return frame


def plot_text(
    frame,
    text,
    origin,
    color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    fntScale=0.8,
    thickness=2,
):
    frame = cv2.putText(
        frame, text, origin, font, fntScale, color, thickness
    )
    return frame


class VideoFrameHandler:
    def __init__(self):
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        self.RED_COLOR = (0, 0, 255)
        self.GREEN_COLOR = (0, 255, 0)
        self.BLUE_COLOR = (255, 0, 0)
        self.SKIN_COLOR = (172, 190, 232)

        self.facemesh_model = get_face_mesh()

        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DOZEOFF_TIME": 0.0,
            "HEAD_TIME": 0.0,
            "COLOR": self.GREEN_COLOR,
            "play_alarm": False,
        }

    def process(self, frame: np.array, thresholds: dict):
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = self.facemesh_model.process(frame)

        VERTICAL_OFFSET = 30
        DOZEOFF_TIME_TEXT_POSN = (10, int(frame_h // 2 * 1.9))
        HEADTIME_TEXT_POSN = (
            DOZEOFF_TIME_TEXT_POSN[0],
            DOZEOFF_TIME_TEXT_POSN[1] - VERTICAL_OFFSET,
        )
        HEADPOSN_TEXT_POSN = (
            HEADTIME_TEXT_POSN[0],
            HEADTIME_TEXT_POSN[1] - VERTICAL_OFFSET,
        )
        ALARM_TEXT_POSN = (200, int(frame_h // 2 * 1.0))

        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            # NOTE: calculation for dozeoff, returns EAR and coordinates
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(
                landmarks,
                self.eye_idxs["left"],
                self.eye_idxs["right"],
                frame_w,
                frame_h,
            )

            # NOTE: calculation for headposition, returns x and y
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if (
                        idx == 33
                        or idx == 263
                        or idx == 1
                        or idx == 61
                        or idx == 291
                        or idx == 199
                    ):
                        if idx == 1:
                            nose_2d = (lm.x * frame_w, lm.y * frame_h)

                        x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * frame_w
                cam_matrix = np.array(
                    [
                        [focal_length, 0, frame_h / 2],
                        [0, focal_length, frame_w / 2],
                        [0, 0, 1],
                    ]
                )
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360

            # NOTE: check if eye if closing EAR < threshold OR HEADPOSN not right
            if EAR < thresholds["EAR_THRESH"]:
                end_doze_timestamp = time.perf_counter()

                self.state_tracker["DOZEOFF_TIME"] += (
                    end_doze_timestamp - self.state_tracker["start_time"]
                )
                self.state_tracker[
                    "start_time"
                ] = end_doze_timestamp  # ; for next time
                self.state_tracker["COLOR"] = self.RED_COLOR

                if (
                    self.state_tracker["DOZEOFF_TIME"]
                    >= thresholds["WAIT_DOZEOFF_TIME"]
                ):
                    self.state_tracker["play_alarm"] = True
                    plot_text(
                        frame,
                        "PLEASE WAKE UP!!",
                        ALARM_TEXT_POSN,
                        self.state_tracker["COLOR"],
                    )

            elif (
                y < -thresholds["LEFT_THRESH"]
                or y > thresholds["RIGHT_THRESH"]
                or x < -thresholds["DOWN_THRESH"]
                or x > thresholds["UP_THRESH"]
            ):
                end_head_timestamp = time.perf_counter()

                self.state_tracker["HEAD_TIME"] += (
                    end_head_timestamp - self.state_tracker["start_time"]
                )
                self.state_tracker[
                    "start_time"
                ] = end_head_timestamp  # ; for next time
                self.state_tracker["COLOR"] = self.RED_COLOR

                if (
                    self.state_tracker["HEAD_TIME"]
                    >= thresholds["WAIT_HEADPOSN_TIME"]
                ):
                    self.state_tracker["play_alarm"] = True
                    plot_text(
                        frame,
                        "FOCUS ON THE ROAD!!",
                        ALARM_TEXT_POSN,
                        self.state_tracker["COLOR"],
                    )

            # NOTE: or eye is still open EAR > 0.18
            # DANGER: RESET EVERYTHING in state_tracker
            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DOZEOFF_TIME"] = 0.0
                self.state_tracker["HEAD_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN_COLOR
                self.state_tracker["play_alarm"] = False

            # NOTE: storing headpositions for plotting
            if y < -thresholds["LEFT_THRESH"]:
                headposn = "Looking Left"
            elif y > thresholds["RIGHT_THRESH"]:
                headposn = "Looking Right"
            elif x < -thresholds["DOWN_THRESH"]:
                headposn = "Looking Down"
            elif x > thresholds["UP_THRESH"]:
                headposn = "Looking Up"
            else:
                headposn = "Forward"

            # NOTE: plotting everything
            HEADPOSN_text = f"HEAD POSITION: {headposn}"
            DOZEOFF_TIME_text = f"DOZEOFF TIME: {round(self.state_tracker['DOZEOFF_TIME'], 3)} secs"
            HEAD_TIME_text = f"HEAD TIME: {round(self.state_tracker['HEAD_TIME'], 3)} secs"

            plot_text(
                frame,
                HEADPOSN_text,
                HEADPOSN_TEXT_POSN,
                self.state_tracker["COLOR"],
            )
            plot_text(
                frame,
                DOZEOFF_TIME_text,
                DOZEOFF_TIME_TEXT_POSN,
                self.state_tracker["COLOR"],
            )
            plot_text(
                frame,
                HEAD_TIME_text,
                HEADTIME_TEXT_POSN,
                self.state_tracker["COLOR"],
            )

            nose_pt1 = (int(nose_2d[0]), int(nose_2d[1]))
            nose_pt2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(
                frame,
                nose_pt1,
                nose_pt2,
                color=self.state_tracker["COLOR"],
                thickness=2,
            )

            frame = plot_eye_landmarks(
                frame,
                coordinates[0],
                coordinates[1],
                self.state_tracker["COLOR"],
            )

        else:
            # NOTE: when facemesh is not found do this, here i can add when drive is not present on seat
            # DANGER: RESET EVERYTHING in state_tracker +
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DOZEOFF_TIME"] = 0.0
            self.state_tracker["HEAD_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN_COLOR
            self.state_tracker["play_alarm"] = False

        return frame, self.state_tracker["play_alarm"]
