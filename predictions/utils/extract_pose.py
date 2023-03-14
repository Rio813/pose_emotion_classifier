import cv2
import mediapipe as mp
import numpy as np
import h5py
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("pos/FormatFactoryPart1.mp4")

# Create an empty numpy array for storing pose data
pose_data = np.empty((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 48), dtype=float)



with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    frame_count = 0

    while cap.isOpened():
        # Get the next frame
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the image to RGB format and process it with MediaPipe Pose
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert the image back to BGR format and draw the pose landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        # 这是下列16个人体关键部位坐标
        # 根坐标暂时置零
        root = [0.0, 0.0, 0.0]
        # 脊柱坐标可以目前采用取平均值（12.right_shoulder, 11.left_shoulder, 24.right_hip, 23.left_hip)
        #spine = [0.0, 0.0, 0.0]
        spine_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x + \
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x
                   ) / 4

        spine_y =(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
                  ) / 4

        spine_z =(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z
                  ) / 4

        spine = [spine_x, spine_y, spine_z]
        # 脖子坐标可以目前采用取平均值（9.mouth_right, 10.mouth_left, 11.right_shoulder, 12.left_shoulder)
        #neck = [0.0, 0.0, 0.0]
        neck_x =(results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x + \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + \
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                   ) / 4

        neck_y =(results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y + \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + \
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                   ) / 4

        neck_z =(results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].z + \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].z + \
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + \
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
                   ) / 4
        neck = [neck_x, neck_y, neck_z]
        #头部使用nose替代
        head = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z]

        left_shoulder_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z]

        left_elbow_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z]
        #手对应手腕
        left_hand = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z]

        right_shoulder_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
        right_elbow_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z]

        right_hand = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z]

        left_hip_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z]
        left_knee_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z]
        #脚使用脚踝
        left_foot = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z]

        right_hip_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z]

        right_knee_pose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z]
        #脚使用脚踝
        right_foot = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z]

        # Extract and store the 3D coordinates of each pose landmark
        pose_landmarks = np.array([[landmark.x, landmark.y, landmark.z]
                                   for landmark in results.pose_landmarks.landmark]).flatten()
        #合并pose数组
        pose_16 = combined_list = root + spine + neck + head + left_shoulder_pose + left_elbow_pose + left_hand + right_shoulder_pose + \
                                  right_elbow_pose + right_hand + left_hip_pose + left_knee_pose + \
                                  left_foot + right_hip_pose + right_knee_pose + right_foot
        #追加到二维数组
        pose_data[frame_count] = pose_16

        #实时显示
        print(pose_landmarks, end='\r')

        # Display the annotated image
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        # Update the frame count
        frame_count += 1

        # Check if the user has pressed the "ESC" key
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()

#创建HDF5文件保存数据
# 打开HDF5文件并创建一个数据集
with h5py.File('example.h5', 'w') as f:
    dset = f.create_dataset("pose", pose_data.shape, dtype=pose_data.dtype)
    dset[...] = pose_data
    print(f["pose"].shape)
# Print the pose data
print(pose_data.shape)
