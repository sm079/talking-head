import cv2
import sys
import torch
import mimetypes
import contextlib
import numpy as np

from tqdm import tqdm
from pathlib import Path
from numpy.linalg import norm
from skimage import transform as trans
from insightface.app import FaceAnalysis


def is_video(path):
    return "video" in mimetypes.guess_type(path)[0] if mimetypes.guess_type(path)[0] else False


def is_image(path):
    return "image" in mimetypes.guess_type(path)[0] if mimetypes.guess_type(path)[0] else False


def cosine_similarity(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim


def estimate_norm(landmarks, image_size=512, zoom_out_factor=1.1):

    arcface_dst = np.array([
        [38.2946, 51.6963], 
        [73.5318, 51.5014], 
        [56.0252, 71.7366],
        [41.5493, 92.3655], 
        [70.7299, 92.2041]
        ], dtype=np.float32)
    
    assert landmarks.shape == (5, 2)
    assert image_size % 128 == 0

    ratio = float(image_size) / 128.0
    diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x

    dst_center = np.mean(dst, axis=0)
    dst = (dst - dst_center) * zoom_out_factor + dst_center

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    return tform.params[0:2, :]


def norm_crop(image, landmarks, image_size=512):
    return cv2.warpAffine(
        src=image, 
        M=estimate_norm(landmarks, image_size), 
        dsize=(image_size, image_size), 
        borderValue=0.0)


@contextlib.contextmanager
def suppress_output():
    original_stdout = sys.stdout
    sys.stdout = open('nul', 'w')
    yield
    sys.stdout = original_stdout


class FaceDetector:
    def __init__(self, models_dir):
        with suppress_output():
            self.app = FaceAnalysis(
                root = models_dir, 
                allowed_modules = ["detection", "recognition"], 
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=0, det_size=(640, 480))
        

    def get_embeds_from_image(self, image_path):
        image = cv2.imread(image_path)
        face  = self.app.get(image, max_num=1)[0]
        return face["embedding"] if face else []
    

    def get_mean_embeds_from_folder(self, folder_path):
        paths = Path(folder_path).iterdir()
        return np.mean([self.get_embeds_from_image(str(path)) for path in paths if is_image(str(path))], axis=0)
    

    def save_faces_from_image(
        self,
        image: np.ndarray, 
        output_path: str, 
        reference_embedding = None,
        max_faces:int = 4,
        similarity_threshold: float = 0.6,
        min_face_size: int = 400,
        image_size: int = 512
    ):

        for i, face in enumerate(self.app.get(image, max_num=max_faces)):
            if min_face_size > 0:
                face_height = face["bbox"][3] - face["bbox"][1]
                if face_height < min_face_size:
                    return

            aligned_image = norm_crop(image, face["kps"], image_size=image_size)
            if reference_embedding is not None:
                similarity_score = cosine_similarity(reference_embedding, face["embedding"])
                if similarity_score > similarity_threshold:
                    cv2.imwrite(f"{output_path}_{i:02d}.jpg", aligned_image)
            else:
                # cv2.imwrite(f"{output_path}_{i:02d}.jpg", aligned_image)
                cv2.imwrite(f"{output_path}.jpg", aligned_image)


    def save_faces_from_video(
        self,
        video_path: str, 
        fps: int,
        output_folder: str,
        reference_embedding = None, 
        max_faces_per_frame:int = 4,
        similarity_threshold: float = 0.6,
        min_face_size: int = 400,
        image_size: int = 512
    ):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Couldn't open the video file.")
            exit()

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_factor = int(round(video_fps / fps))

        print(f"\nVideo: [{Path(video_path).name}] | Extracting at {fps} fps")

        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break

                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.save_faces_from_image(
                    image=frame, 
                    output_path=f"{output_folder}/{Path(video_path).stem}_{frame_number:06d}",
                    reference_embedding=reference_embedding,
                    max_faces=max_faces_per_frame,
                    similarity_threshold=similarity_threshold,
                    min_face_size=min_face_size,
                    image_size=image_size
                )

                # Skip frames based on the skip factor
                for _ in range(skip_factor - 1):
                    ret, _ = cap.read()
                    pbar.update()
                pbar.update()

        cap.release()


if __name__ == "__main__":
    face_detector = FaceDetector(models_dir="./src")

    reference_embedding = None
    # reference_folder = r"experiments\ref"
    # reference_embedding = face_detector.get_mean_embeds_from_folder(reference_folder)

    input_folder  = r"experiments/001/src/source"
    output_folder = r"experiments/001/src/faces"
    video_fps = 3 # extract faces at this fps
    max_faces_per_frame = 1
    similarity_threshold = 0.6
    min_face_size = 200
    saved_image_size = 512

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(input_folder)
    files = list(Path(input_folder).iterdir())

    for file in tqdm(files):
        if is_image(str(file)):
            face_detector.save_faces_from_image(
                image=cv2.imread(str(file)), 
                output_path=f"{output_folder}/{file.stem}",
                reference_embedding=reference_embedding, 
                max_faces=max_faces_per_frame,
                similarity_threshold=similarity_threshold,
                min_face_size=min_face_size,
                image_size=saved_image_size
            )

        if is_video(str(file)):
            face_detector.save_faces_from_video(
                    video_path=str(file),
                    fps=video_fps,
                    output_folder=output_folder,
                    reference_embedding=reference_embedding, 
                    max_faces_per_frame=max_faces_per_frame,
                    similarity_threshold=similarity_threshold,
                    min_face_size=min_face_size,
                    image_size=saved_image_size
                )