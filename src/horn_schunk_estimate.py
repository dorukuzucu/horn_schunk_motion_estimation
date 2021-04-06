import cv2
import numpy as np

from src.image.utils import rgb2gray
from src.video.video_reader import AviReader
from src.image.utils import uv_2_motion_vector
from src.image.utils import draw_motion_vectors
from src.image.horn_schunk import HornSchunkFrame


class HS_Estimate:
    def __init__(self,alpha,num_iter,video_path=None):
        if video_path is None:
            raise Exception("Please enter path for video")

        self.video_path = video_path
        self.alpha = alpha
        self.num_iter = num_iter

        self.reader = None
        self.num_frames = None
        self.hs_estimator = None

        self.motion_vectors = []
        self.matched_frames = []
        self.vector_added_frames = []
        self.vector_field_frames = []
        self.loss = []

        self.reset_state(video_path,alpha=alpha,num_iter=num_iter)

    def _init_frames(self,path=None):
        if path is None:
            path = self.video_path
        self.reader = AviReader(path, mode="np")
        self.num_frames = len(self.reader)

    def _init_hs_estimator(self,alpha=None,num_iter=None):
        if alpha is None:
            alpha = self.alpha
        if num_iter is None:
            num_iter = self.num_iter
        self.hs_estimator = HornSchunkFrame(alpha=alpha, num_iter=num_iter)

    def reset_state(self,path=None,alpha=None,num_iter=None):
        self._init_frames(path=path)
        self._init_hs_estimator(alpha=alpha,num_iter=num_iter)

    def match_with_original(self):
        for frame in range(self.num_frames-1):
            print("Working on frame:{}".format(frame + 1))
            anchor_frame = self.reader[frame]
            next_frame = self.reader[frame+1]

            anchor_frame = rgb2gray(anchor_frame)
            next_frame = rgb2gray(next_frame)

            u, v = self.hs_estimator(anchor_frame, next_frame)


            matched_img = block_swap(block_pairs=matched_pairs, anchor=anchor_frame)
            vector_image = np.copy(matched_img)



            vector_image = np.copy(matched_img)

            vectors = uv_2_motion_vector(u=u,v=v)
            vector_image = draw_motion_vectors(image=vector_image, u=u,v=v, color=(255, 255, 255), thickness=1)

            self.motion_vectors.append(vectors)
            self.matched_frames.append(matched_img)
            self.vector_added_frames.append(vector_image)
            self.loss.append(image_mse(img1=matched_img, img2=next_frame))

    def save_results(self,save_path):
        save_path = "../"+save_path+"/"
        len_results = len(self.matched_frames)

        size = (self.matched_frames[0].shape[1],self.matched_frames[0].shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(save_path+"output_video.avi",fourcc,20.0,size,0)

        for res in range(len_results):

            frame = self.matched_frames[res]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)

            cv2.imwrite(save_path + "matched_frame_" + str(res) + ".jpg", self.matched_frames[res])
            cv2.imwrite(save_path+"vector_frame_"+str(res)+".jpg",self.vector_added_frames[res])

        video_writer.release()

        with open(save_path+"loss.txt", "w") as file:
            file.write(str(self.loss)+"\n")
            file.close()

        with open(save_path+"vectors.txt", "w") as file:
            for im_idx,im_vectors in enumerate(self.motion_vectors):
                for vec_idx,vector in enumerate(im_vectors):
                    print("Writing Frame:{} Vector:{}".format(im_idx,vec_idx))
                    file.write("Frame No:{} Vector No:{} Start:{} End:{}".format(im_idx,vec_idx,vector.start,vector.end))
                    file.write("\n")
            file.close()