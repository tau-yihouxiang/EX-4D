import random
import torch
import torch.utils.data as data
import cv2
import numpy as np
from glob import glob
import os


class FourAnyDataset_OfflineVal(data.Dataset):
    def __init__(self, video_folder, num_frames, height, width, is_train, steps_per_epoch=10):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.steps_per_epoch = steps_per_epoch
        self.is_train = is_train

        self.mask = []
        self.cond = []
        self.video = []
        for folder in video_folder:
            mask_paths = glob(os.path.join(folder, "*/*mask_180.mp4"))
            mask_paths = sorted(mask_paths)
            random.shuffle(mask_paths)

            cond_paths = [mask_path.replace("mask", "render") for mask_path in mask_paths]
            video_paths = [os.path.join(os.path.dirname(mask_path), "gt.mp4") for mask_path in mask_paths]

            self.mask.extend(mask_paths)
            self.cond.extend(cond_paths)
            self.video.extend(video_paths)
        print(f"loading {len(self.mask)} videos")

    def __len__(self):
        return min(self.steps_per_epoch, len(self.mask))

    def __getitem__(self, idx):
        # try:
            if not os.path.exists(self.video[idx]):
                self.video[idx] = self.cond[idx]
            video_cap = cv2.VideoCapture(self.video[idx])
            mask_cap = cv2.VideoCapture(self.mask[idx])
            cond_cap = cv2.VideoCapture(self.cond[idx])
            # Get video frame count
            # video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # mask_frame_count = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cond_frame_count = int(cond_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            input_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            input_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # assert video_frame_count == mask_frame_count, f"Video and mask frame count mismatch: {video_frame_count} != {mask_frame_count}"
            # assert video_frame_count == cond_frame_count, f"Video and cond frame count mismatch: {video_frame_count} != {cond_frame_count}"
            # frame_count = min(video_frame_count, mask_frame_count, cond_frame_count)
            # assert frame_count >= self.num_frames, f"Video length is too short: {video_frame_count} < {self.num_frames}"
            # index = np.arange(self.num_frames)
            index = np.linspace(0, self.num_frames - 1, self.num_frames, endpoint=False, dtype=int)

            video_frames = []
            mask_frames = []
            cond_frames = []
            # Process video frames
            for i in index:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret_video, frame_video = video_cap.read()
                if ret_video:
                    frame_video = cv2.resize(frame_video, (self.width, self.height))
                    frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
                    video_frames.append(frame_video)
            
            if len(video_frames) < self.num_frames:
                # pad video_frames with final frame
                video_frames += [video_frames[-1]] * (self.num_frames - len(video_frames))
            else:
                # trim video_frames to num_frames
                video_frames = video_frames[:self.num_frames]

            video_frames = np.array(video_frames)
            video = torch.from_numpy(video_frames).permute(3, 0, 1, 2)
            video_cap.release()
            video = video.float() / 255
            
            # Process mask frames
            for i in index:
                mask_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret_mask, frame_mask = mask_cap.read()
                if ret_mask:
                    frame_mask = cv2.resize(frame_mask, (self.width, self.height))
                    # if i >= index[0]:
                    kernel = random.randint(2, 8)
                    frame_mask = cv2.erode(frame_mask, np.ones((kernel, kernel), np.uint8), iterations=1)
                    mask_frames.append(frame_mask)
            mask_frames = np.array(mask_frames)
            mask = torch.from_numpy(mask_frames).permute(3, 0, 1, 2).float()
            mask_cap.release()
            mask = (mask / 255.0 > 0.5).float()

            # Process cond frames
            for i in index:
                cond_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret_cond, frame_cond = cond_cap.read()
                if ret_cond:
                    frame_cond = cv2.resize(frame_cond, (self.width, self.height))
                    frame_cond = cv2.cvtColor(frame_cond, cv2.COLOR_BGR2RGB)
                    cond_frames.append(frame_cond)
            cond_frames = np.array(cond_frames)
            cond = torch.from_numpy(cond_frames).permute(3, 0, 1, 2)
            cond_cap.release()
            cond = cond.float() / 255

            video =  video * 2 - 1
            cond =  cond * mask * 2 - 1
            mask =  mask * 2 - 1

            assert video.shape[1] == self.num_frames, f"Video shape mismatch: {video.shape} != {self.num_frames}"
            assert mask.shape[1] == self.num_frames, f"Mask shape mismatch: {mask.shape} != {self.num_frames}"
            assert cond.shape[1] == self.num_frames, f"Cond shape mismatch: {cond.shape} != {self.num_frames}"

            prompt = ""
            return {
                "prompt":  prompt,
                "video": video,
                "mask": mask,
                "cond": cond,
                "input_height": input_height,
                "input_width": input_width,
                "render_path": self.cond[idx],
            }