from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import cv2
print("device_count:", torch.cuda.device_count())

#config_file = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
#checkpoint_file = 'checkpoints/resnet50-19c8e357.pth'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# test a single image and show the results
img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
# inference the demo image
result = inference_detector(model, img)
#print(result)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video_reader = mmcv.VideoReader('demo/demo.mp4')
#for frame in video_reader:
#    result = inference_detector(model, frame)
#    model.show_result(frame, result, wait_time=1)

video_writer = None

# for frame in mmcv.track_iter_progress(video_reader):
#         result = inference_detector(model, frame)
#         frame = model.show_result(frame, result, score_thr=0.3)
#         if args.show:
#             cv2.namedWindow('video', 0)
#             mmcv.imshow(frame, 'video', args.wait_time)
#         if args.out:
#             video_writer.write(frame)

#     if video_writer:
#         video_writer.release()
#     cv2.destroyAllWindows()