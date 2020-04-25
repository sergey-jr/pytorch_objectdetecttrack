import cv2
from torchvision import transforms

from models import *
from sort import *
from utils import utils


def detect_image(img, model, img_size=416, conf_thres=0.75, nms_thres=0.4):
    Tensor = torch.cuda.FloatTensor
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0)),
                                                        (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--video_path",
                    help="path to video file including it's name and extension")
args = parser.parse_args()
# load weights and set defaults
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'
img_size = 416

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
model.eval()

classes = ['person']

videopath = args.video_path

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128),
          (128, 128, 0), (0, 128, 128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort()

cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800, 600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = vid.read()
fps = vid.get(cv2.CAP_PROP_FPS)
orig_res = frame.shape[:-1]
hd = (1280, 720)
full_hd = (1920, 1080)

outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det" + videopath[-4:]), fourcc, fps, orig_res)

frames = 0
starttime = time.time()
while True:
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg, model)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            if int(cls_pred) in range(len(classes)):
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 3)

    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time() - starttime
print(frames, "frames", totaltime / frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()
