import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

from flask import Flask,request,jsonify,render_template
from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop

app = Flask(__name__)

@app.route('/')
def hello_world():
    
    return render_template('index.html')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response
class Model_Server():
    print("Ready to handel object detection requests >>>>")
    def __init__(self):
        self.cfg='cfg/Y3_OID_500.cfg'
        self.names='data/Y3_OID_500.names'
        self.weights='weights/Y3_OID_500.weights'
        self.source='data/samples'
        self.output='static/output'
        self.img_size=960
        self.conf_thres=0.2
        self.iou_thres=0.5
        self.device='cpu'
        self.model = Darknet(self.cfg, self.img_size)
        load_darknet_weights(self.model, self.weights)
    def detect(self, save_img=False):
        img_size = (320, 192) if ONNX_EXPORT else self.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source, weights = self.output, self.source, self.weights
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.device)
        if not os.path.exists(out):
            os.makedirs(out)  # make new output folder

        # Eval mode
        self.model.to(device).eval()

        save_img = True
        dataset = LoadImages(source, img_size=img_size)

        # Get names and colors
        names = load_classes(self.names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = self.model(img)[0]
            t2 = torch_utils.time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        if n>1:
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        else:
                            s += '%g %s, ' % (n, names[int(c)])  # add to string
                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
        print('Done. (%.3fs)' % (time.time() - t0))
        
        return s

    

MS = Model_Server()
@app.route('/predict',methods=['GET','POST'])
def Model_Response():
    # image = request.files['file'].read()
    for uploaded in request.files.getlist("file"):
        type = uploaded.filename.split('.')[1]#request.files['file'].filename.split('.')[1]
        imPath = '/home/dell/Documents/Rajashekar/yolov3/input/{}.{}'.format(time.strftime("%Y%m%d-%H%M%S%s"),type)
        print(uploaded,imPath)
        uploaded.save(imPath)
    
        fileName = 'output'+ os.sep + os.path.basename(imPath)
    MS.source = imPath
    with torch.no_grad():
        result = MS.detect(save_img=True)
    return render_template("result.html",image_name=fileName,ress = result)

if __name__ == '__main__':   
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5775,address='0.0.0.0')
    IOLoop.instance().start()