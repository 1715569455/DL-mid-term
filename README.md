# DL-mid-term
Codes for mid-term work of Deep Learning Course
## Building the environment
### 1. Create a new conda environment and install pytorch
### 2. Install the necessary dependence of mmdetection
```angular2html
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
### 3. Clone the mmdetection repository, change into the master directory of mmdetection and build the environment by 
```
pip install -v -e .
```
## Usage
### For training, we can use the following command
```angular2html
python .\train.py [config_path]
```
Then a pretrained CNN backbone will be downloaded automatically, and the results will be saved at ```.\work_dirs```. The config files are located in ```.\config\faster_rcnn``` and ```.\config\yolo``` respectively.
### For testing, we can use the following command
```angular2html
python .\test.py [config_path] [checkpoint_path]
```
The model will we evaluated on the test set, and the result will also be saved at ```.\work_dirs```.
### For proposal generation, we can use the following command
```angular2html
python .\show_proposals.py [config_path] [checkpoint_path] --image-file [tested_image_path] --out-dir [output_path]
```
Then the region proposals of ```tested_image_path``` will be saved at ```output_path```.
### For single-image prediction, we can use the following command
```angular2html
python .\image_demo.py [tested_iamge_path] [config_path] --weights [checkpoint_path]
```
The predicted bounding box, its related class and scores will be saved at ```.\outputs```.

To run the model, just download the original PASCAL VOC 2007 dataset and place them in the ```.\data``` directory. The saved checkpoint can be downloaded at https://drive.google.com/drive/folders/1OSrXVrDq7uFKt15HCsPNvorux7k8isHD?usp=sharing
