![](https://github.com/hualin95/YOLO_v1/blob/master/docs/logo.png)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/hualin95/Everyone_Is_Van_Gogh/blob/master/LICENSE) 
![Build Status](https://img.shields.io/appveyor/ci/gruntjs/grunt/master.svg)
# YOLO_v1
YOLO_v1 is a simple implementation of YOLO v1 by Keras with Tensorflow backend which described in the next paper:
* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

But actually this is not a good implementation and I have not achieved the same performance as the original one.
Further more,[YOLO v2](https://pjreddie.com/darknet/yolo/) has come out,so you can turn to [experiencor/basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras)
which I think is a good implementation of YOLO v2 with Keras if you want to train faster and better.

To use this implementation, you have to download the weights files from [here](https://pan.baidu.com/s/1YsJAVf0CnlG87-MlKd1X2w).

# Environment
* 1.tensorflow v1.4
* 2.keras v2.1.2
* 3.Cudnn v6.0
* 4.Python v3.6 
* 5.Python packages:numpy,PIL,scipy
* 6.Bash(git、ssh)

# Sample
## detection of an image
![]()

[The Starry Night](https://www.moma.org/learn/moma_learning/vincent-van-gogh-the-starry-night-1889) is an oil on canvas by the Dutch post-impressionist painter Vincent van Gogh in June 1889.
## detection of a video
![]() 
