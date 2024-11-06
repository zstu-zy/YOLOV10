1.  构建你的datasets数据集

    为datasets文件夹包含train文件夹、test文件夹、valid文件夹，每个文件夹中包含images文件夹以及labels文件夹。

2.  编写data.yaml文件

    names:

    \- 0:\'Title\'

    \- 1:\'Header\'

    \- 2:\'Text\'

    \- 3:\'Figure\'

    \- 4:\'Foot\'

    nc: 5

    path: /root/autodl-tmp/yolov10-main/data

    train: train/images

    val: valid/images

    test: test/images

    其中names为类别，nc为类别数量，path为指向你的数据集的的路径，下面train等均为相对你的数据集文件夹的相对路径。

3.  配置相应的环境

    首先租一个服务器，将数据集在本地划分完成后将整个yolov10-main的文件夹上传至服务器，服务器参考如下：

    Pytorch2.1.2 python 3.10（ubuntu22.04） cuda 11.8

4.  配置相应的环境

    使用vscode连接服务器后，在终端切换至yolov10-main的文件夹输入以下代码：\
    pip install requirements.txt -i
    http://mirrors.aliyun.com/pypi/simple \--trusted-host
    mirrors.aliyun.com

    pip install -e.

5.  进行训练

    通过以下代码进行训练，其中参数可以在zy-yolov10-main/ultralytics/cfg/default.yaml中查看并进行更改

    yolo detect train
    data=/root/autodl-tmp/yolov10-main/datasets/my_data.yaml
    model=yolov10n.pt epochs=200 batch=16 imgsz=640 device=0

6.  模型的量化

    通过以下change.py代码进行模型的量化，将你的.pt模型转化为.xml以及.bin。

7.  模型的部署

    通过load.py进行模型的部署，其中读取test文件夹中的所有数据，通过你的.xml以及.bin对所读取的数据进行预测，输出.xml结果在output文件夹中。
