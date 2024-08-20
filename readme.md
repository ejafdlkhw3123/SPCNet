# SPCNet: MR Sequence Parameter Controllable Network for Multi-Contrast Magnetic Resonance Image Synthesis via Style Alignment and Parameter Inference

In this study, we propose the Sequence Parameter Controllable Network (SPCNet) that synthesizes multi-contrast images from a single acquired image. SPCNet decouples content and style within an MR image and adjusts the style based on the target MR sequence parameters. In conclusion, SPCNet enables the control of
contrast in MR images using sequence parameters.

## Results 
<p align="center">
<img src="./figure/result.png" width="90%" height="90%">
</p>
Results of controlling MR image contrast using
MR sequence parameters. SPCNet receives T1-w and T2-w images,
along with six sequence parameters. When the
own sequence parameters are specified, the output images
remain unchanged. In contrast, when different sequence parameters
are specified, the output images are converted to the
contrast of the specified sequence. <br>


## Framework
<p align="center">
<img src="./figure/framework.png" width="90%" height="90%">
</p> 
Overview of the SPCNet architecture, which controls the contrast of MR images using MR sequence parameters. The
proposed network consists of four main parts: 1) an encoder that decouples MR image into content and style components, 2) a
parameter-guided style conversion module (PSCM) that adjusts the style based on the MR sequence parameters, 3) a generator
that synthesizes the decoupled content and modified style, and 4) a parameter extractor and dual discriminators that calculate
the loss function to update the network.

## Requirements
* python 3.11
* pytorch 2.2.2
* numpy, scikit-image, yaml, argparse, cv2, h5py
<br> 

## Datasets
The dataset is constructed as a dictionary in HDF5 format.
Example dataset
```
import h5py, json

data = h5py.File(path)
img = data['array']
info = json.loads(data.attrs['header'])
```

## Training
```
python main.py --dataset_path ./dataset --save_path test
```

## Testing
```
python main_test.py  --dataset_path ./dataset --save_path test
```
