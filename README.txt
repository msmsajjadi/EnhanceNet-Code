 ------------
| EnhanceNet |
 ------------

This is a pre-trained reference implementation of ENet-PAT from "EnhanceNet:
Single Image Super-Resolution through Automated Texture Synthesis" for a
magnification ratio of 4.

It comes with a script which takes care of installing all necessary packages
inside a virtual environment, i.e. all installations take place inside the
folder and can be simply uninstalled by deleting the folder. Advanced users
may install the packages directly (see FAQ section).

If you use this code as part of a publication, please cite:

@inproceedings{enhancenet,
  title={{EnhanceNet}:
         Single Image Super-Resolution through Automated Texture Synthesis},
  author={Sajjadi, Mehdi~S.~M. and Sch{\"o}lkopf, Bernhard and Hirsch, Michael},
  booktitle={{ICCV}},
  year={2017},
  url={https://arxiv.org/abs/1612.07919/}}

 -------
| HOWTO |
 -------

Upscale images 4x:
    > Copy all high-resolution images to the input directory.
    > Enter the following into a terminal window:
          ./run.sh
    The images will be first downscaled and then upscaled with EnhanceNet.

Requirements:
    > an internet connection (for the first run only)
    > python (python.org)
    > the following packages, which usually come with Python)
        > pip (https://pip.pypa.io/en/stable/installing/)
        > virtualenv (pip install --user virtualenv)

 -------
| NOTES |
 -------

> All images in the input folder are downscaled 4x and then upscaled via bicubic
  interpolation and EnhanceNet-PAT.

> Existing files in the output directory will *not* be replaced.

> The first run may take a while, since all necessary packages are installed.
  Subsequent runs are much faster and do not necessitate an internet connection.

> For compatibility reasons, the CPU version of TensorFlow is used and the TF
  computation graph is rebuilt for each image, so this reference implementation
  does not reflect the runtime performance of our model and is not suitable for
  runtime benchmarks.

 -----
| FAQ |
 -----

> Upon calling "./run.sh", the message "./run.sh: Permission denied" appears.
  Solution: Enter "chmod 544 run.sh", then retry the run command.

> I already have TensorFlow and/or don't want to run the virtualenv script.
  Solution: Install the necessary packages (see run.sh), then run
                python enhancenet.py

> I have CUDA and want this to be faster for large images.
  Solution: In run.sh, add "-gpu" to the line with tensorflow:
                tensorflow-gpu=0.12.0

> I don't want to downscale the images before upscaling, I want the actual
  upscaled version of 4x the size of the input images.
  Solution: In "enhancenet.py", set the scaling factor to 1:
                imgs = loadimg('input/'+fn)
                imgs = loadimg('input/'+fn, 1)
            Please note that the model was specifically trained on input images
            downsampled with PIL, i.e. it won't perform as well on images
            downscaled with another method. Furthermore, it is not suitable for
            upscaling noisy or compressed images, since artifacts in the input
            will be heavily amplified. For comparisons on such images, the
            model needs to be trained on such a dataset.

> I get the error message "std::bad_alloc".
  Solution: You are likely running out of memory! Please note that even 1000x1000
            input images need a lot of RAM (when the input is not downsampled).

> Something goes wrong and I am left with an error message.
  Solution: Please email us (even if you solved it, so we can add it to the FAQ).

For any questions, comments or help to get it to run, please don't hesitate to
email us: msajjadi@tue.mpg.de

Version 1.01
