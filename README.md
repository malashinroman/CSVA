# Core structural verification algorithm of keypoint matches
C++ implementation of the robust keypoint elimination algorithm described in the paper

```
Malashin R.O. Core algorithm for structural verification of keypoint matches. Intelligent Systems Reference Library. Computer Vision in Control Systems-3. 2018. P. 251-286
```

## Update 2023

I slightly refactored code that is now compatible with opencv4.x and should be easier to compile.

### Ubuntu 22.04 install

1. Install Boost

```
apt-get install libboost-all-dev
```
2. Download, compile and install opencv with opencv_contrib. Follow opencv [documentation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html "documenation"). Make sure to enable DOPENCV_ENABLE_NONFREE flag when configure.

I used the following command:
```
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DOPENCV_ENABLE_NONFREE:BOOL=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
```

followed by 'make -j8' and 'make install'

4. clone CSVA project from github, then in command line:
```
cd CSVA
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j8
```
5. You should get following files in CSVA/bin directory libcommon_lib.a, libcsva_lib.so, match3D,  match_aero.
If you see them, then everything is OK, and you can proceed to tests.  

## Description

CSVA (“core” structural verification algorithm) is a robust and fast algorithm for outlier elimination of keypoint matches (e.g. SURF, SIFT or BRISK features) that heavily exploits gemetrical constraints. It can be used by developers and researches in different practical tasks with minor adaptation. CSVA an improved version of SIFT clustering proposed by David Lowe in SIFT.

## Demo video

[video1](https://www.youtube.com/watch?v=ik6-zfD-ozk "Demo video") and [video2](https://www.youtube.com/watch?v=9miJET-FDbo)

for more demonstration results visit [project page](https://malashinroman.github.io/CSVA/)

## Prerequisites

```
opencv
```
CSVA uses openCV structures like cv::DMatch and cv::KeyPoint internally, opencv3.1 and opencv4.0 were tested

```


opencv-contrib
```
You will need xfeatures2d from opencv-contrib to compile and test with SURF and SIFT


```
Boost 
```
CSVA uses boost library in it's core


### Operating system
Windows 7 and Ubuntu 14.04 were tested

## Compile

Cmake is a recommended tool for creating CSVA project


## Running test examples

CSVA has two regimes for 3D scene matching and aerospace matching.

### 3D scene matching

After compilation you can run
```
match3D image1.jpg image2.jpg
```
The result showing keypoint matches selected by CSVA  will be drawn in result.jpg

### aerospace matching

After compilation you can run
```
match_aero im1.jpg im2.jpg 0.5 0.5 1
```
where im1.jpg is first aero or space image, im2.jpg is second aero or space image, 0.5 is a resize factor, 1 - is fast mode (use 2 to get robust results).

Folder <im1.jpg> will be created by the program, where images demonstrating the matching result (keypoint mathces, color alighnment, etc) will be saved.


## Use csva_lib in your project
To use CSVA in your C++ project 
1. compile csva_lib
2. link csva_lib to the project
3. Include header in the file you need
```
#include "csva_lib/csva.h"
```
4. CSVA is opencv compatible. You can verify matches generated by opencv and stored in vector<cv::DMatch> by the following command:
```
csva::filter_matches(points1, points2, matches, image1, image2, csva::geometry_mode::THREEDIM_SCENE, LocalMatchtype, goodmatches, confidence, 0.001);
```
where points1 and points2 are vector<cv::KeyPoint>, image1 and image2 are two images in cv::Mat, LocalMatchtype is a flag specifying which feature detector, feature extractor and matcher was used to genereate matches (for example, 352 is HammingMatcher + BRISK descriptor + SURF detector), goodmatches is output vector<cv::DMatch> for inliers, confidence is output double[6] confidence with first element specifying the confidence of the solution taking into accout goodness of the selected matches, 0.001 is confidence of random cluster forming (you can use another value here, it affects computation of output confidence)

Current repository suggests using OpenCVfeatures wrapper to compute features defined in features2d and xfeatures2d modules:
std::vector<DMatch> matches = feat.getLocalPatchMatches2(image1, image2, points1, points2, LocalMatchtype, 0);


## Authors

* **Roman Malashin** - [malashinroman](https://github.com/malashinroman)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citing
```
1. Malashin R.O. Core algorithm for structural verification of keypoint matches. Intelligent Systems Reference Library. Computer Vision in Control Systems-3. 2018. P. 251-286
2. Malashin R. Matching of Aerospace Photographs with the use of Local Features // Journal of Physics: Conference Series - 2014, Vol. 536, No. 1, pp. 012018.
```

