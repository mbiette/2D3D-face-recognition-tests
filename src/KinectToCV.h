#ifndef KINECTTOCV_H
#define KINECTTOCV_H

#include <libfreenect_sync.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

#define KINECT_RGB_W 640
#define KINECT_RGB_H 480
#define KINECT_RGB_DEPTH 8
#define KINECT_RGB_CH 3
#define KINECT_DEPTH_W 640
#define KINECT_DEPTH_H 480
#define KINECT_DEPTH_DEPTH 16
#define KINECT_DEPTH_CH 1

class KinectToCV 
{
private:
	unsigned char index;
	IplImage *srcBGR, *RGB;
	IplImage* srcDepth, *normDepth;

public:
	KinectToCV(int index = 0);
	const IplImage* getIplImageDepth();
	const IplImage* getIplImageDepthNormalized();
	const IplImage* getIplImageBGR();
	const IplImage* getIplImageRGB();
	/*Mat getMatDepth();
	Mat getMatDepthNormalized();
	Mat getMatBGR();
	Mat getMatRGB();*/
};

#endif