#include "KinectToCV.h"

KinectToCV::KinectToCV(int index)
{
	this->index = index;
	this->srcDepth = NULL;
	this->srcBGR = NULL;
	this->normDepth = NULL;
	this->RGB = NULL;
}

const IplImage* KinectToCV::getIplImageDepth()
{
	static int *data = 0;
	if (!srcDepth) srcDepth = cvCreateImageHeader(cvSize(KINECT_DEPTH_W,KINECT_DEPTH_H), KINECT_DEPTH_DEPTH, KINECT_DEPTH_CH);
	unsigned int timestamp;
	//std::cout << "New depth...";
	if (freenect_sync_get_depth((void**)&data, &timestamp, index, FREENECT_DEPTH_11BIT)) return NULL;
	//std::cout << " DONE"<<endl;
	cvSetData(srcDepth, data, KINECT_DEPTH_W*KINECT_DEPTH_CH*(KINECT_DEPTH_DEPTH/8));
	return srcDepth;
}

const IplImage* KinectToCV::getIplImageDepthNormalized()
{
	if(this->getIplImageDepth() == NULL) return NULL;
	if (!normDepth) normDepth = cvCreateImage(cvSize(KINECT_DEPTH_W,KINECT_DEPTH_H),8,1);
	cvNormalize(srcDepth,normDepth,0,255,CV_MINMAX);
	return normDepth;
}

const IplImage* KinectToCV::getIplImageBGR()
{
	static int *data = 0;
	if (!srcBGR) srcBGR = cvCreateImageHeader(cvSize(KINECT_RGB_W,KINECT_RGB_H), KINECT_RGB_DEPTH, KINECT_RGB_CH);
	unsigned int timestamp;
	//std::cout << "New image...";
	if (freenect_sync_get_video((void**)&data, &timestamp, index, FREENECT_VIDEO_RGB)) return NULL;
	//std::cout << " DONE"<<endl;
	cvSetData(srcBGR, data, KINECT_RGB_W*KINECT_RGB_CH*(KINECT_RGB_DEPTH/8));
	return srcBGR;
}

const IplImage* KinectToCV::getIplImageRGB()
{
	if(this->getIplImageBGR() == NULL) return NULL;
	if (!RGB) RGB = cvCreateImage(cvSize(KINECT_RGB_W,KINECT_RGB_H), KINECT_RGB_DEPTH, KINECT_RGB_CH);
	cvCvtColor(srcBGR, RGB, CV_BGR2RGB);
	return RGB;
}