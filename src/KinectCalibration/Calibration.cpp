#include "Calibration.h"

Calibration::Calibration(string pathToH)
{
	KinectToCV kinect;
	rgb = kinect.getIplImageRGB();
	depth = kinect.getIplImageDepthNormalized();
	H = NULL;
	H = (CvMat*)cvLoad(pathToH.c_str());
	
	if(!H)
	{
		cvSaveImage("imageRgb.png",rgb);
		cvSaveImage("imageDepth.png",depth);

		CvPoint2D32f* rgbPts = getPoints("RGB",rgb);
		CvPoint2D32f* depthPts = getPoints("Depth",depth);
		computeH(rgbPts,depthPts);
		delete rgbPts;
		delete depthPts;

		//cvSave(pathToH.c_str(), H);
		cv::FileStorage fs(pathToH, cv::FileStorage::WRITE);
		fs << "calib" << cv::Mat(H);
	}

	IplImage* warpDepth = changePerspective(depth);

	for(bool sw=0; cvWaitKey(500) < 0; sw=sw?0:1)
	{
		if (sw)	cvShowImage("Switcher", rgb);
		else	cvShowImage("Switcher", warpDepth);
	}

	cvReleaseImage(&warpDepth);
}

Calibration::~Calibration()
{
	cvReleaseMat(&H);
}

void Calibration::computeH(CvPoint2D32f* rgbPts, CvPoint2D32f* depthPts)
{
	this->H = cvCreateMat( 3, 3, CV_32F);
	cvGetPerspectiveTransform( depthPts, rgbPts, H);
}

CvPoint2D32f* Calibration::getPoints(const char* windowName, const IplImage* image)
{
	CVPoints pts;

	IplImage *temp = cvCloneImage(image);
	pts.getPointsFromMouse(windowName,temp);
	cvDestroyWindow(windowName);
	cvReleaseImage(&temp);

	return pts.pts;
}

IplImage* Calibration::changePerspective(const IplImage *image) {
	IplImage *tmp = cvCreateImage(
			cvGetSize(image),
			image->depth,
			image->nChannels
			);
	cvWarpPerspective(
		image, 
		tmp, 
		H, 
		CV_INTER_LINEAR 
			//| CV_WARP_INVERSE_MAP 
			| CV_WARP_FILL_OUTLIERS
		);
	return tmp;
}