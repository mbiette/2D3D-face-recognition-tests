#ifndef FACEDETECTION_VJ_H
#define FACEDETECTION_VJ_H

#include <iostream>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

// Create a string that contains the exact cascade name

class FaceDetectionVJ {
private :
	CvHaarClassifierCascade* cascade;
	CvSeq* faces;
	CvMemStorage* storage;

public :
	FaceDetectionVJ( const char * cascadePath = NULL );
	~FaceDetectionVJ();
	
	void loadCascade( const char * cascadePath );
	void detectFromImage(const IplImage* image);
	void drawRectOnImage(IplImage* image);
	IplImage* drawRectOnNewImage(const IplImage* image);
	std::vector<cv::Mat> cropAndScaleImageIntoMat(cv::Mat image, int width, int height, const int _CvMatType);
};
#endif