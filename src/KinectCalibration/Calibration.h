#ifndef CALIBRATION_INC
#define CALIBRATION_INC

#include "CVPoints.h"
#include "../KinectToCV.h"


class Calibration 
{
private:
	const IplImage* rgb;
	const IplImage* depth;
	CvMat *H;
	
	void computeH(CvPoint2D32f *rgbPts, CvPoint2D32f *depthPts); // Generate the matrix H .
	CvPoint2D32f* getPoints(const char *windowName, const IplImage *image);
	IplImage* changePerspective(const IplImage *image); // Edit the perspective of the actualFrame 

public:
	Calibration(string pathToH);
	~Calibration();
};

#endif