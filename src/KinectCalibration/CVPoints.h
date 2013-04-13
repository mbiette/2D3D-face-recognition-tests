#ifndef CVPOINTS_INC
#define CVPOINTS_INC

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;

void callback(int, int, int, int, void*); //Callback for the cvSetMouseCallback function.

class CVPoints {
private:
	int n;
	const char* windowName;
	IplImage* image;

public:
	CvPoint2D32f* pts;

	CVPoints();
	~CVPoints();
	void setPoint(int x,int y);
	void getPointsFromMouse(const char *windowName,IplImage *image);
};

#endif