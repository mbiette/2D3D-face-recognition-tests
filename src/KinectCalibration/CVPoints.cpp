#include "CVPoints.h"

void callback(int event, int x, int y, int flags, void* param) {
	if(event == CV_EVENT_LBUTTONDOWN)
		((CVPoints*)param)->setPoint(x,y);
}

CVPoints::CVPoints()
{
	this->n = 0;
	pts = new CvPoint2D32f[4];
}

CVPoints::~CVPoints()
{
	//delete pts;
}

void CVPoints::getPointsFromMouse(const char *windowName, IplImage *image)
{
	this->windowName = windowName;
	this->image = image;
	cvShowImage( windowName , image );
	cvSetMouseCallback(windowName,callback,this);
	while(n<4) cvWaitKey(1000);
	cvSetMouseCallback(windowName,NULL,NULL);
}
void CVPoints::setPoint(int x, int y)
{
	if(n<4){
		pts[n].x = (double)x;
		pts[n].y = (double)y;
		std::cout << "Coord x : " << pts[n].x << ", Coord y : " << pts[n].y << "\n";
		switch(n)
		{
			case 0: cvCircle( image, cvPointFrom32f(pts[n]), 5, CV_RGB(0,0,255), 2); break;
			case 1: cvCircle( image, cvPointFrom32f(pts[n]), 5, CV_RGB(0,255,0), 2); break;
			case 2: cvCircle( image, cvPointFrom32f(pts[n]), 5, CV_RGB(255,0,255), 2); break;
			case 3:	cvCircle( image, cvPointFrom32f(pts[n]), 5, CV_RGB(255,255,0), 2); break;
		}
		cvShowImage( windowName, image );
		n++;
	}
}