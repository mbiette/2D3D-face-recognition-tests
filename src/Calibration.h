#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv/cv.h>

using namespace cv;

class Calibration
{
private:
	Mat H;
public:
	Calibration(string pathToH);
	//~Calibration();

	Mat warp(const Mat &image);
};

#endif