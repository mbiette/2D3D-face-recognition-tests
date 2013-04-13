#include "Calibration.h"

using namespace cv;

Calibration::Calibration(string pathToH)
{
	FileStorage fs(pathToH, FileStorage::READ);
	fs["calib"] >> H;
}

Mat Calibration::warp(const Mat &image)
{
	Mat warpedMat;
	if(H.empty()) return image;
	cv::warpPerspective(image,warpedMat,H,image.size());
	return warpedMat;
}