#ifndef HOLEFILLING_H
#define HOLEFILLING_H

#include "databasecreator.h"
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

class HoleFilling {
private : 
	int degre;
	int threshold;
	Mat &ImgMat;

public : 
	HoleFilling(cv::Mat &img,int deg,int thresh);
	/*Mat calculCoeff( vector<CvPoint> &Data,int degre);
	bool detectHole(Mat &D,int j,int threshold);
	bool detectHole(Mat &D,int i,int j,int threshold);
	void fillBackground(Mat &data);*/
	void removeHole();
	Mat HoleFilling :: getData();
};

#endif