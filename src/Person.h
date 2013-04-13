#ifndef PERSON_H
#define PERSON_H

#include <time.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>
#include "Enums.h"

class Person
{
private:
	int sizeVal;
	
	void loader(std::vector<cv::Mat>& dest, std::vector<std::string>& path,bool color, bool preprocessing);
	Mixer typeMix;
	void doMix();
	cv::Mat toDouble(cv::Mat& in);
	cv::Mat preProcessing(cv::Mat& A);
	cv::Mat cropper(cv::Mat & in);
	std::vector<cv::Mat>* cible;

public:
	/** Methods **/
	Person(std::string name, std::vector<std::string>& pathRgb, std::vector<std::string>& pathDepth, Mixer _mix);
	int size();
	int rows();
	int cols();

	/** Data **/
	const std::string name;

	std::vector<cv::Mat> rgb;
	//std::vector<cv::Mat> rgbVal; //In case of the M-Fold
	//typename t_rgb;

	std::vector<cv::Mat> depth;
	//std::vector<cv::Mat> depthVal; //In case of the M-Fold
	//typename t_depth;

	std::vector<cv::Mat> mix; //In case of mix
	//typename t_mix;
	void setCibleRGB();
	void setCibleDepth();
	void setCibleMix();

	cv::Mat& operator [] (int id);
};

#endif PERSON_H