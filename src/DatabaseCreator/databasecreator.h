#ifndef DATABASECREATOR_H
#define DATABASECREATOR_H

#include <QtGui/QMainWindow>
#include "ui_databasecreator.h"
#include <conio.h>
#include <iostream>
#include <string>
#include "../KinectToCV.h"
#include <qdir.h>
#include <qstring.h>
#include "../FaceDetectionVJ.h"
#include "../Calibration.h"
#include "holefilling.h"
#include <limits>

typedef struct pix{
	int nb;
	float prob;
}t_pix;

class DatabaseCreator
{
public:
	DatabaseCreator();
	~DatabaseCreator();
	void saveName();
	void core();
	void saveDepth(int a,std::vector<cv::Mat> &A);
	void saveRGB(int a,std::vector<cv::Mat> &B);

private:
	bool session,stop;
	short int nbPers;
	QString name;
	KinectToCV kin;
	QString path,path2;
	const IplImage* tempBGR,*tempDepth;
	cv::Mat Depth,BThres;
	QDir dir1,dir2;
};

#endif // DATABASECREATOR_H
