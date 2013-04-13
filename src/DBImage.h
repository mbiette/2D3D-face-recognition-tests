#ifndef DBIMAGE_H
#define DBIMAGE_H

#include <time.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>
#include <QDebug>
#include <QDir>
#include <QStringlist>
#include <QString>
#include <QWidget>
#include "FaceDetectionVJ.h"

struct Face {
	QString name;

	// RGB
	std::vector<cv::Mat> images; // un vector d'img CV en théorie
	cv::Mat valImage;

	// Depth
	bool depth;
	std::vector<cv::Mat> imgDepth;
	cv::Mat valImgDepth;

	// Fusion
	std::vector<cv::Mat> fusion;
};

class DBImage
{

public:
	DBImage();
	DBImage(QString database);

	void loadDatabase(QString database); // Verification method checking if the dir exists & co.

	std::vector<Face> getFaces();
	std::vector<Face>& getFacesRef();
	int getNbrImgPerFace();
	int getNbrFaces();
	QString getDbName();
	bool getState();

	void fusionMat();
	cv::Mat preProcessing(cv::Mat A);

private:
	std::vector<Face> faces;
	QString loadPath;
	QString dbName;
	
	void imagesLoader(); // The actual doer that load the DB using loadPath.
	int nbrImgPerFace;
	int nbrFaces;
	bool allocated;
};

#endif // DBIMAGE_H
