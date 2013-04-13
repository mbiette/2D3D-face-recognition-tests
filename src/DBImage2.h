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
#include "Enums.h"
#include "Person.h"


class DBImage
{

public:
	//DBImage();
	DBImage(QString database, typeLoading load, Mixer mix);
	~DBImage();

	//void loadDatabase(); // Verification method checking if the dir exists & co.

	//std::vector<Face> getFaces();
	//std::vector<Face>& getFacesRef();
	int getNbrImgPerFace();
	int getNbrFaces();
	QString getLoadPath();
	//bool getState();

	//void fusionMat();
	//cv::Mat preProcessing(cv::Mat A);
	Person& operator [] (int id);
	int size();
	int lastLoaded();
	int cols();
	int rows();
	/*void setValidationOnly();
	void unsetValidationOnly();*/
	std::string getName(int id);
	int getIDFromName(std::string name);

	void setCibleRGB(){cible=0;}
	void setCibleDepth(){cible=1;}
	void setCibleMix(){cible=2;}

	void clearPerson();

private:
	//Config info
	/*typeDB db;*/
	typeLoading load;
	Mixer mix;
	QString loadPath;
	QString dbName;
	int cible;

	//Loading path
	std::vector<string> names;
	std::vector<std::vector<std::string>> pathRgb;
	std::vector<std::vector<std::string>> pathDepth;
	int nbrImgPerFace;
	int nbrFaces;

	//Temporary data for access
	Person *personTmp;
	int idLoaded;
	
	void loadDatabase(); //The actual doer that load the DB using loadPath.
	bool checkDatabase(); //Just do the verification if the directory exist
};


#endif