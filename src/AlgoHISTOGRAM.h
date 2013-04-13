#ifndef ALGOHISTOGRAM_H
#define ALGOHISTOGRAM_H

#include "DBImage2.h"
#include "KNearest.h"
#include "Enums.h"
#include <iostream>
#include <limits>
#include <QDir>
#include <QString>

class AlgoHISTOGRAM {
private : 
	//The picture DB
	//DBImage &dbSource; //The source of the images
	//DBImage &faces;
	/*int nbFaces, nbImgPerFace;*/
	/*std::vector<cv::Mat> cov, mean;*/
	//int cptFace,cptImgFace;
	/*double minNorm;
	double maxNorm;
	inline void addNorm(double norm)
	{
		if(norm<minNorm) minNorm=norm;
		if(norm>maxNorm) maxNorm=norm;
	}*/

	//vector<Face> A;
	/*std::vector<std::vector<cv::Mat>> H;
	typeLoading imgToLoad;*/

	//Apply the algorithm
	cv::Mat HcompCalc(cv::Mat src);



public : 
	AlgoHISTOGRAM(/*DBImage &db*/);
	//void launch();

	vector<vector<cv::Mat>> featureVectOUT(DBImage& databaseToFeatVect);
	//Classification
	//int classification(typeDistance dist, typeDecision classif, double param, cv::Mat& image);
	//inline double getMinNorm(){ return minNorm; }
	//inline double getMaxNorm(){ return maxNorm; }
	//inline string createFVfilename(string name, int i, int j)
	//{
		//std::ostringstream oss;
		//oss << "feature_vector-" << name <<"-["<< i<<"]["<<j<<"].txt";
		//return oss.str();
	//}
	//inline string createFVfilename(string name)
	//{
		//std::ostringstream oss;
		//oss << "feature_vector-" << name <<".txt";
		//return oss.str();
	//}
};

#endif
