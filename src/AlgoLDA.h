#ifndef LDA_H
#define LDA_H

#include "DBImage2.h"
#include "KNearest.h"
#include "Enums.h"
#include <iostream>
#include <limits>
#include <QDir>
#include <QString>
#include <sstream>

class AlgoLDA {
private : 
	//The picture DB
	DBImage &dbSource;
	DBImage &faces;
	int nbrimageperclass,nbrclass,numFisher;
	std::vector<cv::Mat> cov, mean, covRGB, covDepth, meanRGB, meanDepth;
	typeLoading imgToLoad;

	//Features from the algo
	std::vector<std::vector<cv::Mat>> Y;
	std::vector<std::vector<cv::Mat>> Yp;
	cv::Mat EigV;

	/*int cptFace,cptImgFace;*/
	double minNorm;
	double maxNorm;
	inline void addNorm(double norm)
	{
		if(norm<minNorm) minNorm=norm;
		if(norm>maxNorm) maxNorm=norm;
	}

	//Apply the lda
	void execute();
	inline cv::Mat featureCalc(cv::Mat src);
	//	//Intern methods
	void ldaEigen(cv::Mat& EigenValues, cv::Mat& EigenVectors);
	//		//SubInternal Methods
	void ldaMeanSubMean(cv::Mat& m,std::vector<cv::Mat>& mi);
	void ldaScatterBetweenMat(const std::vector<cv::Mat>& mi, cv::Mat& m,cv::Mat& Sb);
	void ldaScatterWithinMat(const std::vector<cv::Mat>& mi,cv::Mat& Sw);

	//Classification
		//One image algo
	//template<typename _Tp> int discriminantFunction(cv::Mat& featureVector, double threshold);
	//template<typename _Tp> int discriminantFunctionKNN(cv::Mat& featureVector, int k);
	//int classifier2DLDA(cv::Mat& featureVector, double threshold);
	//int classifier2DLDASum(cv::Mat& featureVector, double threshold);
	//int classifier2DLDAKNN(cv::Mat& featureVector, int k);
	//	//Mix algo
	//template<typename _Tp> int discriminantFunction(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, double threshold);
	//template<typename _Tp> int discriminantFunctionKNN(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, int k);
	//int classifier2DLDA(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, double threshold);
	//int classifier2DLDASum(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, double threshold);
	//int classifier2DLDAKNN(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, int k);
	
	//Distance
	/*double mahalanobisDistance(cv::Mat covMatrix,cv::Mat meanMatrix, cv::Mat featureMatrix);*/
public : 
	AlgoLDA(DBImage &db/*, typeLoading toLoad*/);
	void launch();

	//Classification
	/*int classification(typeDistance dist, typeDecision classif, double param, cv::Mat& image);
	int classification(typeDistance dist, typeDecision classif, double param, cv::Mat& rgb, cv::Mat& depth);*/
	/*inline double getMinNorm(){ return minNorm; }
	inline double getMaxNorm(){ return maxNorm; }*/
	inline string createFVfilename(string name, int i, int j)
	{
		std::ostringstream oss;
		oss << "feature_vector-" << name <<"-["<< i<<"]["<<j<<"].txt";
		return oss.str();
	}
	inline string createFVfilename(string name)
	{
		std::ostringstream oss;
		oss << "feature_vector-" << name <<".txt";
		return oss.str();
	}

	vector<vector<cv::Mat>> featureVectOUT(DBImage& databaseToFeatVect);
};

#endif
