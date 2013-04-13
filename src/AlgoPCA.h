#ifndef ALGOPCA_H
#define ALGOPCA_H

#include "DBImage2.h"
#include "KNearest.h"
#include "Enums.h"
#include <iostream>
#include <limits>
#include <QDir>
#include <QString>


class AlgoPCA {
private : 
	//The picture DB
	DBImage &dbSource; //The source of the images
	int nbFaces;
	int nbImgPerFace;
	std::vector<cv::Mat> cov, mean, covRGB, covDepth, meanRGB, meanDepth;
	typeLoading imgToLoad;

	//int cptFace,cptImgFace; //??
	double minNorm;
	double maxNorm;
	inline void addNorm(double norm)
	{
		if(norm<minNorm) minNorm=norm;
		if(norm>maxNorm) maxNorm=norm;
		//std::cout << norm << endl;
	}

	//Features from the algo
		//Just the data keeped
	int nbEigenVectors; //The number of eigen vectors we are keeping
	double pourcentageGoal; //The minimal pourcentage wanted
	double pourcentage; //The actual pourcentage
	cv::Mat EigenVectorsReduced; //The reduced Matrix of eigenvectors of nbEigenVectors
		//The projection of the images.
	std::vector<std::vector<cv::Mat>> Y;
	std::vector<std::vector<cv::Mat>> Yp;
	
	//Apply the feature reduction
	void execute();
	inline cv::Mat featureCalc(cv::Mat src);
		//Intern methods
	double computePourcentage(cv::Mat& EigenValues);
	void pcaEigen(cv::Mat& EigenValues, cv::Mat& EigenVectors);
			//SubInternal Methods
	void pcaMean(cv::Mat& mean);
	void pcaCovariance(cv::Mat& G, const cv::Mat& mean);
	
	////Classification
	//	//One image algo
	//template<typename _Tp> int discriminantFunction(cv::Mat& featureVector, double threshold);
	//template<typename _Tp> int discriminantFunctionKNN(cv::Mat& featureVector, int k);
	//int classifier2DPCA(cv::Mat& featureVector, double threshold);
	//int classifier2DPCASum(cv::Mat& featureVector, double threshold);
	//int classifier2DPCAKNN(cv::Mat& featureVector, int k);
	//	//Mix algo
	//template<typename _Tp> int discriminantFunction(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, double threshold);
	//template<typename _Tp> int discriminantFunctionKNN(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, int k);
	//int classifier2DPCA(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, double threshold);
	//int classifier2DPCASum(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, double threshold);
	//int classifier2DPCAKNN(cv::Mat& featureVectorRGB, cv::Mat& featureVectorDepth, int k);
	//
	////Distance
	//double mahalanobisDistance(cv::Mat covMatrix,cv::Mat meanMatrix, cv::Mat featureMatrix);
	

public : 
	
	//Constructor
	AlgoPCA(DBImage &db, double setPct);

	//??
	//void upCptFace(bool up);
	//void upCptImgFace();
	//void setCptFace(int a);
	//int getCptFace();
	//int getCptImgFace();
	//std::vector<std::vector<cv::Mat>>& getFeaturesVectorY();
	//cv::Mat getEigVector1();

	//Apply the feature reduction
	void launch(); //Make a call to the template execute.
	vector<vector<cv::Mat>> featureVectOUT(DBImage& databaseToFeatVect);
	//Infos
	double getPourcentage();

	//Classification
	int classification(typeDistance dist, typeDecision classif, double param, cv::Mat& image);
	int classification(typeDistance dist, typeDecision classif, double param, cv::Mat& rgb, cv::Mat& depth);
	inline double getMinNorm(){ return minNorm; }
	inline double getMaxNorm(){ return maxNorm; }
	/*inline void printFV( string pathprefix )
	{
		for(int i=0;i<
			ofstream file(filename);
			for(int i=0;i<A.rows;i++){
				for(int j=0;j<A.cols;j++)
					file << double(A.at<_Tp>(i,j)) << "\t";
				file << endl;
			}
			file << endl;
			file.close();
	}*/
	//int executeDF(cv::Mat& image);
	//int executeDF(cv::Mat& rgb, cv::Mat& depth);
	//int execute2DPCA(cv::Mat& image);
	//int execute2DPCA(cv::Mat& rgb, cv::Mat& depth);
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
};

#endif
