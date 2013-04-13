#include "algohistogram.h"
#include <string.h>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////////////////////////////
////  Math Operators  --------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
#include "InlineMatrixOperations.cpp"

////////////////////////////////////////////////////////////////////////////////////
////  Constructors  ----------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////

AlgoHISTOGRAM :: AlgoHISTOGRAM(/*DBImage &db, typeLoading toLoad*/)/*:dbSource(db), faces(db)*/{
	/*nbFaces = 0;
	nbImgPerFace = 0;*/
	/*imgToLoad = toLoad;*/
	/*minNorm = numeric_limits<double>::max();
	maxNorm = numeric_limits<double>::min();*/
}

////////////////////////////////////////////////////////////////////////////////////
////  Launchers  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////

//void AlgoHISTOGRAM::launch(){ // Avec en paramètre la database après...
//	nbFaces = dbSource.getNbrFaces(); // The number of faces we have to load (wrt our database)
//	nbImgPerFace = dbSource.getNbrImgPerFace();
//	/*if(!faces[0].depth)
//	{
//		cout << "PCA: Error no depth in DBImage and MIX selected.";
//		exit(1);
//	}*/
//
//		execute();
//
//}


////////////////////////////////////////////////////////////////////////////////////
////  H Comp  --------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
cv::Mat AlgoHISTOGRAM::HcompCalc(cv::Mat image)
{
	int b=0;
	Mat Hcomp(256, 1, CV_64FC1, Scalar(0));

	
	for (int i=0;i<image.rows;i++)
		for (int j=0;j<image.cols;j++)
		{
			b = image.at<double>(i,j);
			//if(b<0)b=0; if(b>255)b=255;
			Hcomp.at<double>(b,0) += 1;
		}

	return Hcomp;
}

////////////////////////////////////////////////////////////////////////////////////
////  Training  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////

//void AlgoHISTOGRAM::execute(){	
//	int b=0;
//	H.resize(nbFaces);
//	for(int i = 0; i < nbFaces ; i++)
//		H[i].reserve(nbImgPerFace);
//	for(int k = 0 ; k < nbFaces ; k++)
//		for(int l = 0 ; l < nbImgPerFace ; l++)
//		{
//			H[k].push_back(Mat(256, 1, CV_64FC1, Scalar(0)));
//			for (int i=0;i<faces.rows();i++)
//				for (int j=0;j<faces.cols();j++)
//				{
//					b = faces[k][l].at<double>(i,j);
//					H[k][l].at<double>(b,0) += 1;
//				}
//#ifdef PRINT_FEATVECT_TRAIN_HISTO
//			matPrint<double>(H[k][l],createFVfilename("histo-depth",k,l));
//#endif
//		}
//	//END EXECUTE
//}

////////////////////////////////////////////////////////////////////////////////////
////  Classification  --------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////

/*template<typename _Tp> int* AlgoHISTOGRAM::classifierHISTOGRAM(cv::Mat& Hcomp, double param)
{
	
	Mat Diff((256, 1, CV_64FC1, Scalar(0)));
	double min=numeric_limits<double>::max();
	double vectNorm=0;
	double vectNormMin=numeric_limits<double>::max();

	int tab=new int[2];
	tab[0]=-1;
	tab[1]=-1;
	for(int k = 0 ; k < nbrfaces ; k++)
		for(int l = 0 ; l < nbrImgPerFace ; l++)
		{
			vectNorm = 0;
			Diff = H[k][l] - Hcomp;
			for (int i=0;i<256;i++)
					vectNorm += Diff.at<double>(i,0)*Diff.at<double>(i,0);
			vectNorm = sqrt(vectNorm);
			if (vectNorm < vectNormMin)
			{
				vectNormMin = vectNorm;
				tab[0] = k;
				tab[1] = l;
			}
		}
	return tab;
}*/

//int AlgoHISTOGRAM::classifierHISTOGRAM(Mat& Hcomp, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//	double tempVal;
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		for(int j = 0; j < nbImgPerFace ; j++)
//		{
//			tempVal=norm(H[i][j],Hcomp);
//			
//			if(tempVal<min)
//			{
//				min=tempVal;
//				tab = i; // minFaceIndex
//			}
//		}
//	}
//	addNorm(min);
//	if(min<threshold) return tab;
//	else return -1;
//}
//
//int AlgoHISTOGRAM::classifierHISTOGRAMSum(Mat& Hcomp, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		double sum=0;
//		for(int j = 0; j < nbImgPerFace ; j++)
//		{
//			sum += norm(H[i][j],Hcomp);
//		}
//		sum /= nbImgPerFace;
//		
//		if(sum<min)
//		{
//			min=sum;
//			tab = i; // minFaceIndex
//		}
//	}
//	addNorm(min);
//	if(min<threshold) return tab;
//	else return -1;
//}
//
//
//int AlgoHISTOGRAM::classifierHISTOGRAMKNN(Mat& Hcomp, int k)
//{
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		for(int j = 0; j < nbImgPerFace ; j++)
//		{
//			double c = norm(H[i][j],Hcomp);
//			knearest.addValue(c,i);
//		}
//	}
//
//	return knearest.majorityDecision();
//}
//
//double AlgoHISTOGRAM::mahalanobisDistance(cv::Mat covMatrix,cv::Mat meanMatrix, cv::Mat HcompMatrix)
//{
//	Mat diff(meanMatrix.rows,meanMatrix.cols, CV_64FC1, Scalar(0));
//	Mat temp(diff.cols, diff.cols,CV_64FC1,Scalar(0));
//	Mat invCov(covMatrix.rows,covMatrix.cols,CV_64FC1,Scalar(0));
//	double r;
//
//	for(int i = 0; i < diff.rows; i++)
//		for(int j = 0; j < diff.cols; j++)
//		{
//			diff.at<double>(i,j) =(double) HcompMatrix.at<double>(i,j) - meanMatrix.at<double>(i,j);
//		}
//
//	//invert(covMatrix,invCov,DECOMP_LU);
//	temp = covMatrix*diff;
//	temp = diff.t()*temp;
//
//	r = norm(temp,NORM_L2);
//	//cout << " r = " << r << endl;
//	return r;
//}
//
//template<typename _Tp> int AlgoHISTOGRAM::discriminantFunction(Mat& Hcomp, double threshold){
//
//	int nature = -1;
//	double temp = 0;
//	double min = numeric_limits<double>::max();
//	
//	for(int i = 0; i < nbFaces ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		if(mean.size() == i){
//			mean.push_back(Mat());
//			cov.push_back(Mat());
//			mean[i] = meanMat<_Tp>(H[i]);
//			cov[i] = (featureCovMat<_Tp>(H[i],mean[i])).inv();
//			if(determinant(cov[i])==0){ cout << "OMG ! WE ARE FUCKED !! "<<i<< endl; system("pause"); exit(1); }
//		}
//		
//		temp = mahalanobisDistance(cov[i],mean[i],Hcomp);
//
//		if(temp < min){
//			min = temp;
//			nature = i;
//		}
//	}
//	//cout <<"min:"<<min;
//	addNorm(min);
//	if(min < threshold) return nature;
//	else return -1; // Return the closest class 
//}

vector<vector<cv::Mat>> AlgoHISTOGRAM::featureVectOUT(DBImage& databaseToFeatVect)
{
	vector<vector<cv::Mat>> feactVectors;
	feactVectors.resize(databaseToFeatVect.size());
	for(int i=0; i < databaseToFeatVect.size(); i++)
	{
		feactVectors[i].resize(databaseToFeatVect[i].size());
		for(int j=0; j< databaseToFeatVect[i].size(); j++)
		{
			feactVectors[i][j] = HcompCalc(databaseToFeatVect[i][j]);
		}
	}

	return feactVectors;
}