#include "algolda.h"
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
AlgoLDA::AlgoLDA(DBImage &db/*, typeLoading toLoad*/):dbSource(db), faces(db){
	nbrclass = 0;
	nbrimageperclass = 0;
	/*cptFace = 0;
	cptImgFace = 0;*/
	//imgToLoad = toLoad;
	minNorm = numeric_limits<double>::max();
	maxNorm = numeric_limits<double>::min();
}

////////////////////////////////////////////////////////////////////////////////////
////  Launchers  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
void AlgoLDA::launch(){
	nbrclass = dbSource.getNbrFaces(); // The number of faces we have to load (wrt our database)
	nbrimageperclass = dbSource.getNbrImgPerFace();
	numFisher = nbrclass - 1;
	/*if(!faces[0].depth && imgToLoad == MIX)
	{
		cout << "PCA: Error no depth in DBImage and MIX selected.";
		exit(1);
	}*/

	//if(imgToLoad == DEPTH /*|| dbSource.getDbName()=="40f"*/)
	//{
	//	execute<uchar>();
	//}
	/*else*/ execute();
}

//#ifdef PRINT_FEATVECT_VAL_LDA
//int i_lda_rgb, i_lda_depth, i_lda_mix;
//#endif
//int AlgoLDA::classification(typeDistance dist, typeDecision classif, double param, cv::Mat& image)
//{
//	int returnVal;
//	Mat featVect;
//
//	if(imgToLoad == DEPTH /*|| dbSource.getDbName()=="40f"*/)
//	{
//		featVect = featureCalc<uchar>(image);
//#ifdef PRINT_FEATVECT_VAL_LDA
//		matPrint<double>(featVect,createFVfilename("lda-test-rgb",i_lda_rgb,0));  i_lda_rgb++;
//#endif
//	}
//	else
//	{
//		featVect = featureCalc<double>(image);
//#ifdef PRINT_FEATVECT_VAL_LDA
//		matPrint<double>(featVect,createFVfilename("lda-test-depth",i_lda_depth,0)); i_lda_depth++;
//#endif
//	}
//
//	if(dist==NORM2)
//	{
//		if(classif==THRESHOLD)			returnVal = classifier2DLDA(featVect, param);
//		else if(classif==THRESHOLDSUM)	returnVal = classifier2DLDASum(featVect, param);
//		else if(classif==KNEAREST)		returnVal = classifier2DLDAKNN(featVect, param);
//	}
//	else if(dist==MAHALANOBIS)
//	{
//		mean.reserve(nbrclass);
//		cov.reserve(nbrclass);
//		if(classif==THRESHOLD || classif==THRESHOLDSUM)	returnVal = discriminantFunction<double>(featVect, param);
//		else if(classif==KNEAREST)		returnVal = discriminantFunctionKNN<double>(featVect, param);
//	}
//
//	return returnVal;
//}
//
//int AlgoLDA::classification(typeDistance dist, typeDecision classif, double param, cv::Mat& rgb, cv::Mat& depth)
//{
//	int returnVal=-1;
//
//	if(imgToLoad == MIX)
//	{
//		Mat featVectRGB = featureCalc<double>(rgb);
//#ifdef PRINT_FEATVECT_VAL_LDA
//		matPrint<double>(featVectRGB,createFVfilename("lda-test-mix-rgb",i_lda_mix,0));
//#endif
//		Mat featVectDepth = featureCalc<uchar>(depth);
//#ifdef PRINT_FEATVECT_VAL_LDA
//		matPrint<double>(featVectDepth,createFVfilename("lda-test-mix-depth",i_lda_mix,0)); i_lda_mix++;
//#endif
//
//		if(dist==NORM2)
//		{
//			if(classif==THRESHOLD)			returnVal = classifier2DLDA(featVectRGB, featVectDepth, param);
//			else if(classif==THRESHOLDSUM)	returnVal = classifier2DLDASum(featVectRGB, featVectDepth, param);
//			else if(classif==KNEAREST)		returnVal = classifier2DLDAKNN(featVectRGB, featVectDepth, param);
//		}
//		else if(dist==MAHALANOBIS)
//		{
//			covRGB.reserve(nbrclass);covDepth.reserve(nbrclass);meanRGB.reserve(nbrclass);meanDepth.reserve(nbrclass);
//
//			if(classif==THRESHOLD || classif==THRESHOLDSUM) returnVal = discriminantFunction<double>(featVectRGB, featVectDepth, param);
//			else if(classif==KNEAREST)		returnVal = discriminantFunctionKNN<double>(featVectRGB, featVectDepth, param);
//		}
//
//	}
//
//	return returnVal;
//}

////////////////////////////////////////////////////////////////////////////////////
////  LDA Comp  --------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
inline cv::Mat AlgoLDA::featureCalc(cv::Mat src)
{
	return matMul<double,double>(EigV.t(),src);
}
vector<vector<cv::Mat>> AlgoLDA::featureVectOUT(DBImage& databaseToFeatVect)
{
	vector<vector<cv::Mat>> feactVectors;
	feactVectors.resize(databaseToFeatVect.size());
	for(int i=0; i < databaseToFeatVect.size(); i++)
	{
		feactVectors[i].resize(databaseToFeatVect[i].size());
		for(int j=0; j< databaseToFeatVect[i].size(); j++)
		{
			feactVectors[i][j] = featureCalc(databaseToFeatVect[i][j]);
		}
	}

	return feactVectors;
}


////////////////////////////////////////////////////////////////////////////////////
////  Trainning  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
void AlgoLDA::execute(){

	Mat EigenValues(1        , faces.rows(), CV_64FC1, Scalar(0));
	Mat EigenVectors(faces.rows(), faces.rows(), CV_64FC1, Scalar(0));
	EigV =  Mat(faces.rows(), faces.rows()-1, CV_64FC1, Scalar(0));
	
	ldaEigen(EigenValues,EigenVectors);
	if(numFisher<EigenVectors.cols) EigV=EigenVectors.colRange(0,numFisher);
	else EigV=EigenVectors;
//	Y.resize(nbrclass);
//	// Computation of Feature vectors : Y[i] (one per image)
//
//		for(int i = 0; i < nbrclass; i++)
//		{
//			Y[i].reserve(nbrimageperclass);
//			for(int j = 0 ; j < nbrimageperclass ; j++)
//			{
//				Y[i].push_back(Mat(faces.rows(), numFisher, CV_64FC1, Scalar(0)));
//				Y[i][j] = featureCalc(faces[i].rgb[j]);
//#ifdef PRINT_FEATVECT_TRAIN_LDA
//				matPrint<double>(Y[i][j],createFVfilename("lda-rgb-Y",i,j));
//#endif
//			}
//		}
//	}
//	else if(imgToLoad == DEPTH)
//	{
//		for(int i = 0; i < nbrclass; i++)
//		{
//			Y[i].reserve(nbrimageperclass);
//			for(int j = 0 ; j < nbrimageperclass ; j++)
//			{
//				Y[i].push_back(Mat(faces.rows(), numFisher, CV_64FC1, Scalar(0)));
//				Y[i][j] = featureCalc(faces[i].depth[j]);
//#ifdef PRINT_FEATVECT_TRAIN_LDA
//				matPrint<double>(Y[i][j],createFVfilename("lda-depth-Y",i,j));
//#endif
//			}
//		}
//	}
//	else if(imgToLoad == MIX)
//	{
//		Yp.resize(nbrclass);
//		for(int i = 0; i < nbrclass; i++)
//		{
//			Y[i].reserve(nbrimageperclass);
//			Yp[i].reserve(nbrimageperclass);
//			for(int j = 0 ; j < nbrimageperclass ; j++)
//			{
//				Y[i].push_back(Mat(faces.rows(), numFisher, CV_64FC1, Scalar(0)));
//				Y[i][j] = featureCalc<_Tp>(faces[i].rgb[j]);
//#ifdef PRINT_FEATVECT_TRAIN_LDA
//				matPrint<double>(Y[i][j],createFVfilename("lda-mix-Yrgb",i,j));
//#endif
//				Yp[i].push_back(Mat(faces.rows(), numFisher, CV_64FC1, Scalar(0)));
//				Yp[i][j] = featureCalc<uchar>(faces[i].depth[j]);
//#ifdef PRINT_FEATVECT_TRAIN_LDA
//				matPrint<double>(Y[i][j],createFVfilename("lda-mix-Ydepth",i,j));
//#endif
//			}
//		}
//	}
}

void AlgoLDA::ldaEigen(cv::Mat& EigenValues, cv::Mat& EigenVectors)
{
	// Computation of the Mean Matrix and the Sub-mean Matrix
	Mat m  (faces.rows(), faces.cols(), CV_64FC1, Scalar(0));
	vector<Mat> mi; 
	mi.reserve(nbrclass);
	for(int i=0;i<nbrclass;i++) mi.push_back(Mat(faces.rows(), faces.cols(), CV_64FC1, Scalar(0)));
	ldaMeanSubMean(m,mi); //Compute the mean between all images faces[*].images[*]

	
	// Computation of Scatter Between Class Matrix
	Mat Sb     (faces.rows(), faces.rows(), CV_64FC1, Scalar(0));
	ldaScatterBetweenMat(mi,m,Sb);
	
	// Computation of Scatter Within Class Matrix
	Mat Sw    (faces.rows(), faces.rows(), CV_64FC1, Scalar(0));
	ldaScatterWithinMat(mi,Sw);
	
	// Calcul of W
	Mat SwInv(faces.rows(), faces.rows(), CV_64FC1, Scalar(0));
	Mat W(faces.rows(), faces.rows(), CV_64FC1, Scalar(0));
	SwInv = Sw.inv();
	W = matMul<double,double>(SwInv,Sb);

	// Computation of eigenval/vec of W
	eigen(W,EigenValues,EigenVectors); // already in descending order.
}

void AlgoLDA::ldaMeanSubMean(cv::Mat& m,std::vector<cv::Mat>& mi)
{
	for(int k = 0 ; k < nbrclass ; k++)
	{
		for(int l = 0 ; l < nbrimageperclass ; l++)
		{
			for(int i = 0 ; i < m.rows ;i++) 
				for(int j = 0; j < m.cols ;j++)
				{
					m.at<double>(i,j)+=faces[k][l].at<double>(i,j);
					mi[k].at<double>(i,j)+=faces[k][l].at<double>(i,j);
				}
			// Computation of the Sub Mean Matrix
		}
		mi[k]/=(double)(nbrimageperclass);
	}
	
	m/=(double)(nbrclass*nbrimageperclass);
}

void AlgoLDA::ldaScatterBetweenMat(const std::vector<cv::Mat>& mi, cv::Mat& m,cv::Mat& Sb){ // For one face
	
	Mat temp  (mi[0].rows, mi[0].cols, CV_64FC1, Scalar(0));
	Mat G     (mi[0].rows, mi[0].rows, CV_64FC1, Scalar(0));
	//Mat Sb    (mi[0].rows, mi[0].rows, CV_64FC1, Scalar(0));
	// Computation of Scatter Between Class Matrix
	for(int o = 0; o < nbrclass; o++)
	{
		for(int i = 0 ; i < m.rows ;i++) for(int j = 0; j < m.cols ;j++)
				temp.at<double>(i,j) = mi[o].at<double>(i,j) - m.at<double>(i,j);
			G+=temp*temp.t();
		G *= (double) nbrimageperclass;
		Sb=Sb+G;
	}
}

void AlgoLDA::ldaScatterWithinMat(const std::vector<cv::Mat>& mi,cv::Mat& Sw){ // For one face
	
	Mat temp  (mi[0].rows, mi[0].cols, CV_64FC1, Scalar(0));
	Mat G     (mi[0].rows, mi[0].rows, CV_64FC1, Scalar(0));
	// Computation of Scatter Within Class Matrix
	for(int o=0; o < nbrclass; o++)
	{
		for(int k = 0 ; k < nbrimageperclass ; k ++)
		{
			for(int i = 0 ; i < mi[0].rows ;i++) for(int j = 0; j < mi[0].cols ;j++)
						temp.at<double>(i,j) = faces[o][k].at<double>(i,j) - mi[o].at<double>(i,j);
			G+=temp*temp.t();
		}
		Sw=Sw+G;
		G = Mat::zeros(G.rows,G.cols,CV_64FC1);
	}
}

////////////////////////////////////////////////////////////////////////////////////
////  Classification  --------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
//int AlgoLDA::classifier2DLDA(Mat& featureVector, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//	double tempVal;
//
//	for(int i = 0; i < nbrclass; i++)
//	{
//		for(int j = 0; j < nbrimageperclass ; j++)
//		{
//			tempVal=norm(Y[i][j],featureVector,NORM_L1);
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
//
//}
//int AlgoLDA::classifier2DLDASum(Mat& featureVector, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//
//	for(int i = 0; i < nbrclass; i++)
//	{
//		double sum=0;
//		for(int j = 0; j < nbrimageperclass ; j++)
//		{
//			sum += norm(Y[i][j],featureVector);
//		}
//		sum /= nbrimageperclass;
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
//int AlgoLDA::classifier2DLDAKNN(Mat& featureVector, int k)
//{
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbrclass; i++)
//	{
//		for(int j = 0; j < nbrimageperclass ; j++)
//		{
//			double c = norm(Y[i][j],featureVector);
//			knearest.addValue(c,i);
//		}
//	}
//
//	return knearest.majorityDecision();
//}
//
//int AlgoLDA::classifier2DLDA(Mat& featureVectorRGB, Mat& featureVectorDepth, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//	double tempVal;
//
//	for(int i = 0; i < nbrclass; i++)
//	{
//		for(int j = 0; j < nbrimageperclass ; j++)
//		{
//			tempVal=norm(Y[i][j],featureVectorRGB,NORM_L1)+norm(Yp[i][j],featureVectorDepth,NORM_L1);
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
//	return -1;
//}
//int AlgoLDA::classifier2DLDASum(Mat& featureVectorRGB, Mat& featureVectorDepth, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//	double tempVal;
//
//	for(int i = 0; i < nbrclass; i++)
//	{
//		tempVal=0;
//		for(int j = 0; j < nbrimageperclass ; j++)
//		{
//			tempVal+=norm(Y[i][j],featureVectorRGB)+norm(Yp[i][j],featureVectorDepth);
//		}
//		
//		if(tempVal<min)
//		{
//			min=tempVal;
//			tab = i; // minFaceIndex
//		}
//	}
//	addNorm(min);
//	if(min<threshold) return tab;
//	return -1;
//}
//int AlgoLDA::classifier2DLDAKNN(Mat& featureVectorRGB, Mat& featureVectorDepth, int k)
//{
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbrclass; i++)
//	{
//		for(int j = 0; j < nbrimageperclass ; j++)
//		{
//			double c = norm(Y[i][j],featureVectorRGB)+norm(Yp[i][j],featureVectorDepth);
//			knearest.addValue(c,i);
//		}
//	}
//
//	return knearest.majorityDecision();
//}
//
//double AlgoLDA::mahalanobisDistance(cv::Mat covMatrix,cv::Mat meanMatrix, cv::Mat featureMatrix)
//{
//	Mat diff(meanMatrix.rows,meanMatrix.cols, CV_64FC1, Scalar(0));
//	Mat temp(diff.cols, diff.cols,CV_64FC1,Scalar(0));
//	//Mat invCov(covMatrix.rows,covMatrix.cols,CV_64FC1,Scalar(0));
//	double r;
//
//	for(int i = 0; i < diff.rows; i++)
//		for(int j = 0; j < diff.cols; j++)
//		{
//			diff.at<double>(i,j) =(double) featureMatrix.at<double>(i,j) - meanMatrix.at<double>(i,j);
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
//template<typename _Tp> int AlgoLDA::discriminantFunction(Mat& featureVector, double threshold){
//	int nature = -1;
//	double temp = 0;
//	double min = numeric_limits<double>::max();
//	
//	for(int i = 0; i < nbrclass ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		if(mean.size() == i){
//			mean.push_back(Mat());
//			cov.push_back(Mat());
//			mean[i] = meanMat<_Tp>(Y[i]);
//			cov[i] = (featureCovMat<_Tp>(Y[i],mean[i])).inv();
//			if(determinant(cov[i])==0){ cout << "OMG ! WE ARE FUCKED !! "<<i<< endl; system("pause"); exit(1); }
//		}
//
//		temp = mahalanobisDistance(cov[i],mean[i],featureVector);
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
//template<typename _Tp> int AlgoLDA::discriminantFunctionKNN(Mat& featureVector, int k){
//	Mat cov;
//	Mat mean;
//
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbrclass ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		mean = meanMat<_Tp>(Y[i]);
//		cov = featureCovMat<_Tp>(Y[i],mean);
//
//		knearest.addValue(mahalanobisDistance(cov,mean,featureVector), i);
//	}
//	//cout <<"rmin:"<<r;//<<" rmax:"<<rmax;
//
//	return knearest.majorityDecision(); // Return the closest class 
//}
//
//template<typename _Tp> int AlgoLDA::discriminantFunction(Mat& featureVectorRGB, Mat& featureVectorDepth, double threshold){
//	int nature = -1;
//	double tempRGB = 0, tempDepth = 0;
//	double min = numeric_limits<double>::max();
//	for(int i = 0; i < nbrclass ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		if(meanRGB.size() == i){
//			meanRGB.push_back(Mat());
//			covRGB.push_back(Mat());
//			meanDepth.push_back(Mat());
//			covDepth.push_back(Mat());
//			meanRGB[i] = meanMat<_Tp>(Y[i]);
//			covRGB[i] = (featureCovMat<_Tp>(Y[i],meanRGB[i])).inv();
//			meanDepth[i] = meanMat<_Tp>(Yp[i]);
//			covDepth[i] = (featureCovMat<_Tp>(Yp[i],meanDepth[i])).inv();
//		}
//
//		tempRGB = mahalanobisDistance(covRGB[i],meanRGB[i],featureVectorRGB);
//		tempDepth = mahalanobisDistance(covDepth[i],meanDepth[i],featureVectorDepth);
//		
//		if(tempRGB+tempDepth < min){
//			min = tempRGB+tempDepth;
//			nature = i;
//		}
//	}
//	addNorm(min);
//	if (min<threshold) return nature; // Return the closest class 
//	else return -1;
//}
//template<typename _Tp> int AlgoLDA::discriminantFunctionKNN(Mat& featureVectorRGB, Mat& featureVectorDepth, int k){
//	Mat covRGB, covDepth;
//	Mat meanRGB, meanDepth;
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbrclass ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		meanRGB = meanMat<_Tp>(Y[i]);
//		covRGB = featureCovMat<_Tp>(Y[i],meanRGB);
//		meanDepth = meanMat<_Tp>(Yp[i]);
//		covDepth = featureCovMat<_Tp>(Yp[i],meanDepth);
//
//		knearest.addValue(	mahalanobisDistance(covRGB,meanRGB,featureVectorRGB)+mahalanobisDistance(covDepth,meanDepth,featureVectorDepth),
//							i);
//	}
//
//	return knearest.majorityDecision(); // Return the closest class 
//}

