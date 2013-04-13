#include "algopca.h"
#include <string.h>
#include <time.h>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////////////////////////////
////  Math Operators  --------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
#include "InlineMatrixOperations.cpp"

////////////////////////////////////////////////////////////////////////////////////
////  Constructors  ----------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
AlgoPCA :: AlgoPCA (DBImage &db, double setPct):dbSource(db){
	nbFaces = 0;
	nbImgPerFace = 0;
	nbEigenVectors = 0;
	pourcentageGoal = setPct;
 	//cptFace = 0;
	//cptImgFace = 0;
	minNorm = numeric_limits<double>::max();
	maxNorm = numeric_limits<double>::min();
}

double AlgoPCA::getPourcentage()
{
	return pourcentage;
}

////////////////////////////////////////////////////////////////////////////////////
////  Launchers  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
void AlgoPCA::launch(){ // Avec en paramètre la database après...
	nbFaces = dbSource.getNbrFaces(); // The number of faces we have to load (wrt our database)
	nbImgPerFace = dbSource.getNbrImgPerFace();

	/*if(!faces[0].depth && imgToLoad == MIX)
	{
		cout << "PCA: Error no depth in DBImage and MIX selected.";
		exit(1);
	}*/

	/*if(imgToLoad == DEPTH)
	{
		execute<uchar>();
	}
	else*/ execute/*<double>*/();
}

//#ifdef PRINT_FEATVECT_PCA
//int i_pca_rgb, i_pca_depth, i_pca_mix;
//#endif
//int AlgoPCA::classification(typeDistance dist, typeDecision classif, double param, cv::Mat& image)
//{
//	int returnVal;
//	Mat featVect;
//
//	if(imgToLoad == DEPTH /*|| dbSource.getDbName()=="40f"*/)
//	{
//		featVect = featureCalc<uchar>(image);
//#ifdef PRINT_FEATVECT_VAL_PCA
//		matPrint<double>(featVect,createFVfilename("pca-test-rgb",i_pca_rgb,0)); i_pca_rgb++;
//#endif
//	}
//	else
//	{
//		featVect = featureCalc<double>(image);
//#ifdef PRINT_FEATVECT_VAL_PCA
//		matPrint<double>(featVect,createFVfilename("pca-test-depth",i_pca_depth,0)); i_pca_depth++;
//#endif
//	}
//
//	if(dist==NORM2)
//	{
//		if(classif==THRESHOLD)			returnVal = classifier2DPCA(featVect, param);
//		else if(classif==THRESHOLDSUM)	returnVal = classifier2DPCASum(featVect, param);
//		else if(classif==KNEAREST)		returnVal = classifier2DPCAKNN(featVect, param);
//	}
//	else if(dist==MAHALANOBIS)
//	{
//		// Reservation taille mean/cov vector for Mahalanobis
//		mean.reserve(nbFaces);
//		cov.reserve(nbFaces);
//		if(classif==THRESHOLD || classif==THRESHOLDSUM)	returnVal = discriminantFunction<double>(featVect, param);
//		else if(classif==KNEAREST)		returnVal = discriminantFunctionKNN<double>(featVect, param);
//	}
//
//	return returnVal;
//}
//
//int AlgoPCA::classification(typeDistance dist, typeDecision classif, double param, cv::Mat& rgb, cv::Mat& depth)
//{
//	int returnVal=-1;
//
//	if(imgToLoad == MIX)
//	{
//		Mat featVectRGB = featureCalc<double>(rgb);
//#ifdef PRINT_FEATVECT_VAL_PCA
//		matPrint<double>(featVectRGB,createFVfilename("pca-test-mix-rgb",i_pca_mix,0));
//#endif
//
//		Mat featVectDepth = featureCalc<uchar>(depth);
//#ifdef PRINT_FEATVECT_VAL_PCA
//		matPrint<double>(featVectDepth,createFVfilename("pca-test-mix-depth",i_pca_mix,0)); i_pca_mix++;
//#endif
//
//		if(dist==NORM2)
//		{
//			if(classif==THRESHOLD)			returnVal = classifier2DPCA(featVectRGB, featVectDepth, param);
//			else if(classif==THRESHOLDSUM)	returnVal = classifier2DPCASum(featVectRGB, featVectDepth, param);
//			else if(classif==KNEAREST)		returnVal = classifier2DPCAKNN(featVectRGB, featVectDepth, param);
//		}
//		else if(dist==MAHALANOBIS)
//		{
//			// Reservation taille mean/cov vector for Mahalanobis
//			covRGB.reserve(nbFaces);covDepth.reserve(nbFaces);meanRGB.reserve(nbFaces);meanDepth.reserve(nbFaces);
//
//			if(classif==THRESHOLD || classif==THRESHOLDSUM) returnVal = discriminantFunction<double>(featVectRGB, featVectDepth, param);
//			else if(classif==KNEAREST)		returnVal = discriminantFunctionKNN<double>(featVectRGB, featVectDepth, param);
//		}
//
//	}
//
//	return returnVal;
//}


//int connard=0;
//int AlgoPCA::execute2DPCA(cv::Mat& image){
//	Mat featVect;//char str[50];
//	if(imgToLoad == DEPTH || dbSource.getDbName()=="40f") featVect = featureCalc<uchar>(image);
//	else featVect = featureCalc<double>(image);
//	//matPrint<double>(featVect,string(_itoa(connard,str,10)).append(".txt")); connard++;
//	int a = classifier2DPCA<double>(featVect);
//	return a;
//}
//
//int AlgoPCA::execute2DPCA(cv::Mat& rgb, cv::Mat& depth){
//	Mat featVectRGB = featureCalc<double>(rgb);
//	Mat featVectDepth = featureCalc<uchar>(depth);
//	int a = classifier2DPCA<double>(featVectRGB, featVectDepth);
//	return a;
//}
//
//int AlgoPCA::executeDF(cv::Mat& image){
//	Mat featVect;
//	if(imgToLoad == DEPTH || dbSource.getDbName()=="40f") featVect = featureCalc<uchar>(image);
//	else featVect = featureCalc<double>(image);
//	int idFace = discriminantFunction<double>(featVect);
//	return idFace;
//}
//
//int AlgoPCA::executeDF(cv::Mat& rgb, cv::Mat& depth){
//	Mat featVectRGB = featureCalc<double>(rgb);
//	Mat featVectDepth = featureCalc<uchar>(depth);
//	int idFace = discriminantFunction<double>(featVectRGB, featVectDepth);
//	return idFace;


////////////////////////////////////////////////////////////////////////////////////
////  PCA Comp  --------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
inline cv::Mat AlgoPCA::featureCalc(cv::Mat src)
{
	return matMul<double,double>(EigenVectorsReduced.t(),src);
}

////////////////////////////////////////////////////////////////////////////////////
////  Trainning  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
void AlgoPCA::execute()
{
	clock_t start,ends;
	// 2DPCA - Method
		// Computation of pca eigen pb
	Mat EigenValues(1, dbSource.cols(), CV_64FC1, Scalar(0));
	Mat EigenVectors(dbSource.cols(), dbSource.cols(), CV_64FC1, Scalar(0));

	start = clock(); cout << "AlgoPCA: coomputing eigen values and eigen vectors...";
	pcaEigen(EigenValues,EigenVectors);
	ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;

	//Save the pourcentage of data we keep
	start = clock(); cout << "AlgoPCA: coomputing eigen vectors to keep...";
	nbEigenVectors=0;
	pourcentage=0;
	while(pourcentage<pourcentageGoal)
	{
		nbEigenVectors++;
		pourcentage = computePourcentage(EigenValues);
	}
	ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;
	cout << "AlgoPCA: keeping #" << nbEigenVectors << "/#" << EigenVectors.cols << " eigen vectors. (" << this->getPourcentage()*100 <<"% of data)" << endl;
	//nbEigenVectors=69;

	// We keep only the data required by the PCA
	start = clock(); cout << "AlgoPCA: reducing the number of eigen vectors...";
	EigenVectorsReduced = Mat(dbSource.cols(),nbEigenVectors,CV_64FC1,Scalar(0));
	EigenVectors.colRange(0,nbEigenVectors).copyTo(EigenVectorsReduced);
	ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;

//	// Computation of Feature vectors : Y[i] (one per image)
//	start = clock(); cout << "AlgoPCA: computing all the feature vectors...";
//	Y.resize(nbFaces);
//	if(imgToLoad == RGB)
//	{
//		for(int i = 0; i < nbFaces; i++)
//		{
//			Y[i].reserve(nbImgPerFace);
//			for(int j = 0 ; j < nbImgPerFace ; j++)
//			{
//				Y[i].push_back(Mat(dbSource.rows(), nbEigenVectors, CV_64FC1, Scalar(0)));
//				Y[i][j] = featureCalc<_Tp>(dbSource[i].rgb[j]);
//#ifdef PRINT_FEATVECT_TRAIN_PCA
//				matPrint<double>(Y[i][j],createFVfilename("pca-rgb-Y",i,j));
//#endif
//			}
//		}
//	}
//	else if(imgToLoad == DEPTH)
//	{
//		for(int i = 0; i < nbFaces; i++)
//		{
//			Y[i].reserve(nbImgPerFace);
//			for(int j = 0 ; j < nbImgPerFace ; j++)
//			{
//				Y[i].push_back(Mat(dbSource.rows(), nbEigenVectors, CV_64FC1, Scalar(0)));
//				Y[i][j] = featureCalc<_Tp>(dbSource[i].depth[j]);
//#ifdef PRINT_FEATVECT_TRAIN_PCA
//				matPrint<double>(Y[i][j],createFVfilename("pca-depth-Y",i,j));
//#endif
//			}
//		}
//	}
//	else if(imgToLoad == MIX)
//	{
//		Yp.resize(nbFaces);
//		for(int i = 0; i < nbFaces; i++)
//		{
//			Y[i].reserve(nbImgPerFace);
//			Yp[i].reserve(nbImgPerFace);
//			for(int j = 0 ; j < nbImgPerFace ; j++)
//			{
//				Y[i].push_back(Mat(dbSource.rows(), nbEigenVectors, CV_64FC1, Scalar(0)));
//				Y[i][j] = featureCalc<_Tp>(dbSource[i].rgb[j]);
//#ifdef PRINT_FEATVECT_TRAIN_PCA
//				matPrint<double>(Y[i][j],createFVfilename("pca-mix-Yrgb",i,j));
//#endif
//				Yp[i].push_back(Mat(dbSource.rows(), nbEigenVectors, CV_64FC1, Scalar(0)));
//				Yp[i][j] = featureCalc<uchar>(dbSource[i].depth[j]);
//#ifdef PRINT_FEATVECT_TRAIN_PCA
//				matPrint<double>(Yp[i][j],createFVfilename("pca-mix-Ydepth",i,j));
//#endif
//			}
//		}
//	}
//	ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;
	//END Execute
}

void AlgoPCA::pcaEigen(cv::Mat& EigenValues, cv::Mat& EigenVectors)
{
	// Computation of the Mean Matrix
	//cout <<"mean";
	Mat mean  (dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0));
	pcaMean(mean); //Compute the mean between all images faces[*].images[*]

	// 2DPCA - Method
	// Computation of Covariance Matrix : G
	//cout <<"cov";
	Mat G     (dbSource.rows(), dbSource.rows(), CV_64FC1, Scalar(0));
	pcaCovariance(G,mean);
	
	// Computation of eigenval/vec of G
	//cout <<"eigen";
	eigen(G,EigenValues,EigenVectors); // already in descending order.
}

void AlgoPCA::pcaMean(cv::Mat& mean)
{
	int total=0;
	/*if(imgToLoad == RGB)
	{*/
		for(int k = 0 ; k < dbSource.size() ; k++)
			for(int l = 0 ; l < dbSource[k].size() ; l++)
			{
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					mean.at<double>(i,j)+=dbSource[k]/*.rgb*/[l].at</*_Tp*/double>(i,j);
				total++;
			}
	/*}
	else if(imgToLoad == DEPTH)
	{
		for(int k = 0 ; k < nbFaces ; k++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					mean.at<double>(i,j)+=dbSource[k].depth[l].at<_Tp>(i,j);
	}
	else if(imgToLoad == MIX)
	{
		for(int k = 0 ; k < nbFaces ; k++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
				{
					mean.at<double>(i,j)+=dbSource[k].mix[l].at<_Tp>(i,j);
				}
	}*/
	mean/=(double)(total);
}

void AlgoPCA::pcaCovariance(cv::Mat& G, const cv::Mat& mean)
{
	Mat temp  (dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0));
	int total=0;
	/*if(imgToLoad == RGB)
	{*/
		for(int k = 0 ; k < dbSource.size() ; k ++)
			for(int l = 0 ; l < dbSource[k].size() ; l++)
			{
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					 temp.at<double>(i,j) = dbSource[k][l].at<double>(i,j) - mean.at<double>(i,j);
				total++;
				G+=temp*temp.t();
			}
	/*}
	else if(imgToLoad == DEPTH)
	{
		for(int k = 0 ; k < nbFaces ; k ++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
			{
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					 temp.at<double>(i,j) = dbSource[k].depth[l].at<_Tp>(i,j) - mean.at<double>(i,j);

				G+=temp.t()*temp;
			}
	}
	else if(imgToLoad == MIX)
	{
		for(int k = 0 ; k < nbFaces ; k ++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
			{
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
				{
					 temp.at<double>(i,j) = dbSource[k].mix[l].at<_Tp>(i,j)-mean.at<double>(i,j);
				}
				G+=temp.t()*temp;
			}
	}*/

	G /= (double) (total);
}

double AlgoPCA::computePourcentage(cv::Mat& EigenValues)
{
	if(nbEigenVectors<=(EigenValues.rows>EigenValues.cols?EigenValues.rows:EigenValues.cols))
	{
		double keep=0;
		for(int i = 0; i < nbEigenVectors; i++)
			keep += EigenValues.at<double>(i);

		double total=keep;
		for(int i = nbEigenVectors;i < (EigenValues.rows>EigenValues.cols?EigenValues.rows:EigenValues.cols);i++)
			total += EigenValues.at<double>(i);
		//cout << "k"<<keep<<" t"<<total<<endl;
		return (total? keep/total : -1.0); //Avoid div by 0
	}
	else return -1.0; //Avoid crazy size
}


////////////////////////////////////////////////////////////////////////////////////
////  Classification  --------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
//int AlgoPCA::classifier2DPCA(Mat& featureVector, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//	double tempVal;
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		for(int j = 0; j < nbImgPerFace ; j++)
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
//}
//int AlgoPCA::classifier2DPCASum(Mat& featureVector, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		double sum=0;
//		for(int j = 0; j < nbImgPerFace ; j++)
//		{
//			sum += norm(Y[i][j],featureVector,NORM_L1);
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
//int AlgoPCA::classifier2DPCAKNN(Mat& featureVector, int k)
//{
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		for(int j = 0; j < nbImgPerFace ; j++)
//		{
//			double c = norm(Y[i][j],featureVector);
//			knearest.addValue(c,i);
//		}
//	}
//
//	return knearest.majorityDecision();
//}
//
//int AlgoPCA::classifier2DPCA(Mat& featureVectorRGB, Mat& featureVectorDepth, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//	double tempVal;
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		for(int j = 0; j < nbImgPerFace ; j++)
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
//int AlgoPCA::classifier2DPCASum(Mat& featureVectorRGB, Mat& featureVectorDepth, double threshold)
//{
//	int tab=-1;
//	double min = numeric_limits<double>::max();
//	double tempVal;
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		tempVal=0;
//		for(int j = 0; j < nbImgPerFace ; j++)
//		{
//			tempVal+=norm(Y[i][j],featureVectorRGB)+norm(Yp[i][j],featureVectorDepth);
//		}
//		
//		if(tempVal<min)
//		{
//			min=tempVal;
//			tab = i; // minFaceIndex
//		}
//
//	}
//	addNorm(min);
//	if(min<threshold) return tab;
//	return -1;
//}
//int AlgoPCA::classifier2DPCAKNN(Mat& featureVectorRGB, Mat& featureVectorDepth, int k)
//{
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbFaces; i++)
//	{
//		for(int j = 0; j < nbImgPerFace ; j++)
//		{
//			double c = norm(Y[i][j],featureVectorRGB)+norm(Yp[i][j],featureVectorDepth);
//			knearest.addValue(c,i);
//		}
//	}
//
//	return knearest.majorityDecision();
//}
//
//double AlgoPCA::mahalanobisDistance(cv::Mat covMatrix,cv::Mat meanMatrix, cv::Mat featureMatrix)
//{
//	Mat diff(meanMatrix.rows,meanMatrix.cols, CV_64FC1, Scalar(0));
//	Mat temp(diff.cols, diff.cols,CV_64FC1,Scalar(0));
//	Mat invCov(covMatrix.rows,covMatrix.cols,CV_64FC1,Scalar(0));
//	double r;
//
//	for(int i = 0; i < diff.rows; i++)
//		for(int j = 0; j < diff.cols; j++)
//		{
//			diff.at<double>(i,j) = (double) featureMatrix.at<double>(i,j) - meanMatrix.at<double>(i,j);
//		}
//
//	//invert(covMatrix,invCov,DECOMP_LU);
//	temp = diff.t()*(covMatrix*diff);
//
//	r = norm(temp,NORM_L2);
//	//cout << " r = " << r << endl;
//	return r;
//}
//
//template<typename _Tp> int AlgoPCA::discriminantFunction(Mat& featureVector, double threshold){
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
//		
//	}
//	//cout <<"min:"<<min;
//	addNorm(min);
//	if(min < threshold) return nature;
//	else return -1; // Return the closest class 
//}
//template<typename _Tp> int AlgoPCA::discriminantFunctionKNN(Mat& featureVector, int k){
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbFaces ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		if(mean.size() == i){
//			mean.push_back(Mat());
//			cov.push_back(Mat());
//			mean[i] = meanMat<_Tp>(Y[i]);
//			cov[i] = (featureCovMat<_Tp>(Y[i],mean[i]));
//		}
//
//		knearest.addValue(mahalanobisDistance(cov[i],mean[i],featureVector), i);
//	}
//	//cout <<"rmin:"<<r;//<<" rmax:"<<rmax;
//
//	return knearest.majorityDecision(); // Return the closest class 
//}
//
//template<typename _Tp> int AlgoPCA::discriminantFunction(Mat& featureVectorRGB, Mat& featureVectorDepth, double threshold){
//
//	int nature = -1;
//	double tempRGB = 0, tempDepth = 0;
//	double min = numeric_limits<double>::max();
//	for(int i = 0; i < nbFaces ; i ++) // For each face
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
//
//	}
//	addNorm(min);
//	if (min<threshold) return nature; // Return the closest class 
//	else return -1;
//}
//template<typename _Tp> int AlgoPCA::discriminantFunctionKNN(Mat& featureVectorRGB, Mat& featureVectorDepth, int k){
//	Mat covRGB, covDepth;
//	Mat meanRGB, meanDepth;
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbFaces ; i ++) // For each face
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

vector<vector<cv::Mat>> AlgoPCA::featureVectOUT(DBImage& databaseToFeatVect)
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