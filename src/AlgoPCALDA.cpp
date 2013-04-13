#include "AlgoPCALDA.h"
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
AlgoPCALDA :: AlgoPCALDA (DBImage &db, double setPct/*, typeLoading toLoad*/):dbSource(db){
	nbFaces = 0;
	nbImgPerFace = 0;
	nbrclass = 0;
	nbrimageperclass = 0;
	nbEigenVectors = 0;
	pourcentageGoal = setPct;
 	//cptFace = 0;
	//cptImgFace = 0;
	//imgToLoad = toLoad;
	//minNorm = numeric_limits<double>::max();
	//maxNorm = numeric_limits<double>::min();
}

double AlgoPCALDA::getPourcentage()
{
	return pourcentage;
}

////////////////////////////////////////////////////////////////////////////////////
////  Launchers  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
void AlgoPCALDA::launch(){ // Avec en paramètre la database après...
	nbFaces = dbSource.getNbrFaces(); // The number of faces we have to load (wrt our database)
	nbrclass = nbFaces;// The number of faces we have to load (wrt our database)
	nbImgPerFace = dbSource.getNbrImgPerFace();
	nbrimageperclass = nbImgPerFace;
	numFisher = nbrclass - 1;
	/*if(!faces[0].depth && imgToLoad == MIX)
	{
		cout << "PCA: Error no depth in DBImage and MIX selected.";
		exit(1);
	}*/

	/*if(imgToLoad == DEPTH)
	{
		execute<uchar>();
	}
	else */execute<double>();
}

//int i_pca_rgb, i_pca_depth, i_pca_mix;
////int i_lda_rgb, i_lda_depth, i_lda_mix;
//
//int AlgoPCA::classification(typeDistance dist, typeDecision classif, double param, cv::Mat& image)
//{
//	int returnVal;
//	Mat featVectTrans(image.rows,image.cols, CV_64FC1, Scalar(0));
//	Mat featVect;
//
//	if(imgToLoad == DEPTH /*|| dbSource.getDbName()=="40f"*/)
//	{
//		for(int k = 0 ; k < image.rows ;k++) 
//			for(int l = 0; l < image.cols ;l++)
//			{
//				featVectTrans.at<double>(k,l) = image.at<uchar>(k,l) - m.at<double>(k,l);
//			}
//		featVect = featureCalcLDA<double>(featVectTrans);
//		//matPrint<double>(featVect,createFVfilename("pca-test-rgb",i_pca_rgb,0)); i_pca_rgb++;
//	}
//	else
//	{
//		featVectTrans = image - m;
//		featVect = featureCalcLDA<double>(featVectTrans);
//		//matPrint<double>(featVect,createFVfilename("pca-test-depth",i_pca_depth,0)); i_pca_depth++;
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
//	Mat featVectRGBTrans;
//	Mat featVectRGB;
//	Mat featVectDepthTrans(depth.rows,depth.cols, CV_64FC1, Scalar(0));
//	Mat featVectDepth;
//
//	//cout << "classification MIX" << endl;
//	if(imgToLoad == MIX)
//	{
//		//cout << "classification MIX 1" << endl;
//		featVectRGBTrans = rgb - m;
//		featVectRGB = featureCalcLDA<double>(featVectRGBTrans);
//		//matPrint<double>(featVectRGB,createFVfilename("pca-test-mix-rgb",i_pca_mix,0));
//		for(int k = 0 ; k < depth.rows ;k++) 
//			for(int l = 0; l < depth.cols ;l++)
//			{
//				featVectDepthTrans.at<double>(k,l) = depth.at<uchar>(k,l) - m.at<double>(k,l);
//			}
//		featVectDepth = featureCalcLDA<double>(featVectDepthTrans);
//		//matPrint<double>(featVectDepth,createFVfilename("pca-test-mix-depth",i_pca_mix,0)); i_pca_mix++;
//
//		if(dist==NORM2)
//		{
//			if(classif==THRESHOLD)			returnVal = classifier2DPCA(featVectRGB, featVectDepth, param);
//			else if(classif==THRESHOLDSUM)	returnVal = classifier2DPCASum(featVectRGB, featVectDepth, param);
//			else if(classif==KNEAREST)		returnVal = classifier2DPCAKNN(featVectRGB, featVectDepth, param);
//		}
//		else if(dist==MAHALANOBIS)
//		{
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
template<typename _Tp> inline cv::Mat AlgoPCALDA::featureCalc(cv::Mat src)
{
	//cout << "EigenVectorsReduced : " << EigenVectorsReduced.rows << "-" << EigenVectorsReduced.cols << "  /  src : " << src.rows << "-" << src.cols << endl;
	return matMul<_Tp,double>(src,EigenVectorsReduced);
}
////////////////////////////////////////////////////////////////////////////////////
////  LDA Comp  --------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
template<typename _Tp> inline cv::Mat AlgoPCALDA::featureCalcLDA(cv::Mat src)
{
	return matMul<double,_Tp>(EigVL.t(),src);
}

vector<vector<cv::Mat>> AlgoPCALDA::featureVectOUT(DBImage& databaseToFeatVect)
{
	vector<vector<cv::Mat>> feactVectors;
	feactVectors.resize(databaseToFeatVect.size());
	for(int i=0; i < databaseToFeatVect.size(); i++)
	{
		feactVectors[i].resize(databaseToFeatVect[i].size());
		for(int j=0; j< databaseToFeatVect[i].size(); j++)
		{
			feactVectors[i][j] = featureCalcLDA<double>(databaseToFeatVect[i][j]-m);
		}
	}

	return feactVectors;
}

////////////////////////////////////////////////////////////////////////////////////
////  Trainning  -------------------------------------------------------------- ////
////////////////////////////////////////////////////////////////////////////////////
template<typename _Tp> void AlgoPCALDA::execute()
{
	////////////////////////////////////////////////////////////////////////////////////
	////  PCA  -------------------------------------------------------------- //////////
	////////////////////////////////////////////////////////////////////////////////////

	clock_t start,ends;
	// 2DPCA - Method
		// Computation of pca eigen pb
	Mat EigenValues(1, dbSource.rows(), CV_64FC1, Scalar(0));
	Mat EigenVectors(dbSource.rows(), dbSource.rows(), CV_64FC1, Scalar(0));

	//start = clock(); cout << "AlgoPCA: coomputing eigen values and eigen vectors...";
	pcaEigen<_Tp>(EigenValues,EigenVectors);
	//ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;

	//Save the pourcentage of data we keep
	//start = clock(); cout << "AlgoPCA: coomputing eigen vectors to keep...";
	nbEigenVectors=0;
	pourcentage=0;
	while(pourcentage<pourcentageGoal)
	{
		nbEigenVectors++;
		pourcentage = computePourcentage(EigenValues);
	}
	//ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;
	nbEigenVectors = 348;
	//cout << "AlgoPCA: keeping #" << nbEigenVectors << "/#" << EigenVectors.cols << " eigen vectors. (" << this->getPourcentage()*100 <<"% of data)" << endl;

	// We keep only the data required by the PCA
	//start = clock(); cout << "AlgoPCA: reducing the number of eigen vectors...";
	EigenVectorsReduced = Mat(dbSource.rows(),nbEigenVectors,CV_64FC1,Scalar(0));
	EigenVectors.colRange(0,nbEigenVectors).copyTo(EigenVectorsReduced);
	//ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;

	////////////////////////////////////////////////////////////////////////////////////
	////  LDA  -------------------------------------------------------------- //////////
	////////////////////////////////////////////////////////////////////////////////////

	//start = clock(); cout << "AlgoLDA: computing eigenvalues and eigenvectors...";

	Mat EigenValuesLDA(1        ,nbEigenVectors, CV_64FC1, Scalar(0));
	Mat EigenVectorsLDA(nbEigenVectors, nbEigenVectors, CV_64FC1, Scalar(0));
	

	ldaEigen<_Tp>(EigenValuesLDA,EigenVectorsLDA);
	//matPrint<double>(m,createFVfilename("pcalda-depth-m1",0,0));
	/*if(imgToLoad == MIX)
	{
		ldaEigenP<double>(EigenValuesLDAP,EigenVectorsLDAP);
	}*/
	EigV =  Mat(EigenVectorsLDA.rows, numFisher, CV_64FC1, Scalar(0));
	EigVL =  Mat(dbSource.rows(), numFisher, CV_64FC1, Scalar(0));

	//ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;
	//start = clock(); cout << "AlgoLDA: Keeping numfisher eigenvectors...";

	if(numFisher<EigenVectorsLDA.cols) EigV=EigenVectorsLDA.colRange(0,numFisher);
	else EigV=EigenVectorsLDA;
	/*if(imgToLoad == MIX)
	{
		if(numFisher<EigenVectorsLDAP.cols) EigVP=EigenVectorsLDAP.colRange(0,numFisher);
		else EigVP=EigenVectorsLDAP;
	}*/
	/*cout << "EigenVectorsReduced : " << EigenVectorsReduced.rows << " " << EigenVectorsReduced.cols << endl;
	cout << "EigV : " << EigV.rows << " " << EigV.cols << endl;*/

	EigVL = EigenVectorsReduced*EigV;
	
	//ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;
	//start = clock(); cout << "AlgoLDA: computing all the feature vectors...";
	//Y.resize(nbrclass);
	//dbTransDepth.resize(nbrclass);
	//dbTransRGB.resize(nbrclass);
	//dbTrans.resize(nbrclass);
	//// Computation of Feature vectors : Y[i] (one per image)
	//if(imgToLoad == RGB)
	//{
	//	for(int i = 0; i < nbrclass; i++)
	//	{
	//		Y[i].reserve(nbrimageperclass);
	//		for(int j = 0 ; j < nbrimageperclass ; j++)
	//		{
	//			Y[i].push_back(Mat(dbSource.rows(), numFisher, CV_64FC1, Scalar(0)));
	//			dbTrans[i].push_back(Mat(dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0)));
	//			dbTrans[i][j]=dbSource[i].rgb[j]-m;
	//			Y[i][j] = featureCalcLDA<double>(dbTransRGB[i][j]);
	//			//matPrint<double>(Y[i][j],createFVfilename("lda-rgb-Y",i,j));
	//		}
	//	}
	//}
	//else if(imgToLoad == DEPTH)
	//{
	//	for(int i = 0; i < nbrclass; i++)
	//	{
	//		Y[i].reserve(nbrimageperclass);
	//		for(int j = 0 ; j < nbrimageperclass ; j++)
	//		{
	//			Y[i].push_back(Mat(dbSource.rows(), numFisher, CV_64FC1, Scalar(0)));
	//			dbTransDepth[i].push_back(Mat(dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0)));
	//			for(int k = 0 ; k < dbSource.rows() ;k++) 
	//				for(int l = 0; l < dbSource.cols() ;l++)
	//				{
	//					dbTransDepth[i][j].at<double>(k,l) = dbSource[i].depth[j].at<uchar>(k,l)-m.at<double>(k,l);
	//				}
	//			Y[i][j] = featureCalcLDA<double>(dbTransDepth[i][j]);
	//			//matPrint<double>(Y[i][j],createFVfilename("lda-depth-Y",i,j));
	//		}
	//	}
	//}
	//else if(imgToLoad == MIX)
	//{
	//	Yp.resize(nbrclass);
	//	for(int i = 0; i < nbrclass; i++)
	//	{
	//		Y[i].reserve(nbrimageperclass);
	//		dbTransRGB[i].reserve(nbrimageperclass);
	//		dbTransDepth[i].reserve(nbrimageperclass);
	//		Yp[i].reserve(nbrimageperclass);
	//		for(int j = 0 ; j < nbrimageperclass ; j++)
	//		{
	//			Y[i].push_back(Mat(numFisher,dbSource.rows(), CV_64FC1, Scalar(0)));
	//			dbTransRGB[i].push_back(Mat(dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0)));
	//			dbTransRGB[i][j]=dbSource[i].rgb[j]-m;
	//			//matPrint<double>(m,createFVfilename("lda-mix-m",i,j));
	//			Y[i][j] = featureCalcLDA<double>(dbTransRGB[i][j]);	
	//			Yp[i].push_back(Mat(numFisher, dbSource.rows(), CV_64FC1, Scalar(0)));
	//			dbTransDepth[i].push_back(Mat(dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0)));
	//			//matPrint<uchar>(dbSource[i].depth[j],createFVfilename("lda-mix-dbSource.Depth",i,j));
	//			for(int k = 0 ; k < dbSource.rows() ;k++) 
	//				for(int l = 0; l < dbSource.cols() ;l++)
	//				{
	//					dbTransDepth[i][j].at<double>(k,l) = dbSource[i].depth[j].at<uchar>(k,l)-m.at<double>(k,l);
	//				}
	//			//matPrint<double>(dbTransDepth[i][j],createFVfilename("lda-mix-dbTransDepth",i,j));
	//			Yp[i][j] = featureCalcLDA<double>(dbTransDepth[i][j]);
	//			//matPrint<double>(Yp[i][j],createFVfilename("lda-mix-Ydepth",i,j));
	//		}
	//	}
	//}
	
	//ends = clock(); cout << " done in "<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s."<<endl;
	//END Execute
}
template<typename _Tp> void AlgoPCALDA::ldaEigen(cv::Mat& EigenValuesLDA, cv::Mat& EigenVectorsLDA)
{
	// Computation of the Mean Matrix and the Sub-mean Matrix
	
	vector<Mat> mi; 
	mi.reserve(nbrclass);
	for(int i=0;i<nbrclass;i++) mi.push_back(Mat(dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0)));
	ldaMeanSubMean<_Tp>(mi); //Compute the mean between all images faces[*].images[*]

	// Computation of Scatter Between Class Matrix
	Mat Sb     (dbSource.rows(),dbSource.rows(), CV_64FC1, Scalar(0));
	Mat SbTemp     (nbEigenVectors, dbSource.rows(), CV_64FC1, Scalar(0));
	Mat SbL     (nbEigenVectors, nbEigenVectors, CV_64FC1, Scalar(0));
	
	ldaScatterBetweenMat<double>(mi,m,Sb);
	
	SbTemp = EigenVectorsReduced.t()*Sb;
	SbL = SbTemp*EigenVectorsReduced;
	
	// Computation of Scatter Within Class Matrix
	Mat Sw    (dbSource.rows(), dbSource.rows(), CV_64FC1, Scalar(0));
	Mat SwTemp    (nbEigenVectors, dbSource.rows(), CV_64FC1, Scalar(0));
	Mat SwL    (nbEigenVectors, nbEigenVectors, CV_64FC1, Scalar(0));
	
	ldaScatterWithinMat<double>(mi,Sw);

	SwTemp = EigenVectorsReduced.t()*Sw;
	SwL = SwTemp*EigenVectorsReduced;
	//matPrint<double>(SwL,createFVfilename("lda-rgb-SwL",0,0));
	
	// Calcul of W
	Mat SwInv(nbEigenVectors,nbEigenVectors, CV_64FC1, Scalar(0));
	Mat W(nbEigenVectors, nbEigenVectors, CV_64FC1, Scalar(0));
	SwInv = SwL.inv();
	W = matMul<double,double>(SwInv,SbL);

	// Computation of eigenval/vec of W
	eigen(W,EigenValuesLDA,EigenVectorsLDA); // already in descending order.
}


template<typename _Tp> inline void AlgoPCALDA::ldaMeanSubMean(std::vector<cv::Mat>& mi)
{
	m = Mat(dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0));

	/*if(imgToLoad == RGB)
	{*/
		for(int k = 0 ; k < nbrclass ; k++)
		{
			for(int l = 0 ; l < nbrimageperclass ; l++)
			{
				for(int i = 0 ; i < m.rows ;i++) 
					for(int j = 0; j < m.cols ;j++)
					{
						m.at<double>(i,j)+=dbSource[k][l].at<_Tp>(i,j);
						mi[k].at<double>(i,j)+=dbSource[k][l].at<_Tp>(i,j);
					}
				// Computation of the Sub Mean Matrix
			}
			mi[k]/=(double)(nbrimageperclass);
		}
	//}
	//else if(imgToLoad == DEPTH)
	//{
	//	for(int k = 0 ; k < nbrclass ; k++)
	//	{
	//		for(int l = 0 ; l < nbrimageperclass ; l++)
	//		{
	//			for(int i = 0 ; i < m.rows ;i++) 
	//				for(int j = 0; j < m.cols ;j++)
	//				{
	//					m.at<double>(i,j)+=dbSource[k].depth[l].at<_Tp>(i,j);
	//					mi[k].at<double>(i,j)+=dbSource[k].depth[l].at<_Tp>(i,j);
	//				}
	//			// Computation of the Sub Mean Matrix
	//		}
	//		mi[k]/=(double)(nbrimageperclass);
	//	}
	//}
	//else if(imgToLoad == MIX)
	//{
	//	for(int k = 0 ; k < nbrclass ; k++)
	//	{
	//		for(int l = 0 ; l < nbrimageperclass ; l++)
	//		{
	//			for(int i = 0 ; i < m.rows ;i++) 
	//				for(int j = 0; j < m.cols ;j++)
	//				{
	//					m.at<double>(i,j)+=dbSource[k].mix[l].at<_Tp>(i,j);
	//					mi[k].at<double>(i,j)+=dbSource[k].mix[l].at<_Tp>(i,j);
	//				}
	//			// Computation of the Sub Mean Matrix
	//		}
	//		mi[k]/=(double)(nbrimageperclass);
	//	}
	//}
	m/=(double)(nbrclass*nbrimageperclass);
}


template<typename _Tp> void AlgoPCALDA::ldaScatterBetweenMat(const std::vector<cv::Mat>& mi, cv::Mat& m,cv::Mat& Sb){ // For one face
	
	Mat temp  (mi[0].rows, mi[0].cols, CV_64FC1, Scalar(0));
	Mat G     (mi[0].rows, mi[0].rows, CV_64FC1, Scalar(0));
	//Mat Sb    (mi[0].rows, mi[0].rows, CV_64FC1, Scalar(0));
	// Computation of Scatter Between Class Matrix
	for(int o = 0; o < nbrclass; o++)
	{
		for(int i = 0 ; i < m.rows ;i++) for(int j = 0; j < m.cols ;j++)
				temp.at<double>(i,j) = mi[o].at<double>(i,j) - m.at<double>(i,j);
			G+=temp*temp.t();
		G *= (_Tp) nbrimageperclass;
		Sb=Sb+G;
		//G = Mat::zeros(G.rows,G.cols,CV_64FC1);
	}
}


template<typename _Tp> void AlgoPCALDA::ldaScatterWithinMat(const std::vector<cv::Mat>& mi,cv::Mat& Sw){ // For one face
	
	Mat temp  (mi[0].rows, mi[0].cols, CV_64FC1, Scalar(0));
	Mat G     (mi[0].rows, mi[0].rows, CV_64FC1, Scalar(0));
	// Computation of Scatter Within Class Matrix
	/*if(imgToLoad == RGB)
	{
		for(int o=0; o < nbrclass; o++)
		{
			for(int k = 0 ; k < nbrimageperclass ; k ++)
			{
				for(int i = 0 ; i < mi[0].rows ;i++) for(int j = 0; j < mi[0].cols ;j++)
						 temp.at<double>(i,j) = dbSource[o].rgb[k].at<double>(i,j) - mi[o].at<double>(i,j);
				G+=temp*temp.t();
			}
			Sw=Sw+G;
			G = Mat::zeros(G.rows,G.cols,CV_64FC1);
		}
	}
	else if(imgToLoad == DEPTH)
	{
		for(int o=0; o < nbrclass; o++)
		{
			for(int k = 0 ; k < nbrimageperclass ; k ++)
			{
				for(int i = 0 ; i < mi[0].rows ;i++) for(int j = 0; j < mi[0].cols ;j++)
						 temp.at<double>(i,j) = dbSource[o].depth[k].at<uchar>(i,j) - mi[o].at<double>(i,j);
				G+=temp*temp.t();
			}
			Sw=Sw+G;
			G = Mat::zeros(G.rows,G.cols,CV_64FC1);
		}
	}
	else if(imgToLoad == MIX)
	{*/
		for(int o=0; o < nbrclass; o++)
		{
			for(int k = 0 ; k < nbrimageperclass ; k ++)
			{
				for(int i = 0 ; i < mi[0].rows ;i++) for(int j = 0; j < mi[0].cols ;j++)
						 temp.at<double>(i,j) = dbSource[o][k].at<double>(i,j) - mi[o].at<double>(i,j);
				G+=temp*temp.t();
			}
			Sw=Sw+G;
			G = Mat::zeros(G.rows,G.cols,CV_64FC1);
		}
	/*}*/
}



template<typename _Tp> void AlgoPCALDA::pcaEigen(cv::Mat& EigenValues, cv::Mat& EigenVectors)
{
	// Computation of the Mean Matrix
	Mat mean  (dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0));
	pcaMean<_Tp>(mean); //Compute the mean between all images faces[*].images[*]

	// 2DPCA - Method
	// Computation of Covariance Matrix : G
	Mat G     (dbSource.rows(), dbSource.rows(), CV_64FC1, Scalar(0));
	pcaCovariance<_Tp>(G,mean);
	
	// Computation of eigenval/vec of G
	eigen(G,EigenValues,EigenVectors); // already in descending order.
}

template<typename _Tp> void AlgoPCALDA::pcaMean(cv::Mat& mean)
{
	/*if(imgToLoad == RGB)
	{
		for(int k = 0 ; k < nbFaces ; k++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					mean.at<double>(i,j)+=dbSource[k].rgb[l].at<_Tp>(i,j);
	}
	else if(imgToLoad == DEPTH)
	{
		for(int k = 0 ; k < nbFaces ; k++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					mean.at<double>(i,j)+=dbSource[k].depth[l].at<_Tp>(i,j);
	}
	else if(imgToLoad == MIX)
	{*/
		for(int k = 0 ; k < nbFaces ; k++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
				{
					mean.at<double>(i,j)+=dbSource[k][l].at<_Tp>(i,j);
				}
	/*}*/
	mean/=(double)(nbFaces*nbImgPerFace);
}

template<typename _Tp> void AlgoPCALDA::pcaCovariance(cv::Mat& G, const cv::Mat& mean)
{
	Mat temp  (dbSource.rows(), dbSource.cols(), CV_64FC1, Scalar(0));
	/*if(imgToLoad == RGB)
	{
		for(int k = 0 ; k < nbFaces ; k ++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
			{
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					 temp.at<double>(i,j) = dbSource[k].rgb[l].at<_Tp>(i,j) - mean.at<double>(i,j);

				G+=temp*temp.t();
			}
	}
	else if(imgToLoad == DEPTH)
	{
		for(int k = 0 ; k < nbFaces ; k ++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
			{
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
					 temp.at<double>(i,j) = dbSource[k].depth[l].at<_Tp>(i,j) - mean.at<double>(i,j);

				G+=temp*temp.t();
			}
	}
	else if(imgToLoad == MIX)
	{*/
		for(int k = 0 ; k < nbFaces ; k ++)
			for(int l = 0 ; l < nbImgPerFace ; l++)
			{
				for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
				{
					 temp.at<double>(i,j) = dbSource[k][l].at<_Tp>(i,j)-mean.at<double>(i,j);
				}
				G+=temp*temp.t();
			}
	/*}*/

	G /= (double) (nbFaces*nbImgPerFace);
}

double AlgoPCALDA::computePourcentage(cv::Mat& EigenValues)
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
//			tempVal=norm(Y[i][j],featureVector);
//			addNorm(tempVal);
//			if(tempVal<min)
//			{
//				min=tempVal;
//				tab = i; // minFaceIndex
//			}
//		}
//	}
//
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
//			sum += norm(Y[i][j],featureVector);
//		}
//		sum /= nbImgPerFace;
//		addNorm(sum);
//		if(sum<min)
//		{
//			min=sum;
//			tab = i; // minFaceIndex
//		}
//	}
//
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
//			tempVal=norm(Y[i][j],featureVectorRGB)+norm(Yp[i][j],featureVectorDepth);
//			addNorm(tempVal);
//			if(tempVal<min)
//			{
//				min=tempVal;
//				tab = i; // minFaceIndex
//			}
//		}
//	}
//
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
//		addNorm(tempVal);
//		if(tempVal<min)
//		{
//			min=tempVal;
//			tab = i; // minFaceIndex
//		}
//	}
//
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
//			diff.at<double>(i,j) =(double) featureMatrix.at<double>(i,j) - meanMatrix.at<double>(i,j);
//		}
//
//	//invert(covMatrix,invCov,DECOMP_LU);
//	temp = covMatrix.inv()*diff;
//	temp = diff.t()*temp;
//
//	r = norm(temp,NORM_L2);
//	//cout << " r = " << r << endl;
//	return r;
//}
//
//template<typename _Tp> int AlgoPCA::discriminantFunction(Mat& featureVector, double threshold){
//	Mat cov;
//	Mat mean;
//
//	int nature = -1;
//	double temp = 0;
//	double min = numeric_limits<double>::max();
//	
//	for(int i = 0; i < nbFaces ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		mean = meanMat<_Tp>(Y[i]);
//		cov = featureCovMat<_Tp>(Y[i],mean);
//
//		temp = mahalanobisDistance(cov,mean,featureVector);
//		addNorm(temp);
//		if(temp < min){
//			min = temp;
//			nature = i;
//		}
//	}
//	//cout <<"min:"<<min;
//
//	if(min < threshold) return nature;
//	else return -1; // Return the closest class 
//}
//template<typename _Tp> int AlgoPCA::discriminantFunctionKNN(Mat& featureVector, int k){
//	Mat cov;
//	Mat mean;
//
//	KNearest knearest(k);
//
//	for(int i = 0; i < nbFaces ; i ++) // For each face
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
//template<typename _Tp> int AlgoPCA::discriminantFunction(Mat& featureVectorRGB, Mat& featureVectorDepth, double threshold){
//	Mat covRGB, covDepth;
//	Mat meanRGB, meanDepth;
//
//	int nature = -1;
//	double tempRGB = 0, tempDepth = 0;
//	double min = numeric_limits<double>::max();
//	for(int i = 0; i < nbFaces ; i ++) // For each face
//	{
//		// Computation of normal density parameters : u and E
//		meanRGB = meanMat<_Tp>(Y[i]);
//		covRGB = featureCovMat<_Tp>(Y[i],meanRGB);
//		meanDepth = meanMat<_Tp>(Yp[i]);
//		covDepth = featureCovMat<_Tp>(Yp[i],meanDepth);
//
//		tempRGB = mahalanobisDistance(covRGB,meanRGB,featureVectorRGB);
//		tempDepth = mahalanobisDistance(covDepth,meanDepth,featureVectorDepth);
//		addNorm(tempRGB+tempDepth);
//		if(tempRGB+tempDepth < min){
//			min = tempRGB+tempDepth;
//			nature = i;
//		}
//	}
//
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