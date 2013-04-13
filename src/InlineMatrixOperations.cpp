#include <windows.h>
#include <conio.h>
#include <fstream>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

template<typename _Tp> inline double vectNorm2(const cv::Mat A){
	double vectNorm=0;
	for(int k = 0; k < A.rows ; k++) vectNorm += A.at<_Tp>(k,0)*A.at<_Tp>(k,0);
	return sqrt(vectNorm);
}

template<typename _TpA, typename _TpB> inline cv::Mat matMul(const cv::Mat A, const cv::Mat B)
{
	cv::Mat Y(A.rows, B.cols, CV_64FC1, Scalar(0));
	for(int k = 0; k < A.rows ; k++)
		for(int j = 0; j < B.cols ; j++)
			for(int o = 0; o < A.cols;o++)
				Y.at<double>(k,j) += double(A.at<_TpA>(k,o)) * double(B.at<_TpB>(o,j));
	return Y;
}

template<typename _Tp> inline void matPrint(const cv::Mat A){
	
	for(int i=0;i<A.rows;i++){
		for(int j=0;j<A.cols;j++)
			if( !kbhit() )
				std::cout << double(A.at<_Tp>(i,j)) << " ";
		if( !kbhit() ) std::cout << endl;
	}
	if( !kbhit() ) std::cout << endl;
	getch();
}

template<typename _Tp> inline void matPrint(const cv::Mat A, string filename){
	ofstream file(filename);
	for(int i=0;i<A.rows;i++){
		for(int j=0;j<A.cols;j++)
			file << double(A.at<_Tp>(i,j)) << "\t";
		file << endl;
	}
	file << endl;
	file.close();
}

template<typename _Tp> cv::Mat meanMat(const std::vector<cv::Mat> A){
	Mat mean (A[0].rows,A[0].cols,CV_64FC1, Scalar(0));

	// Computation of the Mean Matrix
		for(int l = 0 ; l < (int) A.size(); l++)
			for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
				mean.at<double>(i,j)+=A[l].at<_Tp>(i,j);
	mean/=(double)(A.size());

	return mean;
}

template<typename _Tp> cv::Mat featureCovMat(const std::vector<cv::Mat> A, cv::Mat mean){ // For one face
	Mat cov  (A[0].rows,A[0].rows,CV_64FC1, Scalar(0));
	Mat temp (A[0].rows,A[0].cols,CV_64FC1, Scalar(0));

	// Computation of the Cov Matrix
	for(int l = 0 ; l < (int) A.size() ; l++)
		{
			for(int i = 0 ; i < mean.rows ;i++) for(int j = 0; j < mean.cols ;j++)
				 temp.at<double>(i,j) = A[l].at<_Tp>(i,j) - mean.at<double>(i,j);

		cov+=temp*temp.t();
		}
	cov /= (double) (A.size());

	return cov;
}