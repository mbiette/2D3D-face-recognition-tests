#include "holefilling.h"

using namespace cv;
using namespace std;
ofstream fichier("correlation.txt");

HoleFilling :: HoleFilling (cv::Mat &img,int deg,int thresh):ImgMat(img){
	degre = deg;
	threshold = thresh;
}

Mat calculCoeff( vector<CvPoint2D64f> &data,int degre)
{
	int taille = data.size();
	/** Ressources **/
	Mat X (taille, degre+1, CV_64FC1, Scalar(1));
	Mat Y (taille, 1, CV_64FC1, Scalar(0));
    Mat Xt (degre+1, taille, CV_64FC1, Scalar(0));
	Mat XtY (degre+1, 1,CV_64FC1, Scalar(0));
	Mat XtX (degre+1, degre+1,CV_64FC1, Scalar(0));
	Mat invXtX (degre+1, degre+1,CV_64FC1, Scalar(0));
	Mat B (degre+1, 1,CV_64FC1, Scalar(0));
	
	// Initialisation of X
	for(int i=0; i<X.rows; i++)
		for(int j=1; j<X.cols; j++)
			X.at<double>(i,j) = X.at<double>(i,j-1) * data[i].x;

	//Initialisation of Y
	for(int i=0; i<Y.rows; i++)
		Y.at<double>(i,0)=data[i].y;

	/** Calcul of the transpose of x **/
    Xt = X.t();

    /** Calcul of xt*y **/
    XtY = Xt*Y;

    /** Calcul of xt*x **/
    XtX = Xt*X;

    /** Calcul of the inverse of x*xt **/
	invert(XtX,invXtX,DECOMP_SVD);

    /** Calcul of the matrix B (coefficents) **/
    B = invXtX*XtY;
	return B;
}

bool detectHole(Mat &D,int j,int threshold)
{
	for(int i=0;i<D.rows;i++)
	{
		if((double)D.at<uchar>(i,j)>=threshold) return true;
	}
	return false;
}

bool detectHole(Mat &D,int i,int j,int threshold)
{
	if((double)D.at<uchar>(i,j)>=threshold) return true;
	else return false;
}

void fillBackground(Mat &data,int val)
{
	for(int i=0;i<data.rows;i++)
	{
		if((double)data.at<uchar>(i,0)>=220)
		{
			int k=0;
			do
			{
				data.at<uchar>(i,k)=(uchar)val; 
				if (k<data.cols-1) k++;
			}while( (double)data.at<uchar>(i,k)>=220 && (k<data.cols-1));
		}
		if((double)data.at<uchar>(i,data.cols-1)>=220)
		{
			int k=data.cols-1;
			do
			{
				data.at<uchar>(i,k)=(uchar)val;
				if (k>0) k--;
			}while( (double)data.at<uchar>(i,k)>=220 && (k>0));
		}
	}
}

void HoleFilling :: removeHole()
{
	fillBackground(ImgMat,220);
	/*cvNamedWindow("filling background", CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("filling background", 220, 50);
	cv::imshow("filling background",ImgMat);
	cvWaitKey(40);*/
	vector<double> R;
	for(int j=0;j<ImgMat.cols;j++)
	{
		// Detect hole and remove them from the data
		if(detectHole(ImgMat,j,threshold))
		{
			//  Creation of the vector data whithout the hole
			vector<CvPoint2D64f> P;
			int loop=0;
			for(int i=0;i<ImgMat.rows;i++)
			{
				if(!detectHole(ImgMat,i,j,threshold)) 
				{
					CvPoint2D64f temp;
					temp.y=(double)ImgMat.at<uchar>(i,j);
					temp.x=loop;
					loop++;
					P.push_back(temp);
				}
			}
			// Calcul of the coeff
			Mat coef = calculCoeff(P,degre);
			//Filling the hole
			for(int i=0;i<ImgMat.rows;i++)
			{
				if(detectHole(ImgMat,i,j,threshold))
				{
					Mat X (1,degre+1, CV_64FC1, Scalar(1));
					for(int l=1; l<X.cols; l++)
						X.at<double>(0,l) = X.at<double>(0,l-1)*i;
					for(int m=0;m<degre+1;m++)
						ImgMat.at<uchar>(i,j) += uchar(X.at<double>(0,m)*coef.at<double>(m,0));					
				}
			}
			
			// Calculating the corellation coefficient
			double tempR =0;
			double SCR =0;
			double SCT = 0;
			vector<CvPoint2D64f> Pprime;
			for(int i=0;i<P.size();i++)
			{
				CvPoint2D64f temp;
				temp.x=P[i].x;
				temp.y=0;
				Mat X (1,degre+1, CV_64FC1, Scalar(1));
				for(int l=1; l<X.cols; l++)
					X.at<double>(0,l) = X.at<double>(0,l-1)*temp.x;
				for(int m=0;m<degre+1;m++)
					temp.y += X.at<double>(0,m)*coef.at<double>(m,0);
				Pprime.push_back(temp);
			}
			double moy=0;
			for(int i=0;i<P.size();i++)
				moy += P[i].y;
			moy = moy/P.size();
			for(int i=0;i<P.size();i++)
			{
				SCR += (P[i].y - Pprime[i].y)*(P[i].y - Pprime[i].y);
				SCT += (P[i].y - moy)*(P[i].y - moy);
			}
			/*cout << SCR << endl;
			cout << SCT << endl;
			cout << SCR/SCT << endl << endl;*/
			//tempR = 1 - (SCR/SCT);
			tempR = (SCT -SCR)/SCT;
			R.push_back(tempR);
			P.clear();
			Pprime.clear();
			loop=0;
		}
	}
	/*cvNamedWindow("1st pre-processing", CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("1st pre-processing", 390, 50);
	cv::imshow("1st pre-processing",ImgMat);
	cvWaitKey(40);*/

	fillBackground(ImgMat,255);

	/*cvNamedWindow("2d pre-processing", CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("2d pre-processing", 560, 50);
	cv::imshow("2d pre-processing",ImgMat);
	cvWaitKey(40);*/
	//double moyR=0;
	//double min,max;
	//min = max = R[0];
	//for(int i=0;i<R.size();i++)
	//{
	//	if(R[i] < min)
	//		min = R[i];
	//	else if (R[i] >max)
	//		 max = R[i];
	//	moyR += R[i];
	//	cout << R[i] << endl;
	//}
	//moyR = moyR/R.size();

	//
 //   if(fichier)
	//	fichier << moyR << " " << max <<" " <<min << endl;
 //   else
 //       cerr << "Impossible d'ouvrir le fichier !" << endl;
	/*cout << endl;
	cout << moyR << endl;
	cout << max << endl;
	cout << min << endl;
	cout << endl;*/
}

Mat HoleFilling :: getData()
{
	return ImgMat;
}