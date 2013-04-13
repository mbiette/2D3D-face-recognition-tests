#include "databasecreator.h"

#define imgPerPers 21

using namespace cv;
using namespace std;

DatabaseCreator::DatabaseCreator()
{
	session = false;
	stop = false;
	path = "database";
	path2 = "databaseBrute";

	dir1=QDir::currentPath();
	dir2=QDir::currentPath();
	if(!dir1.exists(path)) dir1.mkdir(path);
	dir1.setPath(path);
	if(!dir2.exists(path2)) dir2.mkdir(path2);
	dir2.setPath(path2);

	// IHM
	cout << " How many persons would you like to add in the database ? " << endl;
	cin >> nbPers;
}

DatabaseCreator::~DatabaseCreator()
{
}

void DatabaseCreator::core(){
	int cpt=0,compteurPhoto=0;
	const IplImage*caca;
	// Calibration
	Calibration cab("../KinectCalibration/H3.xml");
	FaceDetectionVJ faceDet("../FaceDetection-VJ.haarcascade_frontalface_alt.xml");
	// Loop
	while(!stop){ // Tant que t'as pas fini de prendre toutes les pers en photo
		// Opening session for 1 person
			//key=getch();
		cout << " New session " << endl;
				session = true;
				saveName();
				dir1.mkdir(name);
				dir1.cd(name);
				dir1.mkdir("rgb");
				dir1.mkdir("depth");
				dir2.mkdir(name);
				dir2.cd(name);
				dir2.mkdir("rgb");
				dir2.mkdir("depth");
				system("pause");

		// Annonce
		//cout << "Click on any button to get photographied when you're ready" << endl;

		// Database construction for 1 person
		while(session)
		{
			caca = kin.getIplImageDepthNormalized();
			tempBGR = kin.getIplImageRGB();
			tempDepth = kin.getIplImageDepth();
			if(cvWaitKey(100)!= -1){ // Tu save la photo
				faceDet.detectFromImage(tempBGR);
				Depth = tempDepth;
				Depth = cab.warp(Depth);

				vector<Mat> A = faceDet.cropAndScaleImageIntoMat(tempBGR,100,100,CV_8UC3);
				vector<Mat> B = faceDet.cropAndScaleImageIntoMat(Depth,100,100,CV_16UC1);
				
				if(A.size() == 1){
				saveDepth(compteurPhoto,B);
				saveRGB(compteurPhoto,A);
				cout << " Photo number " << compteurPhoto << " saved" << endl;
				compteurPhoto++;
				}
			}
			cvShowImage("Cam",tempBGR);
			cvShowImage("Caca",caca);
			if(compteurPhoto==imgPerPers){
				compteurPhoto = 0;
				session = false;
				cout << " photoshoot fini " << endl;
				cvDestroyWindow("Cam");
				dir1.cdUp();
				dir2.cdUp();
			}
		}
		
		cpt++; // Une pers de fini en +
		if(cpt == nbPers) stop = true; // Quand on a fini le shoot session
	}
	cout << " finished "<< endl;
}

void DatabaseCreator::saveName(){
	string na;
	cout << " Enter your name : " << endl;
	cin >> na;
	name = na.c_str();
}

void ProcessRGB(Mat &A,Mat &B){
	for(int i = 0; i < A.rows;i++)
		for(int j = 0;j < A.cols;j++)
		{
			if(B.at<uchar>(i,j) == 255){
				A.at<Vec3b>(i,j)[0] = 255;
				A.at<Vec3b>(i,j)[1] = 255;
				A.at<Vec3b>(i,j)[2] = 255;
			}
		}
}

void DatabaseCreator::saveRGB(int a,std::vector<cv::Mat> &A){
	char temp[imgPerPers];
	// RGB after cropping
	string str = qPrintable(dir1.path());str.append("/rgb/img");str.append(_itoa(a,temp,10));str.append(".png");
	cout << str << endl;
	ProcessRGB(A[0],BThres);
	imwrite(str,A[0]);
	// RGB from src
	str = qPrintable(dir2.path());str.append("/rgb/img");str.append(_itoa(a,temp,10));str.append(".png");
	cvSaveImage(str.c_str(),tempBGR);
}

unsigned short int nearest(const Mat &A){
	unsigned short int a = numeric_limits<unsigned short int>::max();
	for(int i = 0; i < A.rows;i++) for(int j = 0; j < A.cols;j++)
		if(a > A.at<unsigned short int>(i,j)) a = A.at<unsigned short int>(i,j);
	return a;
}

Mat translation(Mat &A,unsigned short int a){
	Mat B(A.rows,A.cols,CV_8UC1,Scalar(0));
	int cpt=0;
	for(int i = 0; i < A.rows;i++) for(int j = 0; j < A.cols;j++)
	{
		if ( A.at<unsigned short int>(i,j) - a > 255){
			cpt++;B.at<uchar>(i,j) = 255;
		}
		else B.at<uchar>(i,j) = uchar(A.at<unsigned short int>(i,j) - a);
	}
	cout << cpt<< " de + de 255" << endl;
	return B;
}

Mat Otsu(Mat imgg){
	// Ressources
	t_pix* hist = (t_pix*)malloc(256*sizeof(t_pix));
	float wk=0,muk=0,muT=0;
	float lambda=0,lambdaold=0;
	float w0=0,w1=0,mu0=0,mu1=0,sigma02=0,sigma12=0,sigmaT2=0,sigmaW2=0,sigmaB2=0;
	int threshold,cpt=0;

	// Init
	for(int i = 0 ; i < 256;i++) hist[i].nb = 0;

	Size imgsize = imgg.size();
	int height = imgsize.height;
	int width = imgsize.width;

	int nbTotPix = height*width;

	// Histogram
	for(int k = 0; k < height ; k++)
		for(int j = 0; j < width ; j++)
		{
			cpt++;
			hist[(int)imgg.at<uchar>(k,j)].nb++;
		}

	for(int i = 0 ; i < 256 ; i++) hist[i].prob = (float) hist[i].nb / cpt;

	int uf = 0;
	for(int i=0;i<256;i++) uf+=hist[i].nb;
	// For every different level when divising

	// Calculation of the total Mean of the image ( the expected value ) : muT and sigmaT^2
	for(int i = 0; i < 256; i++) muT+=i*hist[i].prob;
	for(int i = 0; i < 256; i++) sigmaT2+=(i-muT)*(i-muT)*hist[i].prob;

	// The k-levels (the threshold we evaluate).
	for(int k = 0; k < 256; k++)
	{
		// Computing of w0 and w1 : the class occurence
		for(int i = 0 ; i <= k; i++)
		{
			muk += i*hist[i].prob;
			wk += hist[i].prob;
		}
		w0 = wk;
		w1 = 1 - wk;

		// Computing of mu0, mu1 : class mean levels
		mu0 = muk / wk;
		mu1 = (muT - muk) / (1 - wk);

		// Computing sigma0, sigma1
		for(int i = 0; i <=k;i++)
			sigma02+=((i-mu0)*(i-mu0)*hist[i].prob)/w0;
		for(int i = k+1 ; i < 256 ; i++)
			sigma12+=((i-mu1)*(i-mu1)*hist[i].prob)/w1;

		// Criterion measure
		sigmaW2 = w0*sigma02+w1*sigma12;
		sigmaB2 = w0*w1*(mu1 - mu0)*(mu1 - mu0);
		lambda = sigmaB2/sigmaW2;

		if(lambda > lambdaold)
		{
			threshold = k;
			lambdaold = lambda;
		}
		// Réinitialisation
		sigma12=0;
		sigma02=0;
		wk=0;
		muk=0;
	}

	for(int i = 0; i < height ; i++)
		for(int j = 0; j < width ; j++)
		{
			if(imgg.at<uchar>(i,j) > threshold)
			imgg.at<uchar>(i,j) = 255;
		}

	cout << threshold << " as a threslhold " << endl;

	//
	Mat exa;
	medianBlur(imgg,exa,3);
	free(hist);
	return exa;
}

void DatabaseCreator::saveDepth(int a,std::vector<cv::Mat> &B){
	char temp[imgPerPers];
	// Depth after cropping
	string str = qPrintable(dir1.path());str.append("/depth/img");str.append(_itoa(a,temp,10));str.append(".png");
	cout << str << endl;

	// Normalisation
	// Nearest
	int b = nearest(B[0]);
	Mat BNorm = translation(B[0],b);
	BThres = Otsu(BNorm);
	// Hole filling
	HoleFilling A(BThres,3,230);
	A.removeHole();
	imwrite(str,BThres);
	// Depth from src
	str = qPrintable(dir2.path());str.append("/depth/img");str.append(_itoa(a,temp,10));str.append(".png");
	cvSaveImage(str.c_str(),tempDepth);
}