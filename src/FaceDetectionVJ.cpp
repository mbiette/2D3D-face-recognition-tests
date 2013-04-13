#include "FaceDetectionVJ.h"

using namespace cv;
using namespace std;

FaceDetectionVJ::FaceDetectionVJ( const char * cascadePath )
{
	storage = cvCreateMemStorage(0);
	faces = NULL;
	cascade = NULL;
	loadCascade ( cascadePath );
}

FaceDetectionVJ::~FaceDetectionVJ()
{
	cvReleaseMemStorage(&storage);
}
	
void FaceDetectionVJ::loadCascade( const char * cascadePath )
{
	if(cascadePath) cascade = (CvHaarClassifierCascade*)cvLoad( cascadePath, 0, 0, 0 );
}

// Function to detect and store them in the class
void FaceDetectionVJ::detectFromImage(const IplImage* image)
{
	// Check whether the cascade has loaded successfully. Else report and error and quit
    if( !cascade )
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        exit(-1);
    }

    // Clear the memory storage which was used before
    cvClearMemStorage( storage );
	faces = NULL;

    // Find whether the cascade is loaded, to find the faces. If yes, then:
    if( cascade )
    {
        // There can be more than one face in an image. So create a growable sequence of faces.
        // Detect the objects and store them in the sequence
        faces = cvHaarDetectObjects(image,
									cascade,
									storage,
									1.1,
									2,
									CV_HAAR_DO_CANNY_PRUNING,
									cvSize(40, 40)
									);
		//std::cout << "faces:" << (faces? faces->total : 0) <<endl;
	}
}

void FaceDetectionVJ::drawRectOnImage(IplImage* image)
{
	// Loop the number of faces found.
	for( int i=0 ; i < (faces? faces->total : 0) ; i++ )
	{
		// Create a new rectangle for drawing the face
		CvRect* r = (CvRect*)cvGetSeqElem( faces, i );

		// Translate the rectangle into points
		CvPoint ptUL;	ptUL.x = r->x;			ptUL.y = r->y;
		CvPoint ptLR;	ptLR.x = r->x+r->width;	ptLR.y = r->y+r->height;
		//cout<< r->x << " " << r->y << " " << r->width << " " << r->height <<endl;
		// We draw
		cvRectangle( image, ptUL, ptLR, CV_RGB(255,0,0), 2, 8, 0 );
	}
}

IplImage* FaceDetectionVJ::drawRectOnNewImage(const IplImage* image)
{
	IplImage* newImage = cvCloneImage(image);
	this->drawRectOnImage(newImage);

	return newImage;
}

std::vector<cv::Mat> FaceDetectionVJ::cropAndScaleImageIntoMat(cv::Mat image, int width, int height, const int _CvMatType)
{
	// Allocate a vector of Mat for storing the images
	vector<Mat> vect;
	
	if(!faces) return vect;
	vect.reserve(faces->total);
	
	// Loop the number of faces found.
	for( int i=0 ; i < (faces? faces->total : 0) ; i++ )
	{
		CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
		CvPoint ptUL;	ptUL.x = r->x;			ptUL.y = r->y;
		CvPoint ptLR;	ptLR.x = r->x+r->width;	ptLR.y = r->y+r->height;

		double faceRatio = double(r->width)/double(r->height);
		double destRatio = double(width)/double(height);

		if(faceRatio>destRatio) // face is fat
		{
			int diff = int( ( (r->width*(1/destRatio)) - r->height )/2 );
			ptUL.y -= diff;
			ptLR.y += diff;
		}
		else if(faceRatio<destRatio) // face is tall
		{
			int diff = int( ( (r->height*destRatio) - r->width )/2 );
			ptUL.x -= diff;
			ptLR.x += diff;
		}

		Mat mat(image);
		Mat matROI = mat(Rect(ptUL,ptLR));
		//cv::imshow("matROI",matROI);
		Mat scaledFace(height,width,_CvMatType,Scalar(0));
		
		//resize(const Mat& src, Mat& dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR)
		if(Rect(ptUL,ptLR).width*Rect(ptUL,ptLR).height >= width*height) // It's a reduction
		{
			resize(matROI,scaledFace,Size(width,height),0,0,INTER_AREA);
		}
		else //It an increase
		{
			resize(matROI,scaledFace,Size(width,height),0,0,INTER_LINEAR);
		}
		//cv::imshow("scaledFace", scaledFace);

		vect.push_back(scaledFace);

	}
	return vect;
}
