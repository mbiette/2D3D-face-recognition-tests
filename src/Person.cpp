#include "Person.h"

Person::Person(std::string name, std::vector<std::string>& pathRgb, std::vector<std::string>& pathDepth, Mixer _mix): name(name), typeMix(_mix)
{
	//Load all the images
	loader(rgb,pathRgb,true,true);
	loader(depth,pathDepth,false,false);

	//Set the size
	if(rgb.size()!=0) sizeVal=rgb.size();
	else if(depth.size()!=0) sizeVal=depth.size();

	//Load the mix data if required.
	if(rgb.size()!=0 && depth.size()!=0) doMix();
}

cv::Mat Person::cropper(cv::Mat & in)
{
	return cv::Mat( in( cv::Rect(cv::Point(88,173),cv::Point(429,527)) ) );
	//return cv::Mat( in( cv::Rect(cv::Point(38,147),cv::Point(478,553)) ) );

}

cv::Mat Person::toDouble(cv::Mat& in)
{
	cv::Mat out(in.rows, in.cols, CV_64FC1);
	for(int k = 0 ; k < in.rows; k++)
		for(int l = 0 ; l < in.cols; l++)
			out.at<double>(k,l) = in.at<uchar>(k,l);
	return out;
}

void Person::loader(std::vector<cv::Mat>& dest, std::vector<std::string>& path,bool color, bool preprocessing)
{
	dest.reserve(path.size());
	if(preprocessing) for(int i=0;i<path.size();i++) dest.push_back(preProcessing(cropper(cv::imread(path[i],color))));
	else for(int i=0;i<path.size();i++) dest.push_back(toDouble(cropper(cv::imread(path[i],color))));
}

void Person::doMix()
{
	mix.reserve(sizeVal);
	for(int j = 0; j < sizeVal; j++)
	{
		if(typeMix == MIX_SUM)
		{
			mix.push_back(cv::Mat(rgb[0].rows,rgb[0].cols,CV_64FC1,cv::Scalar(0)));
			for(int k = 0 ; k < rgb[0].rows; k++)
			{
				for(int l = 0 ; l < depth[0].cols; l++)
				{
					mix[j].at<double>(k,l) = rgb[j].at<double>(k,l)+depth[j].at<double>(k,l);
				}
			}
		}
		else if(typeMix == MIX_CONCAT)
		{
			mix.push_back(cv::Mat(rgb[0].rows,rgb[0].cols*2,CV_64FC1,cv::Scalar(0)));
			for(int k = 0 ; k < rgb[0].rows; k++)
			{
				for(int l = 0 ; l < depth[0].cols; l++)
				{
					mix[j].at<double>(k,l) = rgb[j].at<double>(k,l);
					mix[j].at<double>(k,l+depth[0].cols) = depth[j].at<double>(k,l);
				}
			}
		}
	}
}

int Person::size()
{
	return sizeVal;
}

cv::Mat Person::preProcessing(cv::Mat& A)
{
	cv::Mat proC(A.rows,A.cols,CV_64FC1,cv::Scalar(0));
	cv::Mat temp(A.rows,A.cols,CV_64FC3,cv::Scalar(0));
	// Passage en Chromatics
	for(int i = 0 ; i < A.rows ; i ++)
		for(int j = 0 ; j < A.cols ; j++)
		{
			//double denum = A.at<cv::Vec3b>(i,j)[0] + A.at<cv::Vec3b>(i,j)[1] + A.at<cv::Vec3b>(i,j)[2];
			//if(denum != 0)
			//{
			//	temp.at<cv::Vec3d>(i,j)[0] = (double) A.at<cv::Vec3b>(i,j)[0] / denum; // B
			//	temp.at<cv::Vec3d>(i,j)[1] = (double) A.at<cv::Vec3b>(i,j)[1] / denum; // G
			//	temp.at<cv::Vec3d>(i,j)[2] = (double) A.at<cv::Vec3b>(i,j)[2] / denum; // R
			//}
			//else 
			//{
				temp.at<cv::Vec3d>(i,j)[0] = (double) A.at<cv::Vec3b>(i,j)[0]; // B
				temp.at<cv::Vec3d>(i,j)[1] = (double) A.at<cv::Vec3b>(i,j)[1]; // G
				temp.at<cv::Vec3d>(i,j)[2] = (double) A.at<cv::Vec3b>(i,j)[2]; // R
			//}
		}

	// Grayscale
	for(int i = 0 ; i < A.rows ; i ++)
		for(int j = 0; j < A.cols ; j++)
			proC.at<double>(i,j) = (0.114 * temp.at<cv::Vec3d>(i,j)[0] + 0.587 * temp.at<cv::Vec3d>(i,j)[1] + 0.299 * temp.at<cv::Vec3d>(i,j)[2]);

	return proC;
}

int Person::rows()
{
	int rows;
	if(cible!=0 && cible->size()!=0) rows = (*cible)[0].rows;
	else if(rgb.size()!=0) rows = rgb[0].rows;
	else if(depth.size()!=0) rows = depth[0].rows;
	return rows;
}

int Person::cols()
{
	int cols;
	if(cible!=0 && cible->size()!=0) cols = (*cible)[0].cols;
	else if(rgb.size()!=0) cols = rgb[0].cols;
	else if(depth.size()!=0) cols = depth[0].cols;
	return cols;
}

void  Person::setCibleRGB()
{
	cible = &rgb;
}

void  Person::setCibleDepth()
{
	cible = &depth;
}
void  Person::setCibleMix()
{
	cible = &mix;
}

cv::Mat& Person::operator [] (int id)
{
	//cout << "img["<<id<<"]"<<endl;
	return (*cible)[id];
}