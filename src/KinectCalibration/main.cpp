#include "main.h"

int main(void)
{
	Calibration calib("H3.xml");
	cv::Mat rgb = cv::imread("imageRgb.png");
	cv::Mat depth = cv::imread("imageDepth.png");
	/*for(bool sw=0; cv::waitKey(500) < 0; sw=sw?0:1)
	{
		if (sw) cv::imshow("out",calib.warp(depth));
		else cv::imshow("out",rgb);
	}*/
	
	return EXIT_SUCCESS;
}