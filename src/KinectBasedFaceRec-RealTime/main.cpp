#include "KinectBasedFaceRecRT.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	KinectBasedFaceRecRT w;
	w.show();
	return a.exec();
}
