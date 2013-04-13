#ifndef KINECTBASEDFACERECRT_H
#define KINECTBASEDFACERECRT_H

#include <QtGui/QMainWindow>
#include <QVBoxLayout>
#include <QLabel>
#include "ui_KinectBasedFaceRecRT.h"

class KinectBasedFaceRecRT : public QMainWindow
{
	Q_OBJECT

public:
	KinectBasedFaceRecRT(QWidget *parent = 0, Qt::WFlags flags = 0);
	~KinectBasedFaceRecRT();

private:
	QLabel* labelImage,*labelName;
	QWidget* container;
};

#endif // KINECTBASEDFACERECRT_H
