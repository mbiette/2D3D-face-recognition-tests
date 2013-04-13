#include "KinectBasedFaceRecRT.h"

KinectBasedFaceRecRT::KinectBasedFaceRecRT(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	resize(640,480);
	QVBoxLayout * layout = new QVBoxLayout();
	labelImage = new QLabel("Image",this);
	labelName = new QLabel("Name",this);
	layout->addWidget(labelImage);
	layout->addWidget(labelName);
	labelImage->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
	labelName->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
	container = new QWidget(this);
	container->setLayout(layout);
	setCentralWidget(container);
}

KinectBasedFaceRecRT::~KinectBasedFaceRecRT()
{

}
