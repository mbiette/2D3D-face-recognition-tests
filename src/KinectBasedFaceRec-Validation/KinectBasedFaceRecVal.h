#ifndef KINECTBASEDFACERECVAL_H
#define KINECTBASEDFACERECVAL_H

#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>

#include <iostream>

#include <qstring.h>

#include "../DBImage2.h"
#include "../AlgoPCA.h"
#include "../AlgoLDA.h"
#include "../AlgoHISTOGRAM.h"
#include "../AlgoPCALDA.h"
#include "../Discrimination.h"
#include <pthread.h>

class KinectBasedFaceRecVal : public QMainWindow
{
	Q_OBJECT
public : 
	KinectBasedFaceRecVal();

private :
};

#endif // KINECTBASEDFACERECVAL_H
