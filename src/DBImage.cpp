#include "DBImage.h"

using namespace std;
using namespace cv;

#include "InlineMatrixOperations.cpp"

DBImage::DBImage()
{
	allocated = false;
}

DBImage::DBImage(QString database)
{
	allocated = false;
	loadDatabase(database);
}

void DBImage::loadDatabase(QString database)
{
	QDir directory;
	directory = QDir::current(); // Catch the application's working directory
	nbrImgPerFace = 0;
	nbrFaces = 0;
	if(directory.exists(database))
	{ 
		if(allocated == true)
		{
			faces.resize(0);
		}
		loadPath = database;
		this->imagesLoader();
		allocated = true;
	}
	else
	{
		qWarning() << "DBImage: Error, Folder doesn't exist. <Path:" << database << ">";
	}
}

void DBImage::imagesLoader()
{
	QDir directory,directory2;
	directory = QDir::current(); // Catch the application's working directory
	directory.setPath(loadPath);
	directory2 = QDir::current(); // Catch the application's working directory
	directory2.setPath(loadPath);
	QStringList folders = directory.entryList(QStringList(),QDir::AllDirs | QDir::NoDotAndDotDot,QDir::Name);
	dbName = directory.dirName();
	FaceDetectionVJ fd("FaceDetection-VJ.haarcascade_frontalface_alt.xml");

	//cout << folders.size() << endl; // -- Check -- Size = Nombre de faces
	faces.reserve(folders.size());
	nbrFaces = folders.size();
	string filename,filename2;

	srand(time(NULL));

	// RGB Loading
	for(int i=0, cpt=0; i<folders.size();i++,cpt=0) {
		faces.push_back(Face());

		if(directory2.exists(folders.at(i)+"/depth"))
		{
			faces[i].depth = true;
			directory2.cd(folders.at(i)+"/depth");
			directory.cd(folders.at(i)+"/rgb");
			QStringList image = directory.entryList(QDir::Files,QDir::Name);
			QStringList imageD = directory2.entryList(QDir::Files,QDir::Name);
		
			//faces[i].paths.reserve(image.size()); //For future use
			faces[i].name = folders.at(i);
			nbrImgPerFace = image.size()-1; // On laisse toujours la last image non utilisée pr tester
			faces[i].images.reserve(nbrImgPerFace); // Allocation des images - On utilise pas le last élément pour comparer
			faces[i].imgDepth.reserve(nbrImgPerFace);

			for(int j=0, validation=int(rand()%image.size()); j<image.size();j++)
			{
				if(j==validation)
				{
					filename = qPrintable(directory.path()+"/"+image.at(validation));
					filename2 = qPrintable(directory2.path()+"/"+imageD.at(validation));

					if(dbName == "texas")
					{
						Mat dest(140,100,CV_8UC3,Scalar(0));
						resize(imread(filename),dest,Size(140,100),0,0,INTER_AREA);
						faces[i].valImage = preProcessing(dest);

						faces[i].valImgDepth = Mat(140,100,CV_8UC1,Scalar(0));
						resize(imread(filename2,0),faces[i].valImgDepth,Size(140,100),0,0,INTER_AREA);
					}
					else
					{
						faces[i].valImage = preProcessing(imread(filename));
						faces[i].valImgDepth = imread(filename2,0);
					}
				}
				else
				{
					filename = qPrintable(directory.path()+"/"+image.at(j));
					filename2 = qPrintable(directory2.path()+"/"+imageD.at(j));
					faces[i].images.push_back(Mat()); // Ce sera un Mat() pr le chargement
					faces[i].imgDepth.push_back(Mat());
					//faces[i].images.push_back(std::string()); //For future use
					
					if(dbName == "texas")
					{
						Mat dest(140,100,CV_8UC3,Scalar(0));
						resize(imread(filename),dest,Size(140,100),0,0,INTER_AREA);
						faces[i].images[cpt] = preProcessing(dest);

						faces[i].imgDepth[cpt] = Mat(140,100,CV_8UC1,Scalar(0));
						resize(imread(filename2,0),faces[i].imgDepth[cpt],Size(140,100),0,0,INTER_AREA); // Ce sera un imread ici
						//faces[i].paths[j] = filename; //For future use
					}
					else
					{
						faces[i].images[cpt] = preProcessing(imread(filename));
						faces[i].imgDepth[cpt] =  imread(filename2,0);
					}
					cpt++;
				}
				filename.erase();
				filename2.erase();
			}

			cpt=0;
			directory.setPath(loadPath);
			directory2.setPath(loadPath);
		}
		else
		{
			faces[i].depth = false;
			directory.cd(folders.at(i)+"/rgb");
			QStringList image1 = directory.entryList(QDir::Files,QDir::Name);
			
			faces[i].name = folders.at(i);
			nbrImgPerFace = image1.size()-1; // On laisse toujours la last image non utilisée pr tester
			faces[i].images.reserve(image1.size()-1); // Allocation des images - On utilise pas le last élément pour comparer

			filename.erase();

			for(int j = 0, validation=int(rand()%image1.size()); j<image1.size();j++){
				if(j==validation){
					filename = qPrintable(directory.path()+"/"+image1.at(validation));
					if(dbName == "40f") faces[i].valImage = imread(filename,0);
					else
					{
						faces[i].valImage = preProcessing(imread(filename));
					}
				}
				else{
					filename = qPrintable(directory.path()+"/"+image1.at(j));

					faces[i].images.push_back(Mat());
					// Loading

					if(dbName == "40f") faces[i].images[cpt] = imread(filename,0);
					else
					{
						faces[i].images[cpt] = preProcessing(imread(filename));	
					}
					cpt++;	
				}
				filename.erase();
			}
			cpt=0;
			directory.setPath(loadPath);		
		}
	}
	if(faces[0].depth) fusionMat();
}

void DBImage::fusionMat(){
	for(int i = 0 ; i < nbrFaces; i ++)
	{
		faces[i].fusion.reserve(nbrImgPerFace);
		for(int j = 0; j < nbrImgPerFace; j++)
		{
			faces[i].fusion.push_back(Mat(faces[0].images[0].rows,faces[0].images[0].cols,CV_64FC1,Scalar(0)));
			for(int k = 0 ; k < faces[0].images[0].rows; k++)
				for(int l = 0 ; l < faces[0].images[0].cols; l++)
				{
					faces[i].fusion[j].at<double>(k,l) = faces[i].images[j].at<double>(k,l) + faces[i].imgDepth[j].at<uchar>(k,l);
				}
		}
	}
	//cout << " done " << endl;
}

cv::Mat DBImage::preProcessing(cv::Mat A)
{
	Mat proC(A.rows,A.cols,CV_64FC1,Scalar(0));
	Mat temp(A.rows,A.cols,CV_64FC3,Scalar(0));
	// Passage en Chromatics
	for(int i = 0 ; i < A.rows ; i ++)
		for(int j = 0 ; j < A.cols ; j++)
		{
			double denum = A.at<Vec3b>(i,j)[0] + A.at<Vec3b>(i,j)[1] + A.at<Vec3b>(i,j)[2];
			if(denum != 0)
			{
				temp.at<Vec3d>(i,j)[0] = (double) A.at<Vec3b>(i,j)[0] / denum; // B
				temp.at<Vec3d>(i,j)[1] = (double) A.at<Vec3b>(i,j)[1] / denum; // G
				temp.at<Vec3d>(i,j)[2] = (double) A.at<Vec3b>(i,j)[2] / denum; // R
			}
			else 
			{
				temp.at<Vec3d>(i,j)[0] = (double) A.at<Vec3b>(i,j)[0]; // B
				temp.at<Vec3d>(i,j)[1] = (double) A.at<Vec3b>(i,j)[1]; // G
				temp.at<Vec3d>(i,j)[2] = (double) A.at<Vec3b>(i,j)[2]; // R
			}
		}

	// Grayscale
	for(int i = 0 ; i < A.rows ; i ++)
		for(int j = 0; j < A.cols ; j++)
			proC.at<double>(i,j) = 0.114 * temp.at<Vec3d>(i,j)[0] + 0.587 * temp.at<Vec3d>(i,j)[1] + 0.299 * temp.at<Vec3d>(i,j)[2];

	return proC;
}

vector<Face> DBImage::getFaces(){
	return faces;
}

vector<Face>& DBImage::getFacesRef(){
	return faces;
}

int DBImage::getNbrImgPerFace(){
	return nbrImgPerFace;
}

int DBImage::getNbrFaces(){
	return nbrFaces;
}

QString DBImage::getDbName(){
	return dbName;
}

bool DBImage::getState(){
	return allocated;
}