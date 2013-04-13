#include "DBImage2.h"

DBImage::DBImage (QString database, typeLoading load, Mixer mix): loadPath(database), load(load), mix(mix)
{
	nbrImgPerFace	= -1;
	nbrFaces		= -1;
	personTmp		= NULL;
	idLoaded		= -1;

	if(checkDatabase()) loadDatabase();

}

DBImage::~DBImage()
{
	if(personTmp!=NULL) delete personTmp;
}

void DBImage::clearPerson()
{
	if(personTmp!=NULL) delete personTmp; personTmp=0;
}

Person& DBImage::operator[] (int id){
	//cout << "peon[" << id <<"]";
	if (id!=idLoaded)
	{
		if(personTmp!=NULL) delete personTmp;
		personTmp = new Person(names[id],pathRgb[id],pathDepth[id],mix);
		switch(cible)
		{
		case 0: personTmp->setCibleRGB(); break;
		case 1: personTmp->setCibleDepth(); break;
		case 2: personTmp->setCibleMix(); break;
		}
		idLoaded = id;
	}
	return *personTmp;
}

bool DBImage::checkDatabase()
{
	QDir directory;
	directory = QDir::current(); // Catch the application's working directory
	if(directory.exists(loadPath))
	{ 
		return true;
	}
	else
	{
		qWarning() << "DBImage: Error, Folder doesn't exist. <Path:" << loadPath << ">";
		return false;
	}
}

void DBImage::loadDatabase()
{
	QDir directory;
	directory=QDir::current();
	directory.setPath(loadPath);
	dbName = directory.dirName();
	
	//Getting the number of personnes
	QStringList folders = directory.entryList(QStringList(),QDir::AllDirs | QDir::NoDotAndDotDot,QDir::Name);

	nbrFaces = folders.size();

	//Allocationg the vectors
	names.resize(nbrFaces);
	pathRgb.resize(nbrFaces);
	pathDepth.resize(nbrFaces);

	//Listing each directory
	for(int personID=0; personID<folders.size();personID++)
	{
		names[personID] = qPrintable(folders.at(personID));
		directory.cd(folders.at(personID));
		if(load==RGB || load==BOTH)
		{
			directory.cd("rgb");
			QStringList imageList = directory.entryList(QDir::Files,QDir::Name);

			/*if(nbrImgPerFace!=-1 && nbrImgPerFace != imageList.size())
				qWarning("DBImage: Warning, nbrImgPerFace is not constant during loading.");*/
			nbrImgPerFace = imageList.size();

			pathRgb[personID].resize(nbrImgPerFace);
			for(int imgID=0;imgID<nbrImgPerFace;imgID++)
				pathRgb[personID][imgID] = qPrintable(directory.path()+"/"+imageList.at(imgID));

			directory.cdUp();
		}

		if(load==DEPTH || load==BOTH)
		{
			directory.cd("depth");
			QStringList imageList = directory.entryList(QDir::Files,QDir::Name);

			/*if(nbrImgPerFace!=-1 && nbrImgPerFace != imageList.size())
				qWarning("DBImage: Warning, nbrImgPerFace is not constant during loading.");*/
			nbrImgPerFace = imageList.size();

			pathDepth[personID].resize(nbrImgPerFace);
			for(int imgID=0;imgID<nbrImgPerFace;imgID++)
				pathDepth[personID][imgID] = qPrintable(directory.path()+"/"+imageList.at(imgID));

			directory.cdUp();
		}
		directory.cdUp();
	}
}

int DBImage::getNbrImgPerFace()
{
	return nbrImgPerFace;
}

int DBImage::getNbrFaces()
{
	return nbrFaces;
}

int DBImage::size()
{
	return getNbrFaces();
}

int DBImage::lastLoaded()
{
	return (idLoaded>=0?idLoaded:0);
}

int DBImage::cols()
{
	return (*this)[this->lastLoaded()].cols();
}
int DBImage::rows()
{
	return (*this)[this->lastLoaded()].rows();
}

//void DBImage::setValidationOnly()
//{
//	if(onlyValidationImages==false && personTmp!=NULL)
//	{
//		delete personTmp; personTmp=NULL;
//		idLoaded = -1;
//	}
//	onlyValidationImages=true;
//}
//void DBImage::unsetValidationOnly()
//{
//	if(onlyValidationImages==true && personTmp!=NULL)
//	{
//		delete personTmp; personTmp=NULL;
//		idLoaded = -1;
//	}
//	onlyValidationImages=false;
//}

std::string DBImage::getName(int id)
{
	return names[id];
}

QString DBImage::getLoadPath()
{
	return loadPath;
}

int DBImage::getIDFromName(std::string name)
{
	int retour = -1;
	for(int i=0; i<names.size(); i++)
	{
		if(names[i]==name){ retour=i; break; }
	}
	return retour;
}