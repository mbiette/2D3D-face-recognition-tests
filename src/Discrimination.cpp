#include "Discrimination.h"
#include <pthread.h>

using namespace std;


//Confirm::Confirm(DBImage &db, Print *toPrint, typeLoading image, typeDistance distance, typeDecision decision, double parameter, typeDBVal val): db(db), img(image),dist(distance),classif(decision),param(parameter),val(val)
//{
//	db.setValidationOnly();
//	print=toPrint;
//}
//
//
//void Confirm::exec(AlgoPCA &pca)
//{
//	print->newPass();
//	if(val==MFOLD)
//	{
//		for(int i=0; i<db.size(); i++)
//		{
//			int val;
//			//cout << algoToUse <<endl;
//
//			switch(img){
//			case RGB:
//				val = pca.classification(dist,classif,param,db[i].rgbVal[0]); break;
//			case DEPTH:
//				val = pca.classification(dist,classif,param,db[i].depthVal[0]); break;
//			case MIX:
//				val = pca.classification(dist,classif,param,db[i].rgbVal[0],db[i].depthVal[0]); break;
//			}
//
//			print->add(i,val);
//		}
//	}
//	else
//	{
//		DBImage valDB(db.getLoadPath(),img,val,TESTPROBE);
//		//DBImage database(config->pathToDB,config->db,config->imgToLoad,config->val,config->set);
//		for(int i=0; i<valDB.size(); i++)
//		{
//			for(int j=0; j<valDB[i].size(); j++)
//			{
//				int val;
//				//cout << algoToUse <<endl;
//
//				switch(img){
//				case RGB:
//					val = pca.classification(dist,classif,param,valDB[i].rgb[j]); break;
//				case DEPTH:
//					val = pca.classification(dist,classif,param,valDB[i].depth[j]); break;
//				case MIX:
//					val = pca.classification(dist,classif,param,valDB[i].rgb[j],valDB[i].depth[j]); break;
//				}
//				//cout << "i:"<<i << " j:"<<val <<endl;
//				print->add(db.getIDFromName(valDB.getName(i)),val);
//			}
//		}
//	}
//}
//
//void Confirm::exec(AlgoLDA &lda)
//{
//	print->newPass();
//	if(val==MFOLD)
//	{
//		for(int i=0; i<db.size(); i++)
//		{
//			int val;
//			//cout << algoToUse <<endl;
//
//			switch(img){
//			case RGB:
//				val = lda.classification(dist,classif,param,db[i].rgbVal[0]); break;
//			case DEPTH:
//				val = lda.classification(dist,classif,param,db[i].depthVal[0]); break;
//			case MIX:
//				val = lda.classification(dist,classif,param,db[i].rgbVal[0],db[i].depthVal[0]); break;
//			}
//
//			print->add(i,val);
//		}
//	}
//	else
//	{
//		DBImage valDB(db.getLoadPath(),img,val,TESTPROBE);
//		//DBImage database(config->pathToDB,config->db,config->imgToLoad,config->val,config->set);
//		for(int i=0; i<valDB.size(); i++)
//		{
//			for(int j=0; j<valDB[i].size(); j++)
//			{
//				int val;
//				//cout << algoToUse <<endl;
//
//				switch(img){
//				case RGB:
//					val = lda.classification(dist,classif,param,valDB[i].rgb[j]); break;
//				case DEPTH:
//					val = lda.classification(dist,classif,param,valDB[i].depth[j]); break;
//				case MIX:
//					val = lda.classification(dist,classif,param,valDB[i].rgb[j],valDB[i].depth[j]); break;
//				}
//				//cout << "i:"<<i << " j:"<<val <<endl;
//				print->add(db.getIDFromName(valDB.getName(i)),val);
//			}
//		}
//	}
//}
//
//void Confirm::exec(AlgoHISTOGRAM &histo)
//{
//	print->newPass();
//	if(val==MFOLD)
//	{
//		for(int i=0; i<db.size(); i++)
//		{
//			int val;
//			//cout << algoToUse <<endl;
//
//			switch(img){
//			//case RGB:
//			//	val = histo.classification(dist,classif,param,db[i].rgbVal[0]); break;
//			case DEPTH:
//				val = histo.classification(dist,classif,param,db[i].depthVal[0]); break;
//			//case MIX:
//			//	val = histo.classification(dist,classif,param,db[i].rgbVal[0],db[i].depthVal[0]); break;
//			}
//
//			print->add(i,val);
//		}
//	}
//	else
//	{
//		DBImage valDB(db.getLoadPath(),img,val,TESTPROBE);
//		//DBImage database(config->pathToDB,config->db,config->imgToLoad,config->val,config->set);
//		for(int i=0; i<valDB.size(); i++)
//		{
//			for(int j=0; j<valDB[i].size(); j++)
//			{
//				int val;
//				//cout << algoToUse <<endl;
//
//				switch(img){
//				//case RGB:
//					//val = histo.classification(dist,classif,param,valDB[i].rgb[j]); break;
//				case DEPTH:
//					val = histo.classification(dist,classif,param,valDB[i].depth[j]); break;
//				//case MIX:
//					//val = histo.classification(dist,classif,param,valDB[i].rgb[j],valDB[i].depth[j]); break;
//				}
//				//cout << "i:"<<i << " j:"<<val <<endl;
//				print->add(db.getIDFromName(valDB.getName(i)),val);
//			}
//		}
//	}
//}

Print::Print(DBImage &db)
{
	nbPass = 0;

	//Confusion matrix
	confusion.resize(db.getNbrFaces());
	for(int i=0;i<db.getNbrFaces();i++) confusion[i].resize(db.getNbrFaces(),0);
	confusionNbSamples.resize(db.getNbrFaces(),0);
	/*confusionSumSamples.resize(db.getNbrFaces(),0);*/
	confusionRecognized.resize(db.getNbrFaces(),0);
	confusionErrors.resize(db.getNbrFaces(),0);
	confusionIncorrectRecognized.resize(db.getNbrFaces(),0);
	confusionNotRecognized.resize(db.getNbrFaces(),0);
	sumRecognized = 0;
	sumErrors = 0;
	sumIncorrectRecognized = 0;
	sumNotRecognized = 0;

	//General representation of errors
	sumTruePositiv = 0; sumFalseNegativ = 0;
	sumFalsePositiv = 0; sumTrueNegativ = 0;

	nbFaces = db.getNbrFaces();
	nbSamples = 0; nbSamplesDetected = 0;

	nameFaces.resize(db.getNbrFaces());
	for(int i=0;i<db.getNbrFaces();i++) nameFaces[i] = db.getName(i);
}

pthread_mutex_t mutPass = PTHREAD_MUTEX_INITIALIZER;
void Print::newPass()
{
	pthread_mutex_lock(&mutPass);
	nbPass++;
	pthread_mutex_unlock(&mutPass);
}

pthread_mutex_t mutAdd = PTHREAD_MUTEX_INITIALIZER;
void Print::add(int idWanted, int idFound)
{
	pthread_mutex_lock(&mutAdd);
	nbSamples++;
	if(idWanted>=0 && idWanted<nbFaces)
	{
		confusionNbSamples[idWanted]++;
		nbSamplesDetected++;
		if(idFound>=0 && idFound<nbFaces)
		{
			confusion[idWanted][idFound]++;
			//confusionSumSamples[idFound]++;
			if(idWanted==idFound)
			{
				confusionRecognized[idWanted]++; sumRecognized++;
				sumTruePositiv++;
			}
			else	
			{
				confusionIncorrectRecognized[idWanted]++; sumIncorrectRecognized++;
				confusionErrors[idWanted]++; sumErrors++;
				sumFalsePositiv++;
			}
		}
		else
		{
			confusionNotRecognized[idWanted]++; sumNotRecognized++;
			confusionErrors[idWanted]++; sumErrors++;
			sumFalseNegativ++;
		}
	}
	else
	{
		if(idFound>=0 && idFound<nbFaces)
		{
			//confusionErrors[i]++; sumErrors++;
			sumFalsePositiv++;
		}
		else sumTrueNegativ++;
	}

	pthread_mutex_unlock(&mutAdd);
}

void Print::print(string filename)
{
	ofstream file(filename);

	if(!file) { 
		cout << "Print: Fatal, Cannot open file.\n"; 
		system("pause");
		exit(1); 
	}

	file << "nbPass:\t" << nbPass <<endl;
	file << "nbFaces:\t" << nbFaces <<endl;
	file << "nbSamples:\t" << nbSamples <<endl;
	file << "sumRecognized:\t" << sumRecognized <<endl;
	file << "sumErrors:\t" << sumErrors <<endl;
	file << "sumIncorrectRecognized:\t" << sumIncorrectRecognized <<endl;
	file << "sumNotRecognized:\t" << sumNotRecognized <<endl;
	file << "Succes rate:\t" << double(sumRecognized)/double(nbSamplesDetected) <<endl;
	file << "Error rate:\t" << double(sumErrors)/double(nbSamplesDetected) <<endl;
	file << "Incorrect rate:\t" << double(sumIncorrectRecognized)/double(nbSamplesDetected) <<endl;
	file << "NotReco rate:\t" << double(sumNotRecognized)/double(nbSamplesDetected) <<endl;
 
	file << endl << endl;

	file << "\t"	<< "yes\t"					<< "no"				<<endl;
	file << "yes\t" << sumTruePositiv <<"\t"	<< sumFalseNegativ	<<endl;
	file << "no\t"  << sumFalsePositiv <<"\t"	<< sumTrueNegativ	<<endl;
	file << endl;
	file << "False Positiv Rate:\t" << this->getFPR() << endl;
	file << "True Positiv Rate:\t" << this->getTPR() << endl;
	file << endl << endl;

	//Write the first line with the names of all the faces.
	file << "\t\t\t\t";
	for(int i=0; i<nbFaces; i++)
		file << nameFaces[i] << "\t";
	file << "Nb Samples\tRecognized\tErrors\tIncorrectRecognized\tNotRecognized" ;
	file << endl;

	//Writing all the lines of the table
	for(int i=0; i<nbFaces; i++)
	{
		file << "\t\t\t" << nameFaces[i] << "\t";
		for(int j=0; j<nbFaces; j++)
			file << confusion[i][j] << "\t";
		file << confusionNbSamples[i] <<"\t"<< confusionRecognized[i]<<"\t"<< confusionErrors[i]<<"\t"<< confusionIncorrectRecognized[i]<<"\t"<< confusionNotRecognized[i];
		file << endl;
	}

	//Closing
	file.close();
}

double Print::getFPR()
{
	return double(sumFalsePositiv)/double(sumFalsePositiv+sumTrueNegativ);
}
double Print::getTPR()
{
	return double(sumTruePositiv)/double(sumTruePositiv+sumFalseNegativ);
}

ROC::ROC()
{
}

void ROC::add(double th, double fPR, double tPR)
{
	threshold.push_back(th);
	falsePositivRate.push_back(fPR);
	truePositivRate.push_back(tPR);
}

void ROC::print(std::string filename)
{
	ofstream file(filename);

	if(!file) { 
		cout << "ROC: Fatal, Cannot open file.\n"; 
		system("pause");
		exit(1); 
	}

	file <<"threshold\tfalsePositivRate\ttruePositivRate"<<endl;
	for(int i=0; i<threshold.size(); i++)
		file << threshold[i] <<"\t"<< falsePositivRate[i] <<"\t"<< truePositivRate[i]<<endl;

	file.close();
}