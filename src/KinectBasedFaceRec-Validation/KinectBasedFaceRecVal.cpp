#include "KinectBasedFaceRecVal.h"
#include <limits>

#include <time.h>
#include <sstream>

using namespace cv;
using namespace std;

#include "../InlineMatrixOperations.cpp"

string createFilename(string prefix, typeLoading _load, Mixer _mix, typeAlgo _algo, typeDistance _dist, typeDecision _deci, double param)
{
	// Generating the output filename
	string filePath;
	
	filePath.append(prefix);
	filePath.append("-");

	switch(_load){
	case RGB: filePath.append("rgb"); break;
	case DEPTH: filePath.append("depth"); break;
	case BOTH: filePath.append("both"); break;
	}
	filePath.append("-");

	switch(_mix){
	case MIX_NO: filePath.append("no_mix"); break;
	case MIX_SUM: filePath.append("sum_mix"); break;
	case MIX_CONCAT: filePath.append("concat_mix"); break;
	}
	filePath.append("-");

	switch(_algo){
	case ALGOPCA: filePath.append("pca"); break;
	case ALGOLDA: filePath.append("lda"); break;
	case ALGOHISTO: filePath.append("histo"); break;
	case ALGOPCALDA: filePath.append("fishersurface"); break;
	}
	filePath.append("-");

	switch(_dist){
	case NORM1: filePath.append("norm1"); break;
	case NORM2: filePath.append("norm2"); break;
	case MAHALANOBIS: filePath.append("mahalonobis"); break;
	}
	filePath.append("-");

	std::ostringstream oss;
	oss << param;
	switch(_deci){
	case THRESHOLD: filePath.append("threshold("); filePath.append(oss.str()); filePath.append(")"); break;
	case CONFIDANCE: filePath.append("confidance("); filePath.append(oss.str()); filePath.append(")"); break;
	case KNEAREST: filePath.append("knearest("); filePath.append(oss.str()); filePath.append(")"); break;
	}
	filePath.append(".txt");

	return filePath;
}

void launch_NOMIX(string pathToDB, string pathToDBTest, typeLoading _load, typeAlgo _algo, typeDistance _dist, typeDecision _deci)
{
	clock_t start,ends;
	cout << " ######## Launch_NOMIX  ######## " <<endl;

	DBImage *databaseTraining = new DBImage(QString(pathToDB.c_str()),_load,MIX_NO);
	if(_load==RGB) databaseTraining->setCibleRGB();
	else databaseTraining->setCibleDepth();

	/* ALGO */
		/* Training */
	AlgoPCA* pca=NULL;
	AlgoLDA* lda=NULL;
	AlgoHISTOGRAM* histo=NULL;
	AlgoPCALDA* pcalda=NULL;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA training... ";
		pca = new AlgoPCA(*databaseTraining, 0.99);
		pca->launch();
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA training... ";
		lda = new AlgoLDA(*databaseTraining);
		lda->launch();
		break;
	case ALGOHISTO:
		histo = new AlgoHISTOGRAM(/**databaseTraining*/);
	//	histo->launch();
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA training... ";
		pcalda = new AlgoPCALDA(*databaseTraining, 0.99);
		pcalda->launch();
		break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the training feature vectors */
	vector<vector<cv::Mat>> featureVectorTraining;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (training)... ";
		featureVectorTraining = pca->featureVectOUT(*databaseTraining);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (training)... ";
		featureVectorTraining = lda->featureVectOUT(*databaseTraining);
		break;
	case ALGOHISTO:
		start = clock(); cout << "AlgoHISTO feature vectors (training)... ";
		featureVectorTraining = histo->featureVectOUT(*databaseTraining);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA feature vectors (training)... ";
		featureVectorTraining = pcalda->featureVectOUT(*databaseTraining);
		break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the test feature vectors */
	DBImage *databaseTest = new DBImage(QString(pathToDBTest.c_str()),_load,MIX_NO);
	if(_load==RGB) databaseTest->setCibleRGB();
	else databaseTest->setCibleDepth();
	
	vector<vector<cv::Mat>> featureVectorTest;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (testing)... ";
		featureVectorTest = pca->featureVectOUT(*databaseTest);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (testing)... ";
		featureVectorTest = lda->featureVectOUT(*databaseTest);
		break;
	case ALGOHISTO:
		start = clock(); cout << "AlgoHISTO feature vectors (testing)... ";
		featureVectorTest = histo->featureVectOUT(*databaseTest);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA feature vectors (testing)... ";
		featureVectorTest = pcalda->featureVectOUT(*databaseTest);
		break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Fill the correspondance vector */
	start = clock(); cout << "Correspondance Map... ";
	vector<int> trainingCorrespondance(featureVectorTest.size());
	for(int i=0;i<featureVectorTest.size();i++) trainingCorrespondance[i] = databaseTraining->getIDFromName(databaseTest->getName(i));
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;


	start = clock(); cout << "Freeing (some) db space... ";
	//delete databaseTraining; databaseTraining=0;
	delete databaseTest; databaseTest=0;
	databaseTraining->clearPerson();
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;



	/* DIST */
	vector<vector<vector<vector<double>>>> distances(featureVectorTest.size());
	vector<Mat> mahalanobis_mean, mahalanobis_invcov;
		/* Mahalanobis training */
	if(_dist == MAHALANOBIS)
	{
		start = clock(); cout <<"Mahalanobis training... ";
		mahalanobis_mean.reserve(featureVectorTraining.size());
		mahalanobis_invcov.reserve(featureVectorTraining.size());

		for(int i=0;i<featureVectorTraining.size();i++)
		{
			mahalanobis_mean.push_back(Mat());
			mahalanobis_invcov.push_back(Mat());
			mahalanobis_mean[i] = meanMat<double>(featureVectorTraining[i]);
			mahalanobis_invcov[i] = (featureCovMat<double>(featureVectorTraining[i],mahalanobis_mean[i])).inv();
		}
		ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
	}
		/* Distances computation */
	start = clock(); cout <<"Distance computation... ";
	//cout << "featureVectorTest.size():"<<featureVectorTest.size()<<endl;
	for(int test_i=0;test_i<featureVectorTest.size();test_i++)
	{
		distances[test_i].resize(featureVectorTest[test_i].size());
		//cout << "featureVectorTest[test_i].size():"<<test_i<<":"<<featureVectorTest[test_i].size()<<endl;
		for(int test_j=0;test_j<featureVectorTest[test_i].size();test_j++)
		{
			distances[test_i][test_j].resize(featureVectorTraining.size());
			//cout << "featureVectorTraining.size():"<<featureVectorTraining.size()<<endl;
			for(int train_i=0 ; train_i<featureVectorTraining.size() ; train_i++)
			{
				if(_dist == MAHALANOBIS)
				{
					distances[test_i][test_j][train_i].resize(1);
					Mat diff( mahalanobis_mean[train_i].rows, mahalanobis_mean[train_i].cols, CV_64FC1, Scalar(0));
					diff = featureVectorTest[test_i][test_j] - mahalanobis_mean[train_i];
					distances[test_i][test_j][train_i][0] = norm(diff.t()*(mahalanobis_invcov[train_i]*diff),NORM_L2);
				}
				else
				{
					distances[test_i][test_j][train_i].resize(featureVectorTraining[train_i].size());
					//cout << "featureVectorTraining[train_i].size():"<<train_i<<":"<<featureVectorTraining[train_i].size()<<endl;
					for(int train_j=0 ; train_j<featureVectorTraining[train_i].size() ; train_j++)
					{
						if(_dist == NORM1)
							distances[test_i][test_j][train_i][train_j] = norm(featureVectorTraining[train_i][train_j],featureVectorTest[test_i][test_j],NORM_L1);
						else if(_dist == NORM2)
							distances[test_i][test_j][train_i][train_j] = norm(featureVectorTraining[train_i][train_j],featureVectorTest[test_i][test_j],NORM_L2);
					}
				}
			}
		}
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Decision */
	start = clock(); cout <<"Decision... ";

	vector<vector<int>> decisionID;
	decisionID.resize(distances.size());

	vector<vector<double>> decisionDist;
	decisionDist.resize(distances.size());

	for(int test_i=0;test_i<distances.size();test_i++)
	{
		decisionID[test_i].resize(distances[test_i].size(),-1);
		decisionDist[test_i].resize(distances[test_i].size(),numeric_limits<double>::max());
		for(int test_j=0;test_j<distances[test_i].size();test_j++)
		{
			if(_deci==THRESHOLD)
			{
				/* Nearest */
				for(int train_i=0 ; train_i<distances[test_i][test_j].size() ; train_i++)
				{
					for(int train_j=0 ; train_j<distances[test_i][test_j][train_i].size() ; train_j++)
					{
						if(decisionDist[test_i][test_j]>distances[test_i][test_j][train_i][train_j])
						{
							decisionID[test_i][test_j]=train_i;
							decisionDist[test_i][test_j]=distances[test_i][test_j][train_i][train_j];
						}
					}
				}
			}
		}
	}
	ends = clock(); cout<<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Count */
	start = clock(); cout<<"Statistics... ";
	if(_deci==THRESHOLD)
	{
		ROC roc;

		/* We test with a maximum as a threshold */
		{
			Print print(*databaseTraining);
			print.newPass();
			for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
				for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
				{
					print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<numeric_limits<double>::max()? decisionID[deci_i][deci_j]:-1) );
				}
			print.print(createFilename("NOMIX", _load, MIX_NO, _algo, _dist, _deci, numeric_limits<double>::max()));
			roc.add(numeric_limits<double>::max(),print.getFPR(),print.getTPR());
		}
		/* Here each decision correspond to a threhold level */
		for(int thr_i=0;thr_i<decisionDist.size();thr_i++)
			for(int thr_j=0;thr_j<decisionDist[thr_i].size();thr_j++)
			{
				Print print(*databaseTraining);
				print.newPass();
				for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
					for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
					{
						print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<decisionDist[thr_i][thr_j]? decisionID[deci_i][deci_j]:-1) );
					}
				print.print(createFilename("NOMIX", _load, MIX_NO, _algo, _dist, _deci, decisionDist[thr_i][thr_j]));
				roc.add(decisionDist[thr_i][thr_j],print.getFPR(),print.getTPR());
			}
		roc.print(createFilename("ROC-NOMIX", _load, MIX_NO, _algo, _dist, _deci, 0));
	}
	ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
	
}

void launch_LATEMIX(string pathToDB, string pathToDBTest, typeAlgo _algo, typeDistance _dist, typeDecision _deci)
{
	clock_t start,ends;
	cout << " ######## Launch_LATEMIX  ######## " <<endl;

	DBImage *databaseTraining = new DBImage(QString(pathToDB.c_str()),BOTH,MIX_NO);
	//databaseTraining->setCibleMix();

	/* ALGO */
		/* Training */
	AlgoPCA *pcaRGB=NULL,*pcaDepth=NULL;
	AlgoLDA *ldaRGB=NULL,*ldaDepth=NULL;
	AlgoPCALDA *pcaldaRGB=NULL,*pcaldaDepth=NULL;
	//AlgoHISTOGRAM* histo=NULL;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA training... ";
		databaseTraining->setCibleRGB();
		pcaRGB = new AlgoPCA(*databaseTraining, 0.99);
		pcaRGB->launch();
		databaseTraining->setCibleDepth();
		pcaDepth = new AlgoPCA(*databaseTraining, 0.99);
		pcaDepth->launch();
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA training... ";
		databaseTraining->setCibleRGB();
		ldaRGB = new AlgoLDA(*databaseTraining);
		ldaRGB->launch();
		databaseTraining->setCibleDepth();
		ldaDepth = new AlgoLDA(*databaseTraining);
		ldaDepth->launch();
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA training... ";
		databaseTraining->setCibleRGB();
		pcaldaRGB = new AlgoPCALDA(*databaseTraining, 0.99);
		pcaldaRGB->launch();
		databaseTraining->setCibleDepth();
		pcaldaDepth = new AlgoPCALDA(*databaseTraining, 0.99);
		pcaldaDepth->launch();
		break;
	//case ALGOHISTO:
	//	histo = new AlgoHISTOGRAM(*databaseTraining);
	////	histo->launch();
	//	break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the training feature vectors */
	vector<vector<cv::Mat>> featureVectorTrainingRGB;
	vector<vector<cv::Mat>> featureVectorTrainingDepth;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (training)... ";
		databaseTraining->setCibleRGB();
		featureVectorTrainingRGB = pcaRGB->featureVectOUT(*databaseTraining);
		databaseTraining->setCibleDepth();
		featureVectorTrainingDepth = pcaDepth->featureVectOUT(*databaseTraining);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (training)... ";
		databaseTraining->setCibleRGB();
		featureVectorTrainingRGB = ldaRGB->featureVectOUT(*databaseTraining);
		databaseTraining->setCibleDepth();
		featureVectorTrainingDepth = ldaDepth->featureVectOUT(*databaseTraining);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA feature vectors (training)... ";
		databaseTraining->setCibleRGB();
		featureVectorTrainingRGB = pcaldaRGB->featureVectOUT(*databaseTraining);
		databaseTraining->setCibleDepth();
		featureVectorTrainingDepth = pcaldaDepth->featureVectOUT(*databaseTraining);
		break;
	//case ALGOHISTO:
	//	start = clock(); cout << "AlgoHISTO feature vectors (training)... ";
	//	databaseTraining->setCibleRGB();
	//	featureVectorTrainingRGB = histo->featureVectOUT(*databaseTraining);
	//	databaseTraining->setCibleDepth();
	//	featureVectorTrainingDepth = histo->featureVectOUT(*databaseTraining);
	//	break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the test feature vectors */
	DBImage *databaseTest = new DBImage(QString(pathToDBTest.c_str()),BOTH,MIX_NO);
	
	vector<vector<cv::Mat>> featureVectorTestRGB;
	vector<vector<cv::Mat>> featureVectorTestDepth;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (testing)... ";
		databaseTest->setCibleRGB();
		featureVectorTestRGB = pcaRGB->featureVectOUT(*databaseTest);
		databaseTest->setCibleDepth();
		featureVectorTestDepth = pcaDepth->featureVectOUT(*databaseTest);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (testing)... ";
		databaseTest->setCibleRGB();
		featureVectorTestRGB = ldaRGB->featureVectOUT(*databaseTest);
		databaseTest->setCibleDepth();
		featureVectorTestDepth = ldaDepth->featureVectOUT(*databaseTest);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA feature vectors (testing)... ";
		databaseTest->setCibleRGB();
		featureVectorTestRGB = pcaldaRGB->featureVectOUT(*databaseTest);
		databaseTest->setCibleDepth();
		featureVectorTestDepth = pcaldaDepth->featureVectOUT(*databaseTest);
		break;
	/*case ALGOHISTO:
		start = clock(); cout << "AlgoHISTO feature vectors (testing)... ";
		featureVectorTest = histo->featureVectOUT(databaseTest);
		break;*/
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Fill the correspondance vector */
	start = clock(); cout << "Correspondance Map... ";
	vector<int> trainingCorrespondance(featureVectorTestRGB.size());
	for(int i=0;i<featureVectorTestRGB.size();i++) trainingCorrespondance[i] = databaseTraining->getIDFromName(databaseTest->getName(i));
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;


	start = clock(); cout << "Freeing (some) db space... ";
	//delete databaseTraining; databaseTraining=0;
	delete databaseTest; databaseTest=0;
	databaseTraining->clearPerson();
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;



	/* DIST */
	vector<vector<vector<vector<double>>>> distancesRGB(featureVectorTestRGB.size());
	vector<vector<vector<vector<double>>>> distancesDepth(featureVectorTestDepth.size());
	vector<Mat> mahalanobis_meanRGB, mahalanobis_invcovRGB,mahalanobis_meanDepth, mahalanobis_invcovDepth;
		/* Mahalanobis training */
	if(_dist == MAHALANOBIS)
	{
		start = clock(); cout <<"Mahalanobis training... ";
		mahalanobis_meanRGB.reserve(featureVectorTrainingRGB.size());
		mahalanobis_invcovRGB.reserve(featureVectorTrainingRGB.size());

		for(int i=0;i<featureVectorTrainingRGB.size();i++)
		{
			mahalanobis_meanRGB.push_back(Mat());
			mahalanobis_invcovRGB.push_back(Mat());
			mahalanobis_meanRGB[i] = meanMat<double>(featureVectorTrainingRGB[i]);
			mahalanobis_invcovRGB[i] = (featureCovMat<double>(featureVectorTrainingRGB[i],mahalanobis_meanRGB[i])).inv();
		}

		mahalanobis_meanDepth.reserve(featureVectorTrainingDepth.size());
		mahalanobis_invcovDepth.reserve(featureVectorTrainingDepth.size());

		for(int i=0;i<featureVectorTrainingDepth.size();i++)
		{
			mahalanobis_meanDepth.push_back(Mat());
			mahalanobis_invcovDepth.push_back(Mat());
			mahalanobis_meanDepth[i] = meanMat<double>(featureVectorTrainingDepth[i]);
			mahalanobis_invcovDepth[i] = (featureCovMat<double>(featureVectorTrainingDepth[i],mahalanobis_meanDepth[i])).inv();
		}
		ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
	}
		/* Distances computation */
	start = clock(); cout <<"Distance computation... ";
	//cout << "featureVectorTest.size():"<<featureVectorTest.size()<<endl;
	for(int test_i=0;test_i<featureVectorTestRGB.size();test_i++)
	{
		distancesRGB[test_i].resize(featureVectorTestRGB[test_i].size());
		distancesDepth[test_i].resize(featureVectorTestDepth[test_i].size());
		//cout << "featureVectorTest[test_i].size():"<<test_i<<":"<<featureVectorTest[test_i].size()<<endl;
		for(int test_j=0;test_j<featureVectorTestRGB[test_i].size();test_j++)
		{
			distancesRGB[test_i][test_j].resize(featureVectorTrainingRGB.size());
			distancesDepth[test_i][test_j].resize(featureVectorTrainingDepth.size());
			//cout << "featureVectorTraining.size():"<<featureVectorTraining.size()<<endl;
			for(int train_i=0 ; train_i<featureVectorTrainingRGB.size() ; train_i++)
			{
				if(_dist == MAHALANOBIS)
				{
					distancesRGB[test_i][test_j][train_i].resize(1);
					distancesDepth[test_i][test_j][train_i].resize(1);
					Mat diff( mahalanobis_meanRGB[train_i].rows, mahalanobis_meanRGB[train_i].cols, CV_64FC1, Scalar(0));
					diff = featureVectorTestRGB[test_i][test_j] - mahalanobis_meanRGB[train_i];
					distancesRGB[test_i][test_j][train_i][0] = norm(diff.t()*(mahalanobis_invcovRGB[train_i]*diff),NORM_L2);
					diff = featureVectorTestDepth[test_i][test_j] - mahalanobis_meanDepth[train_i];
					distancesDepth[test_i][test_j][train_i][0] = norm(diff.t()*(mahalanobis_invcovDepth[train_i]*diff),NORM_L2);
				}
				else
				{
					distancesRGB[test_i][test_j][train_i].resize(featureVectorTrainingRGB[train_i].size());
					distancesDepth[test_i][test_j][train_i].resize(featureVectorTrainingDepth[train_i].size());
					//cout << "featureVectorTraining[train_i].size():"<<train_i<<":"<<featureVectorTraining[train_i].size()<<endl;
					for(int train_j=0 ; train_j<featureVectorTrainingRGB[train_i].size() ; train_j++)
					{
						if(_dist == NORM1)
						{
							distancesRGB[test_i][test_j][train_i][train_j] = norm(featureVectorTrainingRGB[train_i][train_j],featureVectorTestRGB[test_i][test_j],NORM_L1);
							distancesDepth[test_i][test_j][train_i][train_j] = norm(featureVectorTrainingDepth[train_i][train_j],featureVectorTestDepth[test_i][test_j],NORM_L1);
						}
						else if(_dist == NORM2)
						{
							distancesRGB[test_i][test_j][train_i][train_j] = norm(featureVectorTrainingRGB[train_i][train_j],featureVectorTestRGB[test_i][test_j],NORM_L2);
							distancesDepth[test_i][test_j][train_i][train_j] = norm(featureVectorTrainingDepth[train_i][train_j],featureVectorTestDepth[test_i][test_j],NORM_L2);
						}
					}
				}
			}
		}
	}

	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Decision */
	start = clock(); cout <<"Decision... ";

	vector<vector<int>> decisionID;
	decisionID.resize(distancesRGB.size());

	vector<vector<double>> decisionDist;
	decisionDist.resize(distancesRGB.size());

	for(int test_i=0;test_i<distancesRGB.size();test_i++)
	{
		decisionID[test_i].resize(distancesRGB[test_i].size(),-1);
		decisionDist[test_i].resize(distancesRGB[test_i].size(),numeric_limits<double>::max());
		for(int test_j=0;test_j<distancesRGB[test_i].size();test_j++)
		{
			
			/* Find the 3 Nearest */
			vector<double> threeMinDistRGB(3,numeric_limits<double>::max());
			vector<double> threeIDRGB(3,-1);
			vector<double> threeMinDistDepth(3,numeric_limits<double>::max());
			vector<double> threeIDDepth(3,-1);

			for(int train_i=0 ; train_i<distancesRGB[test_i][test_j].size() ; train_i++)
			{
				for(int train_j=0 ; train_j<distancesRGB[test_i][test_j][train_i].size() ; train_j++)
				{
					double max = threeMinDistRGB[0];
					int maxID = 0;
					for(int i=1;i<threeMinDistRGB.size();i++)
						if(threeMinDistRGB[i]>max)
						{
							max = threeMinDistRGB[i];
							maxID = i;
						}

					if(distancesRGB[test_i][test_j][train_i][train_j]<max)
					{
						threeMinDistRGB[maxID] = distancesRGB[test_i][test_j][train_i][train_j];
						threeIDRGB[maxID] = train_i;
					}

					max = threeMinDistDepth[0];
					maxID = 0;
					for(int i=1;i<threeMinDistDepth.size();i++)
						if(threeMinDistDepth[i]>max)
						{
							max = threeMinDistDepth[i];
							maxID = i;
						}

					if(distancesDepth[test_i][test_j][train_i][train_j]<max)
					{
						threeMinDistDepth[maxID] = distancesDepth[test_i][test_j][train_i][train_j];
						threeIDDepth[maxID] = train_i;
					}
				}
			}

			/* Find the confidance */
				/* Sort the 3 nearest */
			for(int i=0; i<threeMinDistRGB.size()-1;i++)
			{
				double localmin = threeMinDistRGB[i];
				int localminid = i;
				for(int j=i; j<threeMinDistRGB.size() ; j++)
				{
					if(localmin>threeMinDistRGB[j])
					{
						localmin=threeMinDistRGB[j];
						localminid=j;
					}
				}
				if(localminid!=i)
				{
					double valtmp = threeMinDistRGB[i];
					threeMinDistRGB[i] = threeMinDistRGB[localminid];
					threeMinDistRGB[localminid]=valtmp;

					int idtmp = threeIDRGB[i];
					threeIDRGB[i] = threeIDRGB[localminid];
					threeIDRGB[localminid] = idtmp;
				}
			}

			for(int i=0; i<threeMinDistDepth.size()-1;i++)
			{
				double localmin = threeMinDistDepth[i];
				int localminid = i;
				for(int j=i; j<threeMinDistDepth.size() ; j++)
				{
					if(localmin>threeMinDistDepth[j])
					{
						localmin=threeMinDistDepth[j];
						localminid=j;
					}
				}
				if(localminid!=i)
				{
					double valtmp = threeMinDistDepth[i];
					threeMinDistDepth[i] = threeMinDistDepth[localminid];
					threeMinDistDepth[localminid]=valtmp;

					int idtmp = threeIDDepth[i];
					threeIDDepth[i] = threeIDDepth[localminid];
					threeIDDepth[localminid] = idtmp;
				}
			}

			/* Compute the confidance */
			double confRGB = (threeMinDistRGB[1]-threeMinDistRGB[0])/(threeMinDistRGB[2]-threeMinDistRGB[0]);
			double confDepth = (threeMinDistDepth[1]-threeMinDistDepth[0])/(threeMinDistDepth[2]-threeMinDistDepth[0]);

			/* Make a choice based on that */
			if(confRGB>confDepth)
			{
				decisionID[test_i][test_j]= threeIDRGB[0];
				decisionDist[test_i][test_j]=threeMinDistRGB[0];
			}
			else
			{
				decisionID[test_i][test_j]= threeIDDepth[0];
				decisionDist[test_i][test_j]=threeMinDistDepth[0];
			}
		}
	}
	ends = clock(); cout<<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Count */
	start = clock(); cout<<"Statistics... ";
	if(_deci==THRESHOLD)
	{
		ROC roc;

		/* We test with a maximum as a threshold */
		{
			Print print(*databaseTraining);
			print.newPass();
			for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
				for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
				{
					print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<numeric_limits<double>::max()? decisionID[deci_i][deci_j]:-1) );
				}
			print.print(createFilename("LATEMIX", BOTH, MIX_NO, _algo, _dist, _deci, numeric_limits<double>::max()));
			roc.add(numeric_limits<double>::max(),print.getFPR(),print.getTPR());
		}
		/* Here each decision correspond to a threhold level */
		for(int thr_i=0;thr_i<decisionDist.size();thr_i++)
			for(int thr_j=0;thr_j<decisionDist[thr_i].size();thr_j++)
			{
				Print print(*databaseTraining);
				print.newPass();
				for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
					for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
					{
						print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<decisionDist[thr_i][thr_j]? decisionID[deci_i][deci_j]:-1) );
					}
				print.print(createFilename("LATEMIX", BOTH, MIX_NO, _algo, _dist, _deci, decisionDist[thr_i][thr_j]));
				roc.add(decisionDist[thr_i][thr_j],print.getFPR(),print.getTPR());
			}
		roc.print(createFilename("ROC-LATEMIX", BOTH, MIX_NO, _algo, _dist, _deci, 0));
	}
	ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
}

void launch_PERSOMIX(string pathToDB, string pathToDBTest, Mixer _mix, typeAlgo _algo, typeDistance _dist, typeDecision _deci)
{
	clock_t start,ends;
	cout << " ######## Launch_PERSOMIX  ######## " <<endl;

	DBImage *databaseTraining = new DBImage(QString(pathToDB.c_str()),BOTH,_mix);
	databaseTraining->setCibleMix();

	/* ALGO */
		/* Training */
	AlgoPCA* pca=NULL;
	AlgoLDA* lda=NULL;
	AlgoPCALDA* pcalda=NULL;
	//AlgoHISTOGRAM* histo=NULL;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA training... ";
		pca = new AlgoPCA(*databaseTraining, 0.99);
		pca->launch();
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA training... ";
		lda = new AlgoLDA(*databaseTraining);
		lda->launch();
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA training... ";
		pcalda = new AlgoPCALDA(*databaseTraining, 0.99);
		pcalda->launch();
		break;
	//case ALGOHISTO:
	//	histo = new AlgoHISTOGRAM(*databaseTraining);
	////	histo->launch();
	//	break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the training feature vectors */
	vector<vector<cv::Mat>> featureVectorTrainingRGB;
	vector<vector<cv::Mat>> featureVectorTrainingDepth;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (training)... ";
		databaseTraining->setCibleRGB();
		featureVectorTrainingRGB = pca->featureVectOUT(*databaseTraining);
		databaseTraining->setCibleDepth();
		featureVectorTrainingDepth = pca->featureVectOUT(*databaseTraining);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (training)... ";
		databaseTraining->setCibleRGB();
		featureVectorTrainingRGB = lda->featureVectOUT(*databaseTraining);
		databaseTraining->setCibleDepth();
		featureVectorTrainingDepth = lda->featureVectOUT(*databaseTraining);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA feature vectors (training)... ";
		databaseTraining->setCibleRGB();
		featureVectorTrainingRGB = pcalda->featureVectOUT(*databaseTraining);
		databaseTraining->setCibleDepth();
		featureVectorTrainingDepth = pcalda->featureVectOUT(*databaseTraining);
		break;
	//case ALGOHISTO:
	//	start = clock(); cout << "AlgoHISTO feature vectors (training)... ";
	//	databaseTraining->setCibleRGB();
	//	featureVectorTrainingRGB = histo->featureVectOUT(*databaseTraining);
	//	databaseTraining->setCibleDepth();
	//	featureVectorTrainingDepth = histo->featureVectOUT(*databaseTraining);
	//	break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the test feature vectors */
	DBImage *databaseTest = new DBImage(QString(pathToDBTest.c_str()),BOTH,MIX_NO);
	
	vector<vector<cv::Mat>> featureVectorTestRGB;
	vector<vector<cv::Mat>> featureVectorTestDepth;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (testing)... ";
		databaseTest->setCibleRGB();
		featureVectorTestRGB = pca->featureVectOUT(*databaseTest);
		databaseTest->setCibleDepth();
		featureVectorTestDepth = pca->featureVectOUT(*databaseTest);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (testing)... ";
		databaseTest->setCibleRGB();
		featureVectorTestRGB = lda->featureVectOUT(*databaseTest);
		databaseTest->setCibleDepth();
		featureVectorTestDepth = lda->featureVectOUT(*databaseTest);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA feature vectors (testing)... ";
		databaseTest->setCibleRGB();
		featureVectorTestRGB = pcalda->featureVectOUT(*databaseTest);
		databaseTest->setCibleDepth();
		featureVectorTestDepth = pcalda->featureVectOUT(*databaseTest);
		break;
	/*case ALGOHISTO:
		start = clock(); cout << "AlgoHISTO feature vectors (testing)... ";
		featureVectorTest = histo->featureVectOUT(databaseTest);
		break;*/
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Fill the correspondance vector */
	start = clock(); cout << "Correspondance Map... ";
	vector<int> trainingCorrespondance(featureVectorTestRGB.size());
	for(int i=0;i<featureVectorTestRGB.size();i++) trainingCorrespondance[i] = databaseTraining->getIDFromName(databaseTest->getName(i));
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;


	start = clock(); cout << "Freeing (some) db space... ";
	//delete databaseTraining; databaseTraining=0;
	delete databaseTest; databaseTest=0;
	databaseTraining->clearPerson();
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;



	/* DIST */
	vector<vector<vector<vector<double>>>> distances(featureVectorTestRGB.size());
	vector<Mat> mahalanobis_meanRGB, mahalanobis_invcovRGB,mahalanobis_meanDepth, mahalanobis_invcovDepth;
		/* Mahalanobis training */
	if(_dist == MAHALANOBIS)
	{
		start = clock(); cout <<"Mahalanobis training... ";
		mahalanobis_meanRGB.reserve(featureVectorTrainingRGB.size());
		mahalanobis_invcovRGB.reserve(featureVectorTrainingRGB.size());

		for(int i=0;i<featureVectorTrainingRGB.size();i++)
		{
			mahalanobis_meanRGB.push_back(Mat());
			mahalanobis_invcovRGB.push_back(Mat());
			mahalanobis_meanRGB[i] = meanMat<double>(featureVectorTrainingRGB[i]);
			mahalanobis_invcovRGB[i] = (featureCovMat<double>(featureVectorTrainingRGB[i],mahalanobis_meanRGB[i])).inv();
		}

		mahalanobis_meanDepth.reserve(featureVectorTrainingDepth.size());
		mahalanobis_invcovDepth.reserve(featureVectorTrainingDepth.size());

		for(int i=0;i<featureVectorTrainingDepth.size();i++)
		{
			mahalanobis_meanDepth.push_back(Mat());
			mahalanobis_invcovDepth.push_back(Mat());
			mahalanobis_meanDepth[i] = meanMat<double>(featureVectorTrainingDepth[i]);
			mahalanobis_invcovDepth[i] = (featureCovMat<double>(featureVectorTrainingDepth[i],mahalanobis_meanDepth[i])).inv();
		}
		ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
	}
		/* Distances computation */
	start = clock(); cout <<"Distance computation... ";
	//cout << "featureVectorTest.size():"<<featureVectorTest.size()<<endl;
	for(int test_i=0;test_i<featureVectorTestRGB.size();test_i++)
	{
		distances[test_i].resize(featureVectorTestRGB[test_i].size());
		//cout << "featureVectorTest[test_i].size():"<<test_i<<":"<<featureVectorTest[test_i].size()<<endl;
		for(int test_j=0;test_j<featureVectorTestRGB[test_i].size();test_j++)
		{
			distances[test_i][test_j].resize(featureVectorTrainingRGB.size());
			//cout << "featureVectorTraining.size():"<<featureVectorTraining.size()<<endl;
			for(int train_i=0 ; train_i<featureVectorTrainingRGB.size() ; train_i++)
			{
				if(_dist == MAHALANOBIS)
				{
					distances[test_i][test_j][train_i].resize(1);
					Mat diff( mahalanobis_meanRGB[train_i].rows, mahalanobis_meanRGB[train_i].cols, CV_64FC1, Scalar(0));
					diff = featureVectorTestRGB[test_i][test_j] - mahalanobis_meanRGB[train_i];
					distances[test_i][test_j][train_i][0] = norm(diff.t()*(mahalanobis_invcovRGB[train_i]*diff),NORM_L2);
					diff = featureVectorTestDepth[test_i][test_j] - mahalanobis_meanDepth[train_i];
					distances[test_i][test_j][train_i][0] += norm(diff.t()*(mahalanobis_invcovDepth[train_i]*diff),NORM_L2);
				}
				else
				{
					distances[test_i][test_j][train_i].resize(featureVectorTrainingRGB[train_i].size());
					//cout << "featureVectorTraining[train_i].size():"<<train_i<<":"<<featureVectorTraining[train_i].size()<<endl;
					for(int train_j=0 ; train_j<featureVectorTrainingRGB[train_i].size() ; train_j++)
					{
						if(_dist == NORM1)
						{
							distances[test_i][test_j][train_i][train_j] = norm(featureVectorTrainingRGB[train_i][train_j],featureVectorTestRGB[test_i][test_j],NORM_L1);
							distances[test_i][test_j][train_i][train_j] += norm(featureVectorTrainingDepth[train_i][train_j],featureVectorTestDepth[test_i][test_j],NORM_L1);
						}
						else if(_dist == NORM2)
						{
							distances[test_i][test_j][train_i][train_j] = norm(featureVectorTrainingRGB[train_i][train_j],featureVectorTestRGB[test_i][test_j],NORM_L2);
							distances[test_i][test_j][train_i][train_j] += norm(featureVectorTrainingDepth[train_i][train_j],featureVectorTestDepth[test_i][test_j],NORM_L2);
						}
					}
				}
			}
		}
	}

	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Decision */
	start = clock(); cout <<"Decision... ";

	vector<vector<int>> decisionID;
	decisionID.resize(distances.size());

	vector<vector<double>> decisionDist;
	decisionDist.resize(distances.size());

	for(int test_i=0;test_i<distances.size();test_i++)
	{
		decisionID[test_i].resize(distances[test_i].size(),-1);
		decisionDist[test_i].resize(distances[test_i].size(),numeric_limits<double>::max());
		for(int test_j=0;test_j<distances[test_i].size();test_j++)
		{
			if(_deci==THRESHOLD)
			{
				/* Nearest */
				for(int train_i=0 ; train_i<distances[test_i][test_j].size() ; train_i++)
				{
					for(int train_j=0 ; train_j<distances[test_i][test_j][train_i].size() ; train_j++)
					{
						if(decisionDist[test_i][test_j]>distances[test_i][test_j][train_i][train_j])
						{
							decisionID[test_i][test_j]=train_i;
							decisionDist[test_i][test_j]=distances[test_i][test_j][train_i][train_j];
						}
					}
				}
			}
		}
	}
	ends = clock(); cout<<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Count */
	start = clock(); cout<<"Statistics... ";
	if(_deci==THRESHOLD)
	{
		ROC roc;

		/* We test with a maximum as a threshold */
		{
			Print print(*databaseTraining);
			print.newPass();
			for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
				for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
				{
					print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<numeric_limits<double>::max()? decisionID[deci_i][deci_j]:-1) );
				}
			print.print(createFilename("PERSOMIX", BOTH, _mix, _algo, _dist, _deci, numeric_limits<double>::max()));
			roc.add(numeric_limits<double>::max(),print.getFPR(),print.getTPR());
		}
		/* Here each decision correspond to a threhold level */
		for(int thr_i=0;thr_i<decisionDist.size();thr_i++)
			for(int thr_j=0;thr_j<decisionDist[thr_i].size();thr_j++)
			{
				Print print(*databaseTraining);
				print.newPass();
				for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
					for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
					{
						print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<decisionDist[thr_i][thr_j]? decisionID[deci_i][deci_j]:-1) );
					}
				print.print(createFilename("PERSOMIX", BOTH, _mix, _algo, _dist, _deci, decisionDist[thr_i][thr_j]));
				roc.add(decisionDist[thr_i][thr_j],print.getFPR(),print.getTPR());
			}
		roc.print(createFilename("ROC-PERSOMIX", BOTH, _mix, _algo, _dist, _deci, 0));
	}
	ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
}

void launch_EARLYMIX(string pathToDB, string pathToDBTest, Mixer _mix, typeAlgo _algo, typeDistance _dist, typeDecision _deci)
{
	clock_t start,ends;
	cout << " ######## Launch_EARLYMIX  ######## " <<endl;

	DBImage *databaseTraining = new DBImage(QString(pathToDB.c_str()),BOTH,_mix);
	databaseTraining->setCibleMix();

	/* ALGO */
		/* Training */
	AlgoPCA* pca=NULL;
	AlgoLDA* lda=NULL;
	AlgoPCALDA* pcalda=NULL;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA training... ";
		pca = new AlgoPCA(*databaseTraining, 0.99);
		pca->launch();
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA training... ";
		lda = new AlgoLDA(*databaseTraining);
		lda->launch();
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA training... ";
		pcalda = new AlgoPCALDA(*databaseTraining, 0.99);
		pcalda->launch();
		break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the training feature vectors */
	vector<vector<cv::Mat>> featureVectorTraining;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (training)... ";
		featureVectorTraining = pca->featureVectOUT(*databaseTraining);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (training)... ";
		featureVectorTraining = lda->featureVectOUT(*databaseTraining);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoLDA feature vectors (training)... ";
		featureVectorTraining = pcalda->featureVectOUT(*databaseTraining);
		break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Get the test feature vectors */
	DBImage *databaseTest = new DBImage(QString(pathToDBTest.c_str()),BOTH,_mix);
	databaseTest->setCibleMix();
	
	vector<vector<cv::Mat>> featureVectorTest;
	switch(_algo)
	{
	case ALGOPCA:
		start = clock(); cout << "AlgoPCA feature vectors (testing)... ";
		featureVectorTest = pca->featureVectOUT(*databaseTest);
		break;
	case ALGOLDA:
		start = clock(); cout << "AlgoLDA feature vectors (testing)... ";
		featureVectorTest = lda->featureVectOUT(*databaseTest);
		break;
	case ALGOPCALDA:
		start = clock(); cout << "AlgoPCALDA feature vectors (testing)... ";
		featureVectorTest = pcalda->featureVectOUT(*databaseTest);
		break;
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

		/* Fill the correspondance vector */
	start = clock(); cout << "Correspondance Map... ";
	vector<int> trainingCorrespondance(featureVectorTest.size());
	for(int i=0;i<featureVectorTest.size();i++) trainingCorrespondance[i] = databaseTraining->getIDFromName(databaseTest->getName(i));
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;


	start = clock(); cout << "Freeing (some) db space... ";
	//delete databaseTraining; databaseTraining=0;
	delete databaseTest; databaseTest=0;
	databaseTraining->clearPerson();
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;



	/* DIST */
	vector<vector<vector<vector<double>>>> distances(featureVectorTest.size());
	vector<Mat> mahalanobis_mean, mahalanobis_invcov;
		/* Mahalanobis training */
	if(_dist == MAHALANOBIS)
	{
		start = clock(); cout <<"Mahalanobis training... ";
		mahalanobis_mean.reserve(featureVectorTraining.size());
		mahalanobis_invcov.reserve(featureVectorTraining.size());

		for(int i=0;i<featureVectorTraining.size();i++)
		{
			mahalanobis_mean.push_back(Mat());
			mahalanobis_invcov.push_back(Mat());
			mahalanobis_mean[i] = meanMat<double>(featureVectorTraining[i]);
			mahalanobis_invcov[i] = (featureCovMat<double>(featureVectorTraining[i],mahalanobis_mean[i])).inv();
		}
		ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
	}
		/* Distances computation */
	start = clock(); cout <<"Distance computation... ";
	//cout << "featureVectorTest.size():"<<featureVectorTest.size()<<endl;
	for(int test_i=0;test_i<featureVectorTest.size();test_i++)
	{
		distances[test_i].resize(featureVectorTest[test_i].size());
		//cout << "featureVectorTest[test_i].size():"<<test_i<<":"<<featureVectorTest[test_i].size()<<endl;
		for(int test_j=0;test_j<featureVectorTest[test_i].size();test_j++)
		{
			distances[test_i][test_j].resize(featureVectorTraining.size());
			//cout << "featureVectorTraining.size():"<<featureVectorTraining.size()<<endl;
			for(int train_i=0 ; train_i<featureVectorTraining.size() ; train_i++)
			{
				if(_dist == MAHALANOBIS)
				{
					distances[test_i][test_j][train_i].resize(1);
					Mat diff( mahalanobis_mean[train_i].rows, mahalanobis_mean[train_i].cols, CV_64FC1, Scalar(0));
					diff = featureVectorTest[test_i][test_j] - mahalanobis_mean[train_i];
					distances[test_i][test_j][train_i][0] = norm(diff.t()*(mahalanobis_invcov[train_i]*diff),NORM_L2);
				}
				else
				{
					distances[test_i][test_j][train_i].resize(featureVectorTraining[train_i].size());
					//cout << "featureVectorTraining[train_i].size():"<<train_i<<":"<<featureVectorTraining[train_i].size()<<endl;
					for(int train_j=0 ; train_j<featureVectorTraining[train_i].size() ; train_j++)
					{
						if(_dist == NORM1)
							distances[test_i][test_j][train_i][train_j] = norm(featureVectorTraining[train_i][train_j],featureVectorTest[test_i][test_j],NORM_L1);
						else if(_dist == NORM2)
							distances[test_i][test_j][train_i][train_j] = norm(featureVectorTraining[train_i][train_j],featureVectorTest[test_i][test_j],NORM_L2);
					}
				}
			}
		}
	}
	ends = clock(); cout << "DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Decision */
	start = clock(); cout <<"Decision... ";

	vector<vector<int>> decisionID;
	decisionID.resize(distances.size());

	vector<vector<double>> decisionDist;
	decisionDist.resize(distances.size());

	for(int test_i=0;test_i<distances.size();test_i++)
	{
		decisionID[test_i].resize(distances[test_i].size(),-1);
		decisionDist[test_i].resize(distances[test_i].size(),numeric_limits<double>::max());
		for(int test_j=0;test_j<distances[test_i].size();test_j++)
		{
			if(_deci==THRESHOLD)
			{
				/* Nearest */
				for(int train_i=0 ; train_i<distances[test_i][test_j].size() ; train_i++)
				{
					for(int train_j=0 ; train_j<distances[test_i][test_j][train_i].size() ; train_j++)
					{
						if(decisionDist[test_i][test_j]>distances[test_i][test_j][train_i][train_j])
						{
							decisionID[test_i][test_j]=train_i;
							decisionDist[test_i][test_j]=distances[test_i][test_j][train_i][train_j];
						}
					}
				}
			}
		}
	}
	ends = clock(); cout<<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;

	/* Count */
	start = clock(); cout<<"Statistics... ";
	if(_deci==THRESHOLD)
	{
		ROC roc;

		/* We test with a maximum as a threshold */
		{
			Print print(*databaseTraining);
			print.newPass();
			for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
				for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
				{
					print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<numeric_limits<double>::max()? decisionID[deci_i][deci_j]:-1) );
				}
			print.print(createFilename("EARLYMIX", BOTH, _mix, _algo, _dist, _deci, numeric_limits<double>::max()));
			roc.add(numeric_limits<double>::max(),print.getFPR(),print.getTPR());
		}
		/* Here each decision correspond to a threhold level */
		for(int thr_i=0;thr_i<decisionDist.size();thr_i++)
			for(int thr_j=0;thr_j<decisionDist[thr_i].size();thr_j++)
			{
				Print print(*databaseTraining);
				print.newPass();
				for(int deci_i=0;deci_i<decisionDist.size();deci_i++)
					for(int deci_j=0;deci_j<decisionDist[deci_i].size();deci_j++)
					{
						print.add(trainingCorrespondance[deci_i], (decisionDist[deci_i][deci_j]<decisionDist[thr_i][thr_j]? decisionID[deci_i][deci_j]:-1) );
					}
				print.print(createFilename("EARLYMIX", BOTH, _mix, _algo, _dist, _deci, decisionDist[thr_i][thr_j]));
				roc.add(decisionDist[thr_i][thr_j],print.getFPR(),print.getTPR());
			}
		roc.print(createFilename("ROC-EARLYMIX", BOTH, _mix, _algo, _dist, _deci, 0));
	}
	ends = clock(); cout <<"DONE ("<< double(ends-start)/double(CLOCKS_PER_SEC)<<"s)"<<endl;
}

KinectBasedFaceRecVal::KinectBasedFaceRecVal()
{
	const string trainstr = "..\\db\\tests\\texas3\\training";
	const string teststr = "..\\db\\tests\\texas3\\test-probe";

	/*launch_NOMIX(trainstr,teststr,RGB,ALGOPCA,NORM1,THRESHOLD);
	launch_NOMIX(trainstr,teststr,RGB,ALGOPCA,NORM2,THRESHOLD);
	launch_NOMIX(trainstr,teststr,RGB,ALGOPCA,MAHALANOBIS,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOPCA,NORM1,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOPCA,NORM2,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOPCA,MAHALANOBIS,THRESHOLD);*/
	/*launch_NOMIX(trainstr,teststr,RGB,ALGOLDA,NORM1,THRESHOLD);
	launch_NOMIX(trainstr,teststr,RGB,ALGOLDA,NORM2,THRESHOLD);
	launch_NOMIX(trainstr,teststr,RGB,ALGOLDA,MAHALANOBIS,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOLDA,NORM1,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOLDA,NORM2,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOLDA,MAHALANOBIS,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOHISTO,NORM1,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOHISTO,NORM2,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOHISTO,MAHALANOBIS,THRESHOLD);*/
	/*launch_NOMIX(trainstr,teststr,RGB,ALGOPCALDA,NORM1,THRESHOLD);
	launch_NOMIX(trainstr,teststr,RGB,ALGOPCALDA,NORM2,THRESHOLD);
	launch_NOMIX(trainstr,teststr,RGB,ALGOPCALDA,MAHALANOBIS,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOPCALDA,NORM1,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOPCALDA,NORM2,THRESHOLD);
	launch_NOMIX(trainstr,teststr,DEPTH,ALGOPCALDA,MAHALANOBIS,THRESHOLD);*/

	/*launch_LATEMIX(trainstr,teststr,ALGOPCA,NORM1,THRESHOLD);
	launch_LATEMIX(trainstr,teststr,ALGOPCA,NORM2,THRESHOLD);
	launch_LATEMIX(trainstr,teststr,ALGOPCA,MAHALANOBIS,THRESHOLD);
	launch_LATEMIX(trainstr,teststr,ALGOLDA,NORM1,THRESHOLD);
	launch_LATEMIX(trainstr,teststr,ALGOLDA,NORM2,THRESHOLD);
	launch_LATEMIX(trainstr,teststr,ALGOLDA,MAHALANOBIS,THRESHOLD);
	launch_LATEMIX(trainstr,teststr,ALGOPCALDA,NORM1,THRESHOLD);
	launch_LATEMIX(trainstr,teststr,ALGOPCALDA,NORM2,THRESHOLD);*/
	launch_LATEMIX(trainstr,teststr,ALGOPCALDA,MAHALANOBIS,THRESHOLD);

	/*launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOPCA,NORM1,THRESHOLD);
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOPCA,NORM2,THRESHOLD);
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOPCA,MAHALANOBIS,THRESHOLD);
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOLDA,NORM1,THRESHOLD);
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOLDA,NORM2,THRESHOLD);
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOLDA,MAHALANOBIS,THRESHOLD);
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOPCALDA,NORM1,THRESHOLD);
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOPCALDA,NORM2,THRESHOLD);*/
	launch_PERSOMIX(trainstr,teststr,MIX_SUM,ALGOPCALDA,MAHALANOBIS,THRESHOLD);

	/*launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOPCA,NORM1,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOPCA,NORM2,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOPCA,MAHALANOBIS,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOPCA,NORM1,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOPCA,NORM2,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOPCA,MAHALANOBIS,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOLDA,NORM1,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOLDA,NORM2,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOLDA,MAHALANOBIS,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOLDA,NORM1,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOLDA,NORM2,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOLDA,MAHALANOBIS,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOPCALDA,NORM1,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOPCALDA,NORM2,THRESHOLD);*/
	launch_EARLYMIX(trainstr,teststr,MIX_SUM,ALGOPCALDA,MAHALANOBIS,THRESHOLD);
	/*launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOPCALDA,NORM1,THRESHOLD);
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOPCALDA,NORM2,THRESHOLD);*/
	launch_EARLYMIX(trainstr,teststr,MIX_CONCAT,ALGOPCALDA,MAHALANOBIS,THRESHOLD);

	system("pause");
	exit(0);
}