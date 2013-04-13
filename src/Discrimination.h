#ifndef DISCRIMINATION_H
#define DISCRIMINATION_H

#include "DBImage2.h"
//#include "AlgoPCA.h"
//#include "AlgoLDA.h"
//#include "AlgoHISTOGRAM.h"
#include "Enums.h"
#include <QString>
#include <iostream>
#include <fstream> 

class Print
{
private:
	std::vector<std::vector<int>> confusion;
	std::vector<int> confusionNbSamples;
	/*std::vector<int> confusionSumSamples;*/
	std::vector<int> confusionRecognized;
	std::vector<int> confusionErrors;
	std::vector<int> confusionIncorrectRecognized;
	std::vector<int> confusionNotRecognized;
	int nbPass;
	int sumRecognized, sumErrors, sumIncorrectRecognized, sumNotRecognized;
	int sumTruePositiv, sumFalseNegativ, sumFalsePositiv, sumTrueNegativ; 

	int nbFaces, nbSamplesDetected, nbSamples;
	vector<string> nameFaces;
public:
	Print(DBImage &db);
	void newPass();
	void add(int idWanted, int idFound);
	void print(string filename);
	double getFPR();
	double getTPR();
};

//class Confirm
//{
//private:
//	Print *print;
//	//std::vector<Face> faces;
//	DBImage& db;
//	typeLoading img;
//	typeDistance dist;
//	typeDecision classif;
//	//typeDBVal val;
//	double param;
//public:
//	Confirm(DBImage &db, Print *toPrint, typeLoading image, typeDistance distance, typeDecision decision, double parameter, typeDBVal val);
//	void exec(AlgoPCA &pca);
//	void exec(AlgoLDA &lda);
//	void exec(AlgoHISTOGRAM &histo);
//};

class ROC
{
private:
	std::vector<double> threshold;
	std::vector<double> falsePositivRate;
	std::vector<double> truePositivRate;
public:
	ROC();
	void add(double threshold, double fPR, double tPR);
	void print(std::string filename);
};

#endif
