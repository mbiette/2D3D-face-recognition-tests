#ifndef ENUMS_H
#define ENUMS_H

//enum typeDB{
//	ATT,
//	TEXAS,
//	TEXAS2,
//	TEXAS3
//};

enum Mixer{
	MIX_NO,
	MIX_SUM,
	MIX_CONCAT,
};

//enum typeDBVal{
//	NOVAL,
//	MFOLD
//};

//enum typeSet{
//	NONE,
//	TRAIN,
//	TESTGALLERY,
//	TESTPROBE
//};

enum typeAlgo{
	ALGOPCA,
	ALGOLDA,
	ALGOHISTO,
	ALGOPCALDA
};

enum typeLoading{
	RGB,
	DEPTH,
	BOTH
};

enum typeDistance{
	NORM1,
	NORM2,
	MAHALANOBIS
};

enum typeDecision{
	THRESHOLD,
	CONFIDANCE,
	KNEAREST
};

#endif