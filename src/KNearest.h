#ifndef KNEAREST_H
#define KNEAREST_H

#include <iostream>
#include <vector>
#include <limits>

class KNearest
{
private:
	int k;
	std::vector<double> values;
	std::vector<int> id;

	template<typename _Tp> int findVectorIDMax(std::vector<_Tp> valToCheck);

public:
	KNearest(int setK);
	void addValue(double val, int idVal);
	int absoluteDecision();
	int majorityDecision();
};

#endif
