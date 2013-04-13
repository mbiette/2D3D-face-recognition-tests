#include "KNearest.h"

using namespace std;

template<typename _Tp> int KNearest::findVectorIDMax(vector<_Tp> valToCheck)
{
	_Tp max=valToCheck[0];
	int index=0;
	for (int i=1;i<valToCheck.size();i++)
	{
		if(valToCheck[i]>max)
		{
			max=valToCheck[i];
			index=i;
		}
	}
	return index;
}



KNearest::KNearest(int setK): k(setK)
{
	values.resize(setK, numeric_limits<double>::max());
	id.resize(setK, -1);
}

void KNearest::addValue(double val, int idVal)
{
	int maxPosition = findVectorIDMax<double>(values);
	if(values[maxPosition]>val)
	{
		values[maxPosition]=val;
		id[maxPosition]=idVal;
	}
}

int KNearest::absoluteDecision()
{
	int idVal = id[0];
	for(int i=1; i<k; i++)
	{
		if(idVal != id[i])
		{
			idVal=-1;
			break;
		}
	}
	return idVal;
}

int KNearest::majorityDecision()
{
	vector<int> count(1,1);
	vector<int> idVal(1,id[0]);

	for(int i=1; i<k; i++) //Counting algorithm
	{
		bool flag=false; //Flag if update was made
		for(int j=0; j<idVal.size(); j++)
		{
			if(idVal[j]==id[i]) //
			{
				count[j]++;
				flag=true;
				break;
			}
		}
		if(flag==false) //No update so it's the first time the value occured. We add it.
		{
			count.push_back(1);
			idVal.push_back(id[i]);
		}
	}

	//We look for the value that is repeated the most.
	int maxId = findVectorIDMax<int>(count);

	if(count[maxId]>(k/2)) return idVal[maxId]; //if k is even count as to be at least
	else return -1;
}