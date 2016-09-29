#include <iostream>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <utility>
#include <queue>
#include <fstream>
#include <omp.h>
#include <cmath>

#include "kdtree.h"

using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 3) {
		cerr << "missing required parameters." << endl;
		return 0;
	}

	int k;
	int n;
	
	sscanf(argv[1], "%d", &k);
	sscanf(argv[2], "%d", &n);

#ifdef DEBUG
	cerr << "dimemsion:         " << k << "\n";
	cerr << "number of points:  " << n << endl; 
#endif

	vector<vector<float> > pointSet;

	srand((unsigned)time(NULL));
	for(int i = 0; i < n; i++) {
		vector<float> vec;
		for(int j = 0; j < k; j++) {
			vec.push_back(static_cast<float>(rand()) / RAND_MAX);
		}
		pointSet.push_back(vec);
	}

#ifdef DEBUG
//	for(int i = 0; i < n; i++) {
//		for(int j = 0; j < k; j++) {
//			cerr << pointSet[i][j] << " ";
//		}
//		cerr << "\n";
//	}
#endif

	kdtree tree(k, n);

	tree.TreeConstructor(pointSet);
	
//	tree.Print();	

	vector<float> testVec;

	sleep(10);
	srand((unsigned)time(NULL));
	for(int i = 0; i < k; i++) {
		testVec.push_back(static_cast<float>(rand()) / RAND_MAX);
	}	

#ifdef DEBUG
	for(int i = 0; i < k; i++) {
		cerr << testVec[i] << " ";
	}
	cerr << endl;
#endif

	double t0 = omp_get_wtime();
	pair<float, int> dist = tree.NearestNeighbour(pointSet, testVec);
	t0 = omp_get_wtime() - t0;

	cout << "Nearest Neighbour Search of K-d Tree:\n";
	cout << "Nearest distance: " << dist.first << "\n";
	cout << "Nearest point:    " << dist.second << "\n";
	cout << "Elasped time:     " << t0 << "s." << endl; 

	float minDist = 1e5;
	int minPt = 0;

	t0 = omp_get_wtime();
	for(int i = 0; i < n; i++) {
		float dst = Distance(pointSet[i], testVec);
		if(dst < minDist) {
			minDist = dst;
			minPt = i;
		}
	}
	t0 = omp_get_wtime() - t0;	

	cout << "Naive Nearest Neighbour Search:\n";
	cout << "Nearest distance: " << dist.first << "\n";
	cout << "Nearest point:    " << dist.second << "\n";
	cout << "Elasped time:     " << t0 << "s." << endl; 

	int neighbors = 10;
	t0 = omp_get_wtime();
	KNeighbourSet knn = tree.KNearestNeighbour(pointSet, testVec, neighbors);
	t0 = omp_get_wtime() - t0;
	
	vector<pair<float, int> > *vknn = (vector<pair<float, int> >*)&knn;
	
	cout << neighbors << " Nearest Neighbour Search of K-d Tree:\n";
	for(vector<pair<float, int> >::iterator ite = vknn->begin(); ite != vknn->end(); ite++) {
		int idx = static_cast<int>(ite - vknn->begin());
		cout << idx << " Nearest distance: " << ite->first << "\n";
		cout << idx << " Nearest point:    " << ite->second << "\n";
	}
	cout << "Elasped time:     " << t0 << "s." << endl; 

	vector<pair<float, int> > tmp(neighbors, pair<float, int>(MAX_FLOAT_NUM, 0));
	KNeighbourSet _knn(tmp.begin(), tmp.end());
	tmp.swap(vector<pair<float, int> >());

	t0 = omp_get_wtime();
	for(size_t i = 0; i < pointSet.size(); i++) {
		float maxDst = (_knn.top()).first;
		float dist = Distance(pointSet[i], testVec);
		if(dist < maxDst) {
			_knn.pop();
			_knn.push(pair<float, int>(dist, i));
		}	
	}
	t0 = omp_get_wtime() - t0;
	
	vector<pair<float, int> >* _vknn = (vector<pair<float, int> >*)&_knn;
	
	cout << "Naive " << neighbors << " Nearest Neighbour Search of K-d Tree:\n";
	for(vector<pair<float, int> >::iterator ite = _vknn->begin(); ite != _vknn->end(); ite++) {
		int idx = static_cast<int>(ite - _vknn->begin());
		cout << idx << " Nearest distance: " << ite->first << "\n";
		cout << idx << " Nearest point:    " << ite->second << "\n";
	}
	cout << "Elasped time:     " << t0 << "s." << endl; 

	return 0; 
}
