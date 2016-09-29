#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <utility>
#include <queue>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <cmath>

#include "kdtree.h"

void kdtree::Insert(const vector<float>& point)
{
	
}

void kdtree::Delete(const vector<float>& point)
{
	
}

/*
 * Compute the variance of each column of the point set
 * @param pointSet:			k-dimensional point set
 * @param sampleDataIndex:	point index of current tree node
 * @retval colVar:			variance of columns of current tree node
 */
void kdtree::ComputeColVar(const vector<vector<float> >& pointSet, const vector<int>& sampleDataIndex, vector<float>& colVar)
{
	colVar.resize(dims);
	int nSamples = sampleDataIndex.size();
	#pragma omp parallel for \
		num_threads(NUM_THREADS) \
		schedule(dynamic) \
		shared(pointSet, sampleDataIndex, colVar)
	for(int col = 0; col < dims; col++) {
		colVar[col] = 0.0;
		float colSum = 0.0;
		for(int i = 0; i < nSamples; i++) {
			colVar[col] += SQ(pointSet[sampleDataIndex[i]][col]);
			colSum += pointSet[sampleDataIndex[i]][col];
		}
		colVar[col] -= SQ(colSum) / nSamples;
		colVar[col] /= nSamples;
	}
	
	return;
}

/*
 * Get the point with median value in the select column
 * @param pointSet:				k-dimensional point set
 * @param sampleDataIndex:		point index of current tree node
 * @param splitDimensionIndex: 	select column
 * @retval int:					median point index
 */
int kdtree::ComputeColMedian(const vector<vector<float> >& pointSet, const vector<int>& sampleDataIndex, const int splitDimensionIndex)
{
	vector<pair<float, int> > medArray;

	int nSamples = sampleDataIndex.size();

	for(int i = 0; i < nSamples; i++) {
		medArray.push_back(pair<float, int>(pointSet[sampleDataIndex[i]][splitDimensionIndex], sampleDataIndex[i]));
	}

	int median = nSamples / 2;

	nth_element(medArray.begin(), medArray.begin() + median, medArray.end());

	return (medArray.begin() + median)->second;
}
	
/* 
 * A level wise k-dimensional tree constructor
 * @param pointSet: k-dimensional points set
 */
void kdtree::TreeConstructor(const vector<vector<float> >& pointSet)
{
	if(pointSet.empty()) return;

	dims = pointSet[0].size();

	int nPoints = pointSet.size();

	int levelFirst = 0;
	int levelLast = 1;	

	kdt.push_back(node());

	for(int i = 0; i < nPoints; i++) {
		kdt[0].pointsIndex.push_back(i);
	} // end for

	kdt[0].parent = -1;
	kdt[0].depth = 0;
	kdt[0].nodeType = SUB_TREE;

	for(int level = 0; level < maxDepth; level++) {
		int numLevelPoints = 0;
		for(int item = levelFirst; item < levelLast; item++) {
			vector<float> colVar;
			ComputeColVar(pointSet, kdt[item].pointsIndex, colVar);
			vector<float>::const_iterator ite = max_element(colVar.begin(), colVar.end());
			int splDimIdx = kdt[item].splitDimensionIndex = static_cast<int>(ite - colVar.begin());
			int splPtIdx = kdt[item].splitPointIndex = ComputeColMedian(pointSet, kdt[item].pointsIndex, kdt[item].splitDimensionIndex);
			if(static_cast<int>(kdt[item].pointsIndex.size()) <= minItemsPerLeafNode) {
				kdt[item].nodeType = LEAF;
				continue;
			} 			

			vector<int> pointsIndexLeft;
			vector<int> pointsIndexRight;

		//	int splPtIdx = kdt[item].splitPointIndex;
		//	int splDimIdx = kdt[item].splitDimensionIndex;

	//		for(size_t i = 0; i < kdt[item].pointsIndex.size(); i++) {
	//			cout << kdt[item].pointsIndex[i] << " ";
	//		}
	//		cout << "\n";

	//		cout << splPtIdx << "\n";
	//		cout << splDimIdx << "\n";
			for(size_t i = 0; i < kdt[item].pointsIndex.size(); i++) {
				int ptIdx = kdt[item].pointsIndex[i];
				if(pointSet[ptIdx][splDimIdx] <= pointSet[splPtIdx][splDimIdx] && ptIdx != splPtIdx) {
					pointsIndexLeft.push_back(ptIdx);
				} else if(pointSet[ptIdx][splDimIdx] > pointSet[splPtIdx][splDimIdx] && ptIdx != splPtIdx) {
					pointsIndexRight.push_back(ptIdx);
				}
			} // end for
	
		//	cout << pointsIndexLeft.size() << " " << pointsIndexRight.size() << " " << kdt[item].pointsIndex.size() << "\n";
			kdt[item].pointsIndex.swap(vector<int>());				

			if(pointsIndexLeft.size() > 0) {
				node leftChild = node();
				leftChild.parent = item;
				leftChild.nodeType = SUB_TREE;
				leftChild.depth = level + 1;
				leftChild.pointsIndex = pointsIndexLeft;
				kdt.push_back(leftChild);
				numLevelPoints += 1;
				kdt[item].left = kdt.size() - 1;
			}
			
			if(pointsIndexRight.size() > 0) {	
				node rightChild = node();
				rightChild.parent = item;
				rightChild.nodeType = SUB_TREE;
				rightChild.depth = level + 1;
				rightChild.pointsIndex = pointsIndexRight;
				kdt.push_back(rightChild);
				numLevelPoints += 1;
				kdt[item].right = kdt.size() - 1;
			}
			
		} // end for
		levelFirst = levelLast;
		levelLast = levelFirst + numLevelPoints;
		if(levelLast - levelFirst < 1) break;
	} // end for	
}

/*
 * A nearest neighbour query implementation 
 * @param pointSet: 			k-dimensional point set
 * @param pt: 					query point
 * @param nodeEntry:  			entry node index
 * @retval pair<float, int>:	ret.first = nearest distance
 * 								ret.second = nearest point's index in pointSet
 */
pair<float, int> kdtree::NearestNeighbour(const vector<vector<float> >& pointSet, const vector<float>& pt, const int nodeEntry)
{
	if(kdt[nodeEntry].nodeType == LEAF) {
		pair<float, int> nearest(MAX_FLOAT_NUM, 0);
		for(size_t i = 0; i < kdt[nodeEntry].pointsIndex.size(); i++) {
			int idx = kdt[nodeEntry].pointsIndex[i];
			float dist = Distance(pt, pointSet[idx]);
			if(dist < nearest.first) {
				nearest.first = dist;
				nearest.second = idx;
			} // end if
		} // end for
		return nearest;
	}

	int splDimIdx = kdt[nodeEntry].splitDimensionIndex;
	int splPtIdx = kdt[nodeEntry].splitPointIndex;
	int nearerEntry, furtherEntry;
	if(pt[splDimIdx] <= pointSet[splPtIdx][splDimIdx]) {
		nearerEntry = kdt[nodeEntry].left;
		furtherEntry = kdt[nodeEntry].right;
	} else {
		nearerEntry = kdt[nodeEntry].right;
		furtherEntry = kdt[nodeEntry].left;
	}

	pair<float, int> nearer(MAX_FLOAT_NUM, 0);
	pair<float, int> further(MAX_FLOAT_NUM, 0);
	pair<float, int> nearest;
	
	if(nearerEntry != None) nearer = NearestNeighbour(pointSet, pt, nearerEntry);	

	float dist = Distance(pt, pointSet[splPtIdx]);
	nearest = pair<float, int>(dist, splPtIdx);

	if(nearer.first < nearest.first) {
		nearest = nearer;
	}

	if(nearerEntry == None || nearer.first <= SQ((pointSet[splPtIdx][splDimIdx] - pt[splDimIdx]))) {
		return nearest;
	} else {
		if(furtherEntry != None) further = NearestNeighbour(pointSet, pt, furtherEntry);
		
		if(further.first < nearest.first) {
			nearest = further;
		}
	}
	return nearest;
}

/*
 * A implementation of K nearest neighbours search of k-dimensional tree.
 * @param pointSet:		 		k-dimensional points
 * @param pt:				 	the query point
 * @param nearestNeighbours: 	number of neighbours to search
 * @retval KNeighbourSet:		K pairs, first is the square distance, 
 * 								second is the index of corresponding point.
 */
KNeighbourSet kdtree::KNearestNeighbour(const vector<vector<float> >& pointSet, const vector<float>& pt, const int nearestNeighbours)
{
	vector<int> nodeStack;
	
	vector<pair<float, int> > tmp(nearestNeighbours, pair<float, int>(MAX_FLOAT_NUM, 0));

//	priority_queue<pair<float, int>, vector<pair<float, int> >, cmp> knn(tmp.begin(), tmp.end());
	KNeighbourSet knn(tmp.begin(), tmp.end());

	tmp.swap(vector<pair<float, int> >());

	bool reverse = false;

	nodeStack.push_back(0);

	while(!nodeStack.empty()) {
		int curEntry = *(nodeStack.end() - 1);
		nodeStack.pop_back();
		float maxDist = knn.top().first;

		if(!reverse) {
			if(kdt[curEntry].nodeType == LEAF) {
				for(size_t i = 0; i < kdt[curEntry].pointsIndex.size(); i++) {
					int ptIdx = kdt[curEntry].pointsIndex[i];
					float dist = Distance(pt, pointSet[ptIdx]);
					if(dist < maxDist) {
						knn.pop();
						knn.push(pair<float, int>(dist, ptIdx));
					} 
				} // end for
				reverse = true;
			} else {
				int splPtIdx = kdt[curEntry].splitPointIndex;
				int splDimIdx = kdt[curEntry].splitDimensionIndex;
				int nearerEntry, furtherEntry;
				if(pt[splDimIdx] <= pointSet[splPtIdx][splDimIdx]) {
					nearerEntry = kdt[curEntry].left;
					furtherEntry = kdt[curEntry].right;
				} else {
					nearerEntry = kdt[curEntry].right;
					furtherEntry = kdt[curEntry].left;
				} // end if
				float dist = Distance(pt, pointSet[splPtIdx]);
				if(dist < maxDist) {
					knn.pop();
					knn.push(pair<float, int>(dist, splPtIdx));
					maxDist = (knn.top()).first;
				}
				if(nearerEntry == None) {
					if(SQ((pointSet[splPtIdx][splDimIdx] - pt[splDimIdx])) < maxDist && furtherEntry != None) {
						nodeStack.push_back(furtherEntry);
					} else {
						reverse = true;
					}
				} else {
					if(furtherEntry != None) nodeStack.push_back(furtherEntry);
					nodeStack.push_back(nearerEntry);
				}
			}
		} else {
			int parentIdx = kdt[curEntry].parent;
			int splPtIdx = kdt[parentIdx].splitPointIndex;
			int splDimIdx = kdt[parentIdx].splitDimensionIndex;
		//	float dist = Distance(pt, pointSet[splPtIdx]);
		//	if(dist < maxDist) {
		//		knn.pop();
		//		knn.push(pair<float, int>(dist, splPtIdx));
		//	}
			if(SQ((pointSet[splPtIdx][splDimIdx] - pt[splDimIdx])) < maxDist) {	
				int splPtIdx = kdt[curEntry].splitPointIndex;
				int splDimIdx = kdt[curEntry].splitDimensionIndex;
				int nearerEntry, furtherEntry;
				if(pt[splDimIdx] <= pointSet[splPtIdx][splDimIdx]) {
					nearerEntry = kdt[curEntry].left;
					furtherEntry = kdt[curEntry].right;
				} else {
					nearerEntry = kdt[curEntry].right;
					furtherEntry = kdt[curEntry].left;
				} // end if
				float dist = Distance(pt, pointSet[splPtIdx]);
				if(dist < maxDist) {
					knn.pop();
					knn.push(pair<float, int>(dist, splPtIdx));
					maxDist = (knn.top()).first;
				}
				if(nearerEntry == None) {
					if(SQ((pointSet[splPtIdx][splDimIdx] - pt[splDimIdx])) < maxDist && furtherEntry != None) {
						nodeStack.push_back(furtherEntry);
					} else {
						reverse = true;
					}
				} else {
					if(furtherEntry != None) nodeStack.push_back(furtherEntry);
					nodeStack.push_back(nearerEntry);
					reverse = false;
				}
		//		if(furtherEntry != None) nodeStack.push_back(furtherEntry);
		//		if(nearerEntry != None) nodeStack.push_back(nearerEntry);
		//		reverse = false;
			} 
		}
	}
	return knn;
}

void kdtree::PrintTree(const int nodeEntry, string prefix) 
{
	if(kdt[nodeEntry].nodeType == SUB_TREE) {
		cout << prefix << "split dimention: " << kdt[nodeEntry].splitDimensionIndex << ",";
		cout << 		  "split point: " << kdt[nodeEntry].splitPointIndex << endl;
		if(kdt[nodeEntry].left != None) PrintTree(kdt[nodeEntry].left, prefix + "|");
		if(kdt[nodeEntry].right != None) PrintTree(kdt[nodeEntry].right, prefix + "|");
	} else {
		if(kdt[nodeEntry].nodeType == LEAF) {
			cout << prefix << "points index: ";
			for(size_t i = 0; i < kdt[nodeEntry].pointsIndex.size() - 1; i++) {
				cout << kdt[nodeEntry].pointsIndex[i] << ", ";
			}
			cout << *(kdt[nodeEntry].pointsIndex.end() - 1) << endl;
		}	
	}
	
	return;
}

void kdtree::Print()
{
	PrintTree(0, string("")); 
	return;
}
