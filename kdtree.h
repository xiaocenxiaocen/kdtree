#ifndef KDTREE_H
#define KDTREE_H

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define SQ(x) ((x) * (x))
#define MAX_FLOAT_NUM 1e10
#define None -1

using namespace std;

enum NodeType{SUB_TREE, LEAF};

const int NUM_THREADS = 12;

struct cmp {
	bool operator()(const pair<float, int>& _Left, const pair<float, int>& _Right) {
		return _Left.first < _Right.first;
	}
};

typedef priority_queue<pair<float, int>, vector<pair<float, int> >, cmp> KNeighbourSet;

template<class _Ty1, class _Ty2>
inline bool operator<(const std::pair<float,int>& _Left,
					  const std::pair<float,int>& _Right)
{
	return _Left.first < _Right.first;
}

inline static float Distance(const vector<float>& _pt1, const vector<float>& _pt2)
{
//	int nd = _pt1.size();
//	if(_pt1.size() != _pt2.size()) cerr << "pt1's dimemsion mismatches with pt2.\n" << endl;
//	
//	int nd_aligned4 = ((nd + 3)>>2)<<2;
//
//#ifdef DEBUG
//	cerr << nd << " " << nd_aligned4 << "\n";
//#endif
//
//	vector<float> pt1(nd_aligned4, 0.0f);
//	vector<float> pt2(nd_aligned4, 0.0f);
//	
//	pt1.assign(_pt1.begin(), _pt1.end());
//	pt2.assign(_pt2.begin(), _pt2.end());
//
//#ifdef DEBUG
//	for(int i = 0; i < nd_aligned4; i++) {
//		cerr << pt1[i] << " ";
//	}
//	cerr << endl;
//	for(int i = 0; i < nd_aligned4; i++) {
//		cerr << pt2[i] << " ";
//	}
//	cerr << endl;
//#endif
//
//	float distance = 0.0f;
//	#pragma omp parallel for \
//		num_threads(NUM_THREADS) \
//		schedule(static, 1) \
//		shared(pt1, pt2) \
//		reduction(+:distance)
//	for(int i = 0; i < nd_aligned4; i += 4) {
//		float dist[4];
//		__m128 _pt1_,_pt2_;
//		_pt1_ = _mm_loadu_ps(&pt1[i]);
//		_pt2_ = _mm_loadu_ps(&pt2[i]);
//		_pt1_ = _mm_sub_ps(_pt1_, _pt2_);
//		_pt1_ = _mm_mul_ps(_pt1_, _pt1_);
//		_mm_storeu_ps(dist, _pt1_);
//		distance += dist[0] + dist[1] + dist[2] + dist[3];
//	}

//	vector<float> sub(_pt1);

	float distance = 0.0;

	for(size_t i = 0; i < _pt1.size(); i++) {
		distance += (_pt1[i] - _pt2[i]) * (_pt1[i] - _pt2[i]);
	}
//	float distance = inner_product(sub.begin(), sub.end(), sub.begin(), 0.0f);	

	return distance;
}

class node;
class kdtree;

class node {
public:
	node():parent(None), left(None), right(None) {};
	~node() { };
private:
	friend class kdtree;
	int splitDimensionIndex;
	int splitPointIndex;
	vector<int> pointsIndex;
	enum NodeType nodeType;
	int depth;
	int parent;
	int left;
	int right;
};

class kdtree {
public:
	kdtree(const int _dims, const int max_depth, const int min_points_per_leaf_node = 1): 
		dims(_dims), maxDepth(max_depth), minItemsPerLeafNode(min_points_per_leaf_node) {} ;
	void Insert(const vector<float>& point);
	void Delete(const vector<float>& point);
	void TreeConstructor(const vector<vector<float> >& pointSet);
	pair<float, int> NearestNeighbour(const vector<vector<float> >& pointSet, const vector<float>& pt, const int nodeEntry = 0);
	KNeighbourSet KNearestNeighbour(const vector<vector<float> >& pointSet, const vector<float>& pt, const int nearestNeighbours);
	void Print();

private:	
	/*
 	 * the dimension of the point
 	 */	
	int dims;
	/*
 	 * maximum depth of the k-dimensional search tree
 	 * the constructing algorithm will build the k-dimensional tree level by
 	 * level, until the maximum depth is exceeded.
 	 */ 
	int maxDepth;
	
	/*
 	 *  minimum items in a leaf node
 	 * if number of points is less than the threshold, 
 	 * the hyperspace won't be split and the node will be a leaf node.
 	 * the default threshold is 2.
 	 */
	int minItemsPerLeafNode;
	
	/* 
 	 * the k-dimensional tree is stored in a stl vector
 	 */			
	vector<node> kdt;
		
	void ComputeColVar(const vector<vector<float> >& pointSet, const vector<int>& sampleDataIndex, vector<float>& colVar);

	int ComputeColMedian(const vector<vector<float> >& pointSet, const vector<int>& sampleDataIndex, const int splitDimensionIndex);

	void PrintTree(const int nodeEntry, const string prefix);
};

#endif
