#pragma once
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;

#include <unordered_map>
#include <array>
#include <functional>

typedef std::array<int, 4> index4D;
size_t hash4dd(const index4D cell);

class HoughListCpp
{
public:
	HoughListCpp();
	std::unordered_map<index4D, vector<DMatch>, function<size_t(const index4D)> > hash_table;
	int AddMatch(DMatch match, int xbin, int ybin, int orbin, int scalebin);
};
