#pragma once
#include "HoughList.h"


HoughListCpp::HoughListCpp()
{
	hash_table = std::unordered_map<index4D, vector<DMatch>, function<size_t(const index4D)> > { 100, hash4dd };
}
#include <array>
#include <functional>
typedef std::array<int, 4> index4D;

int HoughListCpp::AddMatch(DMatch match, int xbin, int ybin, int orbin, int scalebin)
{
	index4D key = { xbin, ybin, orbin, scalebin };
	hash_table[key].push_back(match);
	return 1;
}
size_t hash4dd(const index4D cell)
{
	return hash<int>()(cell[0] ^ hash<int>()(cell[1]) ^ hash<int>()(cell[2]) ^ hash<int>()(cell[3]));
}
