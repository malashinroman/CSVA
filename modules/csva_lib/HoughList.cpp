/*
Copyright (C) 2014  Roman Malashin
Copyright (C) 2018  Roman Malashin

All rights reserved.

This is the Author's implementation of CSVA: "Core" structural verification algorithm [1]. There are few differences with the paper. 

[1] Malashin R.O. Core algorithm for structural verification of keypoint matches. Intelligent Systems Reference Library. Computer Vision in Control Systems-3. 2018. P. 251-286


Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 
 *The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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
