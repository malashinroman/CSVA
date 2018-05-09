#pragma once
#include <string>
#include <vector>

using namespace std;
vector<string> ListAllFiles(string folder, int* openStatus);
vector<string> ListImagesOnly(string folder, int* openStatus);
int makeDirectory(string folder);
int remove_directory(const char *path);
vector<string> ListDirectoriesOnly(string folder, int* openStatus);
