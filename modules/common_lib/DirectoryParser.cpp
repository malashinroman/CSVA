#ifdef WIN32
#include "DIRENT.h"
#include <direct.h>
#else
#include "dirent.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "DirectoryParser.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <iostream>
#define BOOST_FILESYSTEM_NO_DEPRECATED

vector<string> ListAllFiles(string folder, int* openStatus)
{
	vector<string> files;
#ifndef WIN32
	DIR *dir;
	struct dirent *ent;
	//String s;
	dir = opendir(folder.c_str());

	*openStatus = 1;
	if(dir == NULL)
	{
		*openStatus = 0;
		return files;
	}
	ent = readdir(dir);
	while(ent != NULL)
	{
		string t = string(ent->d_name);
		if((t !=".") && (t !=".."))
		{
			files.push_back(t);
		}
		//free(ent);
		ent = readdir(dir);
	}
	closedir(dir);
	return files;
#else

	namespace fs = boost::filesystem;

	boost::progress_timer t( std::clog );

	fs::path full_path( fs::initial_path<fs::path>() );

	full_path = fs::system_complete( fs::path( folder) );

	unsigned long file_count = 0;
	unsigned long dir_count = 0;
	unsigned long other_count = 0;
	unsigned long err_count = 0;

	if ( !fs::exists( full_path ) )
	{
	  *openStatus = 0;
	  //std::cout << "\nNot found: " << full_path.file_string() << std::endl;
	  return files;
	}
	if ( fs::is_directory( full_path ) )
	{
		//std::cout << "\nIn directory: "
		//	<< full_path.directory_string() << "\n\n";
		fs::directory_iterator end_iter;
		for ( fs::directory_iterator dir_itr( full_path );
			dir_itr != end_iter;
			++dir_itr )
		{
			try
			{
				files.push_back(dir_itr->path().filename().string());
				if ( fs::is_directory( dir_itr->status() ) )
				{
					++dir_count;
					std::cout << dir_itr->path().filename() << " [directory]\n";
				}
				else if ( fs::is_regular_file( dir_itr->status() ) )
				{
					++file_count;
					std::cout << dir_itr->path().filename() << "\n";
				}
				else
				{
					++other_count;
					std::cout << dir_itr->path().filename() << " [other]\n";
				}

			}

			catch ( const std::exception & ex )
			{
				++err_count;
				std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
			}
		}
		std::cout << "\n" << file_count << " files\n"
			<< dir_count << " directories\n"
			<< other_count << " others\n"
			<< err_count << " errors\n";
	}
	else // must be a file
	{
		//std::cout << "\nFound: " << full_path.file_string() << "\n";
	}
#endif
	return files;
}

int makeDirectory(string folder)
{
	//alternative is	mkdir(folder.c_str());
	boost::filesystem::create_directory(folder);

//#ifdef WIN32
//	mkdir(folder.c_str());
//#else
//	mkdir(folder.c_str(), 0700);
//#endif
	return 0;
}

vector<string> ListImagesOnly(string folder, int* openStatus)
{
	vector<string> tmp = ListAllFiles(folder, openStatus);
	vector<string> imageFiles;
	vector<string> extensions;
	extensions.push_back(".jpg");
	extensions.push_back(".png");
	extensions.push_back(".jpeg");
	extensions.push_back(".bmp");
	extensions.push_back(".JPG");
	extensions.push_back(".PNG");
	extensions.push_back(".JPEG");
	extensions.push_back(".BMP");
	for(int i = 0; i < tmp.size(); i++)
	{
		int finded = 0;
		for(int j = 0; j < extensions.size(); j++)
		{
			std::size_t found = tmp.at(i).find(extensions.at(j));
			if(found!= std::string::npos)
			{
				finded = 1;
				break;
			}
		}
		if(finded)
		{
			imageFiles.push_back(tmp.at(i));
		}
	}
	return imageFiles;
}
vector<string> ListDirectoriesOnly(string folder, int* openStatus)
{
	vector<string> files;
	namespace fs = boost::filesystem;
	boost::progress_timer t( std::clog );
	fs::path full_path( fs::initial_path<fs::path>() );
	full_path = fs::system_complete( fs::path( folder) );

	unsigned long file_count = 0;
	unsigned long dir_count = 0;
	unsigned long other_count = 0;
	unsigned long err_count = 0;

	if ( !fs::exists( full_path ) )
	{
		*openStatus = 0;
	  return files;
	}
	if ( fs::is_directory( full_path ) )
	{
		//std::cout << "\nIn directory: "
		//	<< full_path.directory_string() << "\n\n";
		fs::directory_iterator end_iter;
		for ( fs::directory_iterator dir_itr( full_path );
			dir_itr != end_iter;
			++dir_itr )
		{
			try
			{
				if ( fs::is_directory( dir_itr->status() ) )
				{
					files.push_back(dir_itr->path().filename().string());
					++dir_count;
					//std::cout << dir_itr->path().filename() << " [directory]\n";
				}
				else if ( fs::is_regular_file( dir_itr->status() ) )
				{
					++file_count;
					//std::cout << dir_itr->path().filename() << "\n";
				}
				else
				{
					++other_count;
					//std::cout << dir_itr->path().filename() << " [other]\n";
				}

			}

			catch ( const std::exception & ex )
			{
				++err_count;
				std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
			}
		}
		/*std::cout << "\n" << file_count << " files\n"
			<< dir_count << " directories\n"
			<< other_count << " others\n"
			<< err_count << " errors\n";*/
	}
	else // must be a file
	{
		//std::cout << "\nFound: " << full_path.file_string() << "\n";
	}
	return files;
}
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
int remove_directory(const char *path)
{
	DIR *d = opendir(path);
	size_t path_len = strlen(path);
	int r = -1;

	if (d)
	{
		struct dirent *p;

		r = 0;

		while (!r && (p=readdir(d)))
		{
			int r2 = -1;
			char *buf;
			size_t len;

			/* Skip the names "." and ".." as we don't want to recurse on them. */
			if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, ".."))
			{
				continue;
			}

			len = path_len + strlen(p->d_name) + 2;
			buf = (char*)malloc(len);

			if (buf)
			{
				struct stat statbuf;
#ifndef WIN32 //For some reason win doesn't have recommended snprintf
				snprintf(buf, len, "%s/%s", path, p->d_name);
#else
				sprintf(buf, "%s/%s", path, p->d_name);
#endif
				if (!stat(buf, &statbuf))
				{
					if (S_ISDIR(statbuf.st_mode))
					{
						r2 = remove_directory(buf);
					}
					else
					{
						r2 = unlink(buf);
					}
				}

				free(buf);
			}

			r = r2;
		}

		closedir(d);
	}

	if (!r)
	{
		r = rmdir(path);
	}

	return r;
}
