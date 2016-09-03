// Copyright 2008-2014 Simon Ekström

#include "Common.h"

#include "../FileUtil.h"

#include <sys/types.h>
#include <dirent.h>

void file_util::find_files(const char* path, std::vector<std::string>& files)
{
    DIR* dir = opendir(path);
    if (dir)
    {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL)
        {
            if (entry->d_type != DT_DIR)
            {
                files.push_back(entry->d_name);
            }
        }
        closedir(dir);
    }
}

void file_util::find_directories(const char* path, std::vector<std::string>& directories)
{
    DIR* dir = opendir(path);
    if (dir)
    {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL)
        {
            if (entry->d_type == DT_DIR)
            {
                directories.push_back(entry->d_name);
            }
        }
        closedir(dir);
    }
}


