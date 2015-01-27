/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//	Sasha Koruga
//	skoruga@ucsd.edu

#include <iostream>
#include <util/coFileUtil.h>
//#include <config/CoviseConfig.h>
#include "DirectoryAssistance.h"

DirectoryAssistance::DirectoryAssistance()
{
}

DirectoryAssistance::~DirectoryAssistance()
{
}

void DirectoryAssistance::AssureEndSlash(std::string &dir) const
{
    //const char slash = coCoviseConfig::getEntry("COVER.Plugin.PhotosynthVR.Slash")[0];
    const char slash = '/';
    if (dir[dir.length() - 1] != slash)
    {
        dir.push_back(slash);
    }
}

bool DirectoryAssistance::FindFileWithText(const std::string &baseDir, const std::string &text, std::string &output) const
{
    std::string directory = baseDir;

    AssureEndSlash(directory);
    coDirectory *dir = coDirectory::open(directory.c_str());

    if (dir)
    {
        for (int i = 0; i < dir->count(); ++i)
        {
            std::string name = dir->name(i);
            if (name.find(text) && !IsDirectory(directory, name))
            {
                output = name;
                return true;
            }
        }
        dir->close();
        delete dir;
    }

    return false;
}

bool DirectoryAssistance::IsDirectory(const std::string &dir) const
{
    coDirectory *dir2 = coDirectory::open(dir.c_str());
    if (dir2)
        return true;
    else
        return false;
}

bool DirectoryAssistance::IsDirectory(const std::string &baseDir, const std::string &file) const
{
    std::string chckdir = baseDir;
    chckdir += file;

    coDirectory *dir2 = coDirectory::open(chckdir.c_str());
    if (dir2)
        return true;
    else
        return false;
}

void DirectoryAssistance::ReadDirectory(const std::string &baseDir, std::list<std::string> &files) const
{
    coDirectory *dir = coDirectory::open(baseDir.c_str());

    if (dir)
    {
        for (int i = 0; i < dir->count(); ++i)
        {
            std::string name = dir->name(i);
            if (name[0] != '.' && *(--(name.end())) != '~')
            {
                if (IsDirectory(baseDir, name))
                    files.push_back(dir->name(i));
            }
        }
        dir->close();
        delete dir;
    }
    else
        std::cerr << "DirectoryAssistance::ReadDirectory -- Not a directory" << std::endl;

    files.sort();
}

void DirectoryAssistance::ReadFiles(const std::string &baseDir, std::list<std::string> &files) const
{
    coDirectory *dir = coDirectory::open(baseDir.c_str());

    if (dir)
    {
        //printf("\\|/%d\\|/\n",dir->count());
        for (int i = 0; i < dir->count(); ++i)
        {
            std::string name = dir->name(i);
            if (name[0] != '.' && *(--(name.end())) != '~')
                files.push_back(dir->name(i));
        }
        dir->close();
        delete dir;
    }
    //else
    //	std::cerr << "UI::ReadDirectory -- Not a directory" << std::endl;

    files.sort();
}

void DirectoryAssistance::ReadFiles(const std::string &baseDir, std::vector<std::string> &files) const
{
    std::list<std::string> files2;
    for (int i = 0; i < files.size(); ++i)
        files2.push_back(files[i]);

    ReadFiles(baseDir, files2);

    files.clear();
    //printf(":: %d :: \n",files2.size());
    for (std::list<std::string>::iterator iter = files2.begin(); iter != files2.end(); ++iter)
        files.push_back(*iter);
}
