/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//	Sasha Koruga
//	skoruga@ucsd.edu

#ifndef DIRECTORYASSISTANCE_H_
#define DIRECTORYASSISTANCE_H_

#include <list>
#include <string>

class DirectoryAssistance
{
public:
    DirectoryAssistance();
    ~DirectoryAssistance();
    void ReadDirectory(const std::string &baseDir, std::list<std::string> &files) const;
    void ReadFiles(const std::string &baseDir, std::list<std::string> &mFolders) const;
    void ReadFiles(const std::string &baseDir, std::vector<std::string> &mFolders) const;
    bool FindFileWithText(const std::string &baseDir, const std::string &text, std::string &output) const;
    bool IsDirectory(const std::string &dir) const;
    bool IsDirectory(const std::string &baseDir, const std::string &file) const;
    void AssureEndSlash(std::string &dir) const;
};

#endif /* DIRECTORYASSISTANCE_H_ */
