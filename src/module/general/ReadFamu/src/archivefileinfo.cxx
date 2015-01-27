/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file archivefileinfo.cxx
 * container for information about an archive file.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

#include "archivefileinfo.hxx" // container for information about an archive file.
#include <time.h>
#include <sstream>

ArchiveFileInfo::ArchiveFileInfo(std::string version,
                                 double versionNumber,
                                 std::string content,
                                 std::string comment,
                                 std::string inputFile)
    : _version(version)
    , _versionNumber(versionNumber)
    , _content(content)
    , _comment(comment)
    , _inputFile(inputFile)
{
    time(&_saveTime);
}

void ArchiveFileInfo::writeObject(ObjectOutputStream *archive) const
{
    archive->writeString(_version);
    archive->writeDouble(_versionNumber);
    archive->writeString(_content);
    archive->writeString(_comment);
    archive->writeString(_inputFile);
    archive->writeBuffer((CHAR_F *)&_saveTime, sizeof(_saveTime));
}

ArchiveFileInfo::ArchiveFileInfo(ObjectInputStream *archive)
    : _version("")
    , _versionNumber(-1)
    , _content("")
    , _comment("")
    , _inputFile("")
{
    _version = archive->readString(); // don't put this in initialization list as order is important!
    _versionNumber = archive->readDouble();
    _content = archive->readString();
    _comment = archive->readString();
    _inputFile = archive->readString();
    archive->readBuffer((CHAR_F *)&_saveTime, sizeof(time_t));
}

std::string ArchiveFileInfo::getSaveTime(void) const
{
    std::ostringstream s;
    s << ctime(&_saveTime);
    return s.str();
}

void ArchiveFileInfo::writeInfoToFile(std::string filenameOfResultsFile) const
{
    (void)filenameOfResultsFile;
}
