/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_cache.h
 *
 *  Project : AudioServer
 *
 *  Package : AudioServer prototype
 *
 *  Author  : Marc Schreier                              Date: 05/05/2002
 *
 *  Purpose : Header file
 *
 *********************************************************************
 */

#if !defined(AS_CACHE_H__)
#define AS_CACHE_H__

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <stdlib.h>
#include <stdio.h>
#include "as_control.h"

class as_cache
{
public:
    bool skipData;
    unsigned long GetMaxCacheSize(void);
    void SetMaxCacheSize(unsigned long size);
    unsigned long GetFreeDiskSpaceMB(void);
    bool IsInitialised(void);
    char *fileExists(char *filename);
    char *fileExistsDefault(char *filename);
    char *getFilenameInCache(char *filename);
    long writeToFile(char *data, unsigned long blocksize);
    long newFile(char *filename, long size);
    unsigned long getNumberOfCachedFiles(void);
    unsigned long getNumberOfSubdirCachedFiles(char *subdirpath);
    unsigned long getUsedSubDirSpacekB(char *subdirpath);
    unsigned long getUsedDiskSpacekB(void);
    int removeFile(char *filename);
    int clear();
    int clearSubdirectory(char *subdirpath);
    as_cache();
    virtual ~as_cache();

private:
    long numberOfChachedFiles;
    char directory[_MAX_PATH];
    char directoryDefaultSounds[_MAX_PATH];
    unsigned long fileSizeWritten;
    unsigned long fileSizeExpected;
    bool initialised;
    FILE *file;
    unsigned long maxSizeMB;
    char name[_MAX_PATH];
};

extern as_cache *AS_Cache;
#endif // !defined(AS_CACHE_H__)
