/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_cache.cpp
 *
 *  Project : AudioServer
 *
 *  Package : AudioServer prototype
 *
 *  Author  : Marc Schreier                           Date: 05/05/2002
 *
 *  Purpose : Cache functions
 *
 *********************************************************************
 */

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif

#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "common.h"
#include "as_cache.h"
//#include "as_gui.h"

/*
   ######################################################################

   Warning in File Handling help:

   The C run-time libraries have a preset limit for the number of
   files that can be open at any one time. The limit for applications
   that link with the single-thread static library (LIBC.LIB) is 64
   file handles or 20 file streams. Applications that link with either
   the static or dynamic multithread library (LIBCMT.LIB or MSVCRT.LIB
   and MSVCRT.DLL), have a limit of 256 file handles or 40 file
streams. Attempting to open more than the maximum number of file
handles or file streams causes program failure.

######################################################################

*/

/*!
 * Initialize cache
 *
 * @param none
 *
 * @return  :
 */
as_cache::as_cache()
{
    char msg[_MAX_PATH + 1000];
    char appPath[_MAX_PATH + 1000];
    int error;
    unsigned long val;

    initialised = false;

    this->file = NULL;
    this->fileSizeExpected = 0;
    this->fileSizeWritten = 0;

    GetCurrentDirectory(_MAX_PATH + 1000, appPath);

    // default sounds directory
    sprintf(this->directoryDefaultSounds, "%s\\default", appPath);
    if (!SetCurrentDirectory(this->directoryDefaultSounds))
    {
        error = GetLastError();
        if (ERROR_FILE_NOT_FOUND == error)
        {
            // directory does not exist, create it

            int rc;
            rc = MessageBox(hWndMain, "Default sounds directory does not exist.\nIt will now be created.",
                            "Default", MB_OKCANCEL | MB_ICONEXCLAMATION);
            if (IDCANCEL == rc)
            {
                return;
            }

            if (!CreateDirectory(
                    this->directoryDefaultSounds,
                    NULL // security descriptor, NULL means default
                    ))
            {
                sprintf(msg, "Could not create directory '%s'",
                        this->directoryDefaultSounds);
                AddLogMsg(msg);
                return;
            }
            // set working directory to cache path (created above)
            if (!SetCurrentDirectory(this->directoryDefaultSounds))
            {
                error = GetLastError();
                sprintf(msg, "Cache initialisation error %d", error);
                AddLogMsg(msg);
                return;
            }
        }
        else
        {
            sprintf(msg, "Cache initialisation error %d", error);
            AddLogMsg(msg);
            return;
        }
    }

    // cache directory
    sprintf(this->directory, "%s\\cache", appPath);
    if (!SetCurrentDirectory(this->directory))
    {
        error = GetLastError();
        if (ERROR_FILE_NOT_FOUND == error)
        {
            // directory does not exist, create it?

            int rc;
            rc = MessageBox(hWndMain, "Cache sounds directory does not exist.\nIt will now be created.",
                            "Cache", MB_OKCANCEL | MB_ICONEXCLAMATION);
            if (IDCANCEL == rc)
                return;

            if (!CreateDirectory(
                    this->directory,
                    NULL // security descriptor, NULL means default
                    ))
            {
                sprintf(msg, "Could not create directory '%s'", this->directory);
                AddLogMsg(msg);
                return;
            }
            // set working directory to cache path (created above)
            if (!SetCurrentDirectory(this->directory))
            {
                error = GetLastError();
                sprintf(msg, "Cache initialisation error %d", error);
                AddLogMsg(msg);
                return;
            }
        }
        else
        {
            sprintf(msg, "Cache initialisation error %d", error);
            AddLogMsg(msg);
            return;
        }
    }

    sprintf(msg, "* Cache initialized");
    AddLogMsg(msg);
    sprintf(msg, "  - Current sound files cache directory:");
    AddLogMsg(msg);
    sprintf(msg, "    %s", this->directory);
    AddLogMsg(msg);

    val = this->getNumberOfCachedFiles();
    sprintf(msg, "  - Number of files: %ld", val);
    AddLogMsg(msg);

    val = this->getUsedDiskSpacekB();
    sprintf(msg, "  - Used disk space: %ld kb", val);
    AddLogMsg(msg);

    initialised = true;
}

as_cache::~as_cache()
{
}

/*!
 * Clear cache: remove all files and all subdirectories
 *
 * @param none
 *
 * @return int  :
 */
int as_cache::clear()
{
    WIN32_FIND_DATA ffData; // returned information
    HANDLE hFFF;
    char path[_MAX_PATH + 1000];
    char subdir[_MAX_PATH + 1000];
    int error;
    char msg[_MAX_PATH + MSG_LEN];

    if (NULL == this->directory)
        return -1;

    sprintf(path, "%s\\*", this->directory, "\\*");

    hFFF = FindFirstFile(
        path,
        &ffData);

    if (INVALID_HANDLE_VALUE == hFFF)
        return -1;

    while (FindNextFile(hFFF, &ffData))
    {
        // get path and filename
        sprintf(subdir, "%s\\%s", this->directory, ffData.cFileName);
        // subdirectory?
        if (FILE_ATTRIBUTE_DIRECTORY == (FILE_ATTRIBUTE_DIRECTORY & ffData.dwFileAttributes))
        {
            // skip . and .. pseudo files
            if (0 == strcmp(".", ffData.cFileName))
                continue;
            if (0 == strcmp("..", ffData.cFileName))
                continue;

            // clear all files in subdirectories
            clearSubdirectory(subdir);

            // remove subdirectory
            if (!RemoveDirectory(subdir))
            {
                error = GetLastError();
                sprintf(msg, "RemoveDirectory error %d (path=%s)", error, subdir);
                AddLogMsg(msg);
            }
            continue;
        }
        removeFile(subdir);
    }

    FindClose(hFFF);

    AddLogMsg("Cache cleared");
    return 0;
}

/*!
 * Removes subirectory inside cache
 *
 * @param subdirpath : directory to delete
 *
 * @return int  : error code (0 = ok, -1 = error)
 */
int as_cache::clearSubdirectory(char *subdirpath)
{
    WIN32_FIND_DATA ffData; // returned information
    HANDLE hFFF;
    char path[_MAX_PATH + 1000];
    char subdir[_MAX_PATH + 1000];
    int error;
    char msg[_MAX_PATH + 1280];

    if (NULL == subdirpath)
        return -1;
    if (0 != strnicmp(this->directory, subdirpath, strlen(this->directory)))
    {
        sprintf(msg, "clearSubdirectory( \"%s\" ): invalid path outside cache!!!", subdirpath);
        AddLogMsg(msg);
        return -1;
    }

    sprintf(path, "%s\\*", subdirpath);

    hFFF = FindFirstFile(
        path,
        &ffData);

    if (INVALID_HANDLE_VALUE == hFFF)
        return -1;

    while (FindNextFile(hFFF, &ffData))
    {
        // get path and filename
        sprintf(subdir, "%s\\%s", subdirpath, ffData.cFileName);
        // subdirectories
        if (FILE_ATTRIBUTE_DIRECTORY == (FILE_ATTRIBUTE_DIRECTORY & ffData.dwFileAttributes))
        {
            // skip . and .. pseudo files
            if (0 == strcmp(".", ffData.cFileName))
                continue;
            if (0 == strcmp("..", ffData.cFileName))
                continue;

            // clear all files in subdirectories
            clearSubdirectory(subdir);

            // remove subdirectory
            if (!RemoveDirectory(subdir))
            {
                error = GetLastError();
                sprintf(msg, "RemoveDirectory error %d (path=%s)", error, subdir);
                AddLogMsg(msg);
            }
            continue;
        }
        removeFile(subdir);
    }

    FindClose(hFFF);

    return 0;
}

/*!
 * Deletes one file in cache
 *
 * @param filename : file to delete
 *
 * @return int  : error code
 */
int as_cache::removeFile(char *filename)
{
    int error;
    char msg[_MAX_PATH + 1280];

    if (NULL == filename)
        return -1;

    if (!DeleteFile(filename))
    {
        error = GetLastError();
        sprintf(msg, "Could not delete file (error %d):", error);
        AddLogMsg(msg);
        sprintf(msg, "  %s", filename);
        AddLogMsg(msg);
        return -error;
    }
    sprintf(msg, "Removed file from cache: %s", filename);
    AddLogMsg(msg);

    return 0;
}

/*!
 * Recursively calculates used diskspace of files inside subdirectory
 *
 * @param subdirpath : directory name
 *
 * @return unsigned long  : kB of used disk space
 */
unsigned long as_cache::getUsedSubDirSpacekB(char *subdirpath)
{
    WIN32_FIND_DATA ffData; // returned information
    HANDLE hFFF;
    char path[_MAX_PATH + 1000];
    char subdir[_MAX_PATH + 1000];
    unsigned long usedkB = 0;

    if (NULL == subdirpath)
        return 0;

    sprintf(path, "%s\\*", subdirpath);

    hFFF = FindFirstFile(
        path,
        &ffData);

    if (INVALID_HANDLE_VALUE == hFFF)
        return 0;

    while (FindNextFile(hFFF, &ffData))
    {
        // get path and filename
        sprintf(subdir, "%s\\%s", subdirpath, ffData.cFileName);
        // subdirectories
        if (FILE_ATTRIBUTE_DIRECTORY == (FILE_ATTRIBUTE_DIRECTORY & ffData.dwFileAttributes))
        {
            // skip . and .. pseudo files
            if (0 == strcmp(".", ffData.cFileName))
                continue;
            if (0 == strcmp("..", ffData.cFileName))
                continue;
            usedkB += getUsedSubDirSpacekB(subdir);
            continue;
        }
        usedkB += ((ffData.nFileSizeHigh * MAXDWORD) + ffData.nFileSizeLow) / 1024;
    }

    FindClose(hFFF);

    return usedkB;
}

/*!
 * Counts used disk space of cache directory including subdirectories
 *
 * @param none
 *
 * @return unsigned long  : used disk space in kB
 */
unsigned long as_cache::getUsedDiskSpacekB()
{
    unsigned long usedkB = 0; // 2^32 kB or 4096 GB

    if (NULL == this->directory)
        return 0;

    usedkB = getUsedSubDirSpacekB(this->directory);

    return usedkB;
}

/*!
 * Counts number of files inside cache subdirectory
 *
 * @param subdirpath : directory name
 *
 * @return unsigned long  : number of files
 */
unsigned long as_cache::getNumberOfSubdirCachedFiles(char *subdirpath)
{
    WIN32_FIND_DATA ffData; // returned information
    HANDLE hFFF;
    char path[_MAX_PATH + 1000];
    char subdir[_MAX_PATH + 1000];
    unsigned long count = 0;

    if (NULL == subdirpath)
        return 0;

    sprintf(path, "%s\\*", subdirpath);

    hFFF = FindFirstFile(
        path,
        &ffData);

    if (INVALID_HANDLE_VALUE == hFFF)
        return 0;

    while (FindNextFile(hFFF, &ffData))
    {
        // get path and filename
        sprintf(subdir, "%s\\%s", subdirpath, ffData.cFileName);
        // subdirectories
        if (FILE_ATTRIBUTE_DIRECTORY == (FILE_ATTRIBUTE_DIRECTORY & ffData.dwFileAttributes))
        {
            // skip . and .. pseudo files
            if (0 == strcmp(".", ffData.cFileName))
                continue;
            if (0 == strcmp("..", ffData.cFileName))
                continue;
            count += getNumberOfSubdirCachedFiles(subdir);
            continue;
        }
        count++;
    }

    FindClose(hFFF);

    return count;
}

/*!
 * Get number of files in cache including those in subdirectories
 *
 * @param none
 *
 * @return unsigned long  : number of files
 */
unsigned long as_cache::getNumberOfCachedFiles()
{
    unsigned long count = 0;

    if (NULL == this->directory)
        return 0;

    count = getNumberOfSubdirCachedFiles(this->directory);

    return count;
}

/*!
 * Creates new file in cache
 *
 * @param filename : name of file to create
 * @param size : expected size of file
 *
 * @return int  : error code
 */
long as_cache::newFile(char *filename, long size)
{
    char msg[MSG_LEN];
    char *cacheFilename;

    if (NULL == filename)
    {
        sprintf(msg, "newFile: Invalid file name");
        AddLogMsg(msg);
        return -1;
    }

    // strip path to get file name in cache
    cacheFilename = getFilenameInCache(filename);
    if (NULL == cacheFilename)
    {
        sprintf(msg, "newFile: Invalid cache file name");
        AddLogMsg(msg);
        return -1;
    }

    this->skipData = false;
    this->fileSizeExpected = size;
    this->fileSizeWritten = 0;

    SetProgress(0, size);

    if (((this->getUsedDiskSpacekB() + (size / 1024)) / 1024) > this->GetMaxCacheSize())
    {
        this->skipData = true;
        SetStatusMsg("Skipping file...");
        sprintf(msg, "#Error: File too large, it would exceed cache size! -> skipping data...");
        AddLogMsg(msg);
        return size;
    }

    this->file = fopen(cacheFilename, "wb");

    sprintf(msg, "as_cache::newFile(%s, %d)", cacheFilename, size);
    AddLogMsg(msg);

    if (NULL == this->file)
    {
        sprintf(msg, "Could not create new file, error %d", GetLastError());
        AddLogMsg(msg);
        return -1;
    }

    return size;
}

/*!
 * Writes date to previously created file
 *
 * @param *data : pointer to block with data to be written
 * @param blocksize : size of data block to be written
 *
 * @return long  : remaining size of file to be written
 */
long as_cache::writeToFile(char *data, unsigned long blocksize)
{
    if (false == this->skipData)
    {
        char msg[MSG_LEN];
        if (NULL == this->file)
        {
            AddLogMsg("Could not write data to file: wrong pointer!");
            return -1;
        }
        if (blocksize != fwrite(data, sizeof(char), blocksize, this->file))
        {
            sprintf(msg, "Could not write data to file, error %d", GetLastError());
            AddLogMsg(msg);
            return -1;
        }
    }
    this->fileSizeWritten += blocksize;

    //	sprintf(msg, "* Written %ld data blocks to file, now %ld of %ld received",
    //		blocksize, fileSizeWritten, fileSizeExpected);
    //	AddLogMsg(msg);

    SetProgress(fileSizeWritten, 0);

    if (fileSizeExpected <= fileSizeWritten)
    {
        if (false == this->skipData)
            fclose(this->file);
        this->fileSizeExpected = 0;
        SetProgress(0, 0);
        return 0;
    }
    return fileSizeExpected - fileSizeWritten;
}

/*!
 * Tests if file exists inside cache
 *
 * @param *filename : name of file to test
 *
 * @return bool  : true if file exists, false if file has size=0 or doesn't exist at all
 */
char *as_cache::fileExists(char *filename)
{
    WIN32_FIND_DATA ffData; // returned information
    HANDLE hFFF;
    char msg[_MAX_PATH + 1000];

    char *slashpos;
    unsigned int i;

    char filenameNew[_MAX_PATH + 1000];
    char cachePath[_MAX_PATH + 1000];

    assert(filename);
    if (0 == strcmp("", filename))
    {
        AddLogMsg("Cache fileExists: Empty filename");
        return NULL;
    }

    //	sprintf(msg, "Original filename: %s", filename);
    //	AddLogMsg(msg);

    // convert slashes according to OS type
    for (i = 0; i < strlen(filename); i++)
    {
        switch (filename[i])
        {
        case '/':
#ifdef WIN32
            filenameNew[i] = '\\';
#else
            filenameNew[i] = filename[i];
#endif
            break;
        case '\\':
#ifdef WIN32
            filenameNew[i] = filename[i];
#else
            filenameNew[i] = '/';
#endif
            break;
        default:
            filenameNew[i] = filename[i];
            break;
        }
    }
    filenameNew[strlen(filename)] = '\0';

//	sprintf(msg, "Converted slahes: %s", filenameNew);
//	AddLogMsg(msg);

// cut filename at last slash
#ifdef WIN32
    slashpos = strrchr(filenameNew, '\\');
    if (NULL != slashpos)
    {
        // use converted file name
        sprintf(cachePath, "%s\\%s\0", this->directory, ++slashpos);
    }
    else
    {
        // file name contains no slashes, use it as is
        sprintf(cachePath, "%s\\%s\0", this->directory, filename);
    }
#else
    slashpos = strrchr(filenameNew, '/');
    if (NULL != slashpos)
    {
        // use converted file name
        sprintf(cachePath, "%s/%s\0", this->directory, ++slashpos);
    }
    else
    {
        // file name contains no slashes, use it as is
        sprintf(cachePath, "%s/%s\0", this->directory, filename);
    }
#endif
    //	sprintf(msg, "Cache file: %s", cachePath);
    //	AddLogMsg(msg);
    /*	
      file = fopen(filename, "r");
      if (NULL == file) {
         sprintf(msg, "Requested file not found: %s", filename);
         AddLogMsg(msg);
         return false;
      }
      fclose(file);
   */
    hFFF = FindFirstFile(
        cachePath,
        &ffData);

    if (INVALID_HANDLE_VALUE == hFFF)
    {
        //		sprintf(msg, "FindFirstFile error %d, file:\n'%s'", GetLastError(), cachePath);
        //		AddLogMsg(msg);
        FindClose(hFFF);
        return NULL;
    }
    if ((0 == ffData.nFileSizeHigh) && (0 == ffData.nFileSizeLow))
    {
        // empty file
        sprintf(msg, "Requested file is empty: %s", filename);
        AddLogMsg(msg);
        FindClose(hFFF);
        return NULL;
    }

    strncpy(this->name, cachePath, strlen(cachePath));
    this->name[strlen(cachePath)] = '\0';
    //	AddLogMsg(filename);

    FindClose(hFFF);
    return (this->name);
}

/*!
 * Tests if file exists inside default directory
 *
 * @param *filename : name of file to test
 *
 * @return bool  : true if file exists, false if file has size=0 or doesn't exist at all
 */
char *as_cache::fileExistsDefault(char *filename)
{
    WIN32_FIND_DATA ffData; // returned information
    HANDLE hFFF;
    char msg[_MAX_PATH + 1000];

    char *slashpos;
    unsigned int i;

    char filenameNew[_MAX_PATH + 1000];
    char cachePath[_MAX_PATH + 1000];

    assert(filename);
    if (0 == strcmp("", filename))
    {
        AddLogMsg("Cache fileExistsDefault: Empty filename");
        return NULL;
    }

    //	sprintf(msg, "Original filename: %s", filename);
    //	AddLogMsg(msg);

    // convert slashes according to OS type
    for (i = 0; i < strlen(filename); i++)
    {
        switch (filename[i])
        {
        case '/':
#ifdef WIN32
            filenameNew[i] = '\\';
#else
            filenameNew[i] = filename[i];
#endif
            break;
        case '\\':
#ifdef WIN32
            filenameNew[i] = filename[i];
#else
            filenameNew[i] = '/';
#endif
            break;
        default:
            filenameNew[i] = filename[i];
            break;
        }
    }
    filenameNew[strlen(filename)] = '\0';

//	sprintf(msg, "Converted slahes: %s", filenameNew);
//	AddLogMsg(msg);

// cut filename at last slash
#ifdef WIN32
    slashpos = strrchr(filenameNew, '\\');
    if (NULL != slashpos)
    {
        // use converted file name
        sprintf(cachePath, "%s\\%s\0", this->directoryDefaultSounds, ++slashpos);
    }
    else
    {
        // file name contains no slashes, use it as is
        sprintf(cachePath, "%s\\%s\0", this->directoryDefaultSounds, filename);
    }
#else
    slashpos = strrchr(filenameNew, '/');
    if (NULL != slashpos)
    {
        // use converted file name
        sprintf(cachePath, "%s/%s\0", this->directoryDefaultSounds, ++slashpos);
    }
    else
    {
        // file name contains no slashes, use it as is
        sprintf(cachePath, "%s/%s\0", this->directoryDefaultSounds, filename);
    }
#endif
    //	sprintf(msg, "Cache file (default sound): %s", cachePath);
    //	AddLogMsg(msg);
    /*	
      file = fopen(filename, "r");
      if (NULL == file) {
         sprintf(msg, "Requested file not found: %s", filename);
         AddLogMsg(msg);
         return false;
      }
      fclose(file);
   */
    hFFF = FindFirstFile(
        cachePath,
        &ffData);

    if (INVALID_HANDLE_VALUE == hFFF)
    {
        //		sprintf(msg, "FindFirstFile error %d, file:\n'%s'", GetLastError(), cachePath);
        //		AddLogMsg(msg);
        FindClose(hFFF);
        return NULL;
    }
    if ((0 == ffData.nFileSizeHigh) && (0 == ffData.nFileSizeLow))
    {
        // empty file
        sprintf(msg, "Requested file is empty: %s", filename);
        AddLogMsg(msg);
        FindClose(hFFF);
        return NULL;
    }

    strncpy(this->name, cachePath, strlen(cachePath));
    this->name[strlen(cachePath)] = '\0';
    //	AddLogMsg(filename);

    FindClose(hFFF);
    return (this->name);
}

/*!
 * Returns file name with cache path
 *
 * @param *filename : name of file
 *
 * @return: file name
 */
char *as_cache::getFilenameInCache(char *filename)
{
    char *slashpos;
    unsigned int i;

    char filenameNew[_MAX_PATH + 1000];
    char cachePath[_MAX_PATH + 1000];

    assert(filename);
    if (0 == strcmp("", filename))
    {
        AddLogMsg("Cache getFilenameInCache: Empty filename");
        return NULL;
    }

    // convert slashes according to OS type
    for (i = 0; i < strlen(filename); i++)
    {
        switch (filename[i])
        {
        case '/':
#ifdef WIN32
            filenameNew[i] = '\\';
#else
            filenameNew[i] = filename[i];
#endif
            break;
        case '\\':
#ifdef WIN32
            filenameNew[i] = filename[i];
#else
            filenameNew[i] = '/';
#endif
            break;
        default:
            filenameNew[i] = filename[i];
            break;
        }
    }
    filenameNew[strlen(filename)] = '\0';

// cut filename at last slash
#ifdef WIN32
    slashpos = strrchr(filenameNew, '\\');
    if (NULL != slashpos)
    {
        // use converted file name
        sprintf(cachePath, "%s\\%s\0", this->directory, ++slashpos);
    }
    else
    {
        // file name contains no slashes, use it as is
        sprintf(cachePath, "%s\\%s\0", this->directory, filename);
    }
#else
    slashpos = strrchr(filenameNew, '/');
    if (NULL != slashpos)
    {
        // use converted file name
        sprintf(cachePath, "%s/%s\0", this->directory, ++slashpos);
    }
    else
    {
        // file name contains no slashes, use it as is
        sprintf(cachePath, "%s/%s\0", this->directory, filename);
    }
#endif

    strncpy(this->name, cachePath, strlen(cachePath));
    this->name[strlen(cachePath)] = '\0';

    return (this->name);
}

bool as_cache::IsInitialised()
{
    return this->initialised;
}

unsigned long as_cache::GetFreeDiskSpaceMB()
{
    unsigned long SectorsPerCluster;
    unsigned long BytesPerSector;
    unsigned long NumberOfFreeClusters;
    unsigned long TotalNumberOfClusters;
    unsigned long freeMB;
    double dFreeMB;

    char rootPath[4];

    if (NULL == this->directory)
        return 0;

    // create root path (driver letter + backslash)
    strncpy(rootPath, this->directory, 2);
    rootPath[2] = '\\';
    rootPath[3] = '\0';

    if (false == GetDiskFreeSpace(
                     rootPath, // pointer to root path
                     &SectorsPerCluster, // pointer to sectors per cluster
                     &BytesPerSector, // pointer to bytes per sector
                     &NumberOfFreeClusters, // pointer to number of free clusters
                     &TotalNumberOfClusters // pointer to total number of clusters
                     ))
    {
        char msg[MSG_LEN];
        sprintf(msg, "GetDiskFreeSpace(%s) error %d", rootPath, GetLastError());
        AddLogMsg(msg);
    }
    dFreeMB = (double)BytesPerSector * (double)SectorsPerCluster * (double)NumberOfFreeClusters / ((double)1024 * (double)1024);
    freeMB = (unsigned long)ceil(dFreeMB);
    return freeMB;
}

void as_cache::SetMaxCacheSize(unsigned long size)
{
    this->maxSizeMB = size;
}

unsigned long as_cache::GetMaxCacheSize()
{
    return this->maxSizeMB;
}
