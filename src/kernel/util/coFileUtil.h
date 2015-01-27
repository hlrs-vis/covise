/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_FILE_UTIL_H
#define VR_FILE_UTIL_H
//**************************************************************************
//
//                            (C) 1996
//              Computer Centre University of Stuttgart
//                         Allmandring 30
//                       D-70550 Stuttgart
//                            Germany
//
//
//
// COVISE Basic VR Environment Library
//
//
//
// Author: D.Rantzau
// Date  : 04.05.96
// Last  :
//**************************************************************************/
#include "common.h"

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "coExport.h"

#ifdef _WIN32
#include <windows.h>

#include <process.h>
#include <io.h>
#include <direct.h>
#else
#include <dirent.h>
#include <pwd.h>
#include <unistd.h>
#endif

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif

struct _finddata_t;

namespace covise
{

static const int path_buffer_size = 1024 + 1;

class coFileInfo;

//==========================================================================
//
//==========================================================================
class UTILEXPORT coFile
{
protected:
    coFile(coFileInfo *);

public:
    virtual ~coFile();

    virtual const char *name() const;
    virtual long length() const;
    virtual void close();

    virtual void limit(unsigned int buffersize);

    static bool exists(const char *fname);

protected:
    coFileInfo *rep() const;

private:
    coFileInfo *rep_;

private:
    /* not allowed */
    void operator=(const coFile &);
};

//==========================================================================
//
//==========================================================================
class coInputFile : public coFile
{
protected:
    coInputFile(coFileInfo *);

public:
    virtual ~coInputFile();

    static coInputFile *open(const char *name);

    virtual long read(const char *&start);
};

//==========================================================================
//
//==========================================================================
class coFileInfo
{
public:
    char *name_;
    int fd_;
    char *map_;
#ifdef _WIN32
    struct _stat info_;
#else
    struct stat info_;
#endif
    off_t pos_;
    char *buf_;
    unsigned int limit_;

    coFileInfo(const char *, int fd);
    ~coFileInfo();
};

//==========================================================================
//
//==========================================================================
class coStdInput : public coInputFile
{
public:
    coStdInput();
    virtual ~coStdInput();

    virtual long length() const;
    virtual long read(const char *&start);
};

class coDirectoryImpl;

//==========================================================================
//
//==========================================================================
class UTILEXPORT coDirectory
{

    friend class coDirectoryImpl;

protected:
    coDirectory();

public:
    virtual ~coDirectory();

    static coDirectory *current();
    static coDirectory *open(const char *);
    virtual void close();

    virtual const char *path() const;
    virtual int count() const;
    virtual const char *name(int index) const;
    virtual char *full_name(int index);
    virtual int index(const char *) const;
    virtual int is_directory(int index) const;
    virtual int is_exe(int index) const;
    virtual int getSize(int index) const;
    virtual time_t getDate(int index) const;

    static char *canonical(const char *);
    static int match(const char *name, const char *pattern);
    static char *fileOf(const char *); ///return the filename of a canonical file with absolute path
    static char *dirOf(const char *); ///return the directory (without ending / ) of a canonical file with absolute path
private:
    coDirectoryImpl *impl_;

private:
    /* not allowed */
    coDirectory(const coDirectory &);
    void operator=(const coDirectory &);
};

//==========================================================================
//
//==========================================================================
class coDirectoryEntry
{
public:
    const char *name() const;
    coDirectoryEntry()
    {
        name_ = NULL;
        info_ = NULL;
    }
    // ~coDirectoryEntry() { delete name_; name_=NULL; delete info_; info_=NULL; }
private:
    friend class coDirectory;
    friend class coDirectoryImpl;

    char *name_;
#ifdef _WIN32
    int attrib_;
    struct _stat *info_;
#else
    struct stat *info_;
#endif
};

inline const char *coDirectoryEntry::name() const
{
    return name_;
}
}
#endif
