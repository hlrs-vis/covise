/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vpb/FileUtils>
#include <vpb/BuildLog>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#ifdef WIN32

#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#include <stdarg.h>
#if !(defined(__CYGWIN__) || defined(__MINGW32__))
#include <varargs.h>
#endif
#include <io.h>
#include <process.h>
#include <direct.h>
#include <fcntl.h>
#include <winsock.h>
#include <sys/types.h>

#if !(defined(__CYGWIN__) || defined(__MINGW32__))
#define ssize_t int // return type of read() is int on Win32
#define __const const
#define __off_t off_t
#endif

int vpb::access(const char *path, int amode) { return ::_access(path, amode); }
int vpb::open(const char *path, int oflag) { return ::_open(path, oflag); }
FILE *vpb::fopen(const char *filename, const char *mode) { return ::fopen(filename, mode); }
ssize_t vpb::read(int fildes, void *buf, size_t nbyte) { return ::_read(fildes, buf, nbyte); }
ssize_t vpb::write(int fildes, const void *buf, size_t nbyte) { return ::_write(fildes, buf, nbyte); }
int vpb::close(int fildes) { return ::_close(fildes); }
int vpb::fclose(FILE *stream) { return ::fclose(stream); }
int vpb::fchmod(int fildes, mode_t mode) { return 0; }
off_t vpb::lseek(int fildes, off_t offset, int whence) { return ::_lseek(fildes, offset, whence); }
int vpb::lockf(int fildes, int function, off_t size) { return 0; }
int vpb::ftruncate(int fildes, off_t length) { return ::_chsize(fildes, length); }
void vpb::sync() { (void)::_flushall(); }
int vpb::fsync(int fd)
{
    if (fd)
        return ::_commit(fd);
    return 0;
}
int vpb::getpid() { return ::_getpid(); }
int vpb::gethostname(char *name, size_t namelen) { return ::gethostname(name, namelen); }

// See http://cvsweb.xfree86.org/cvsweb/xc/include/Xpoll.h?rev=3.11
// variable XFD_SETSIZE for precedent
int vpb::getdtablesize() { return 256; }
int vpb::mkdir(const char *path, int mode)
{
    int status = ::mkdir(path);
    if (status == 0)
        status = ::chmod(path, mode);
    return status;
}
int vpb::chdir(const char *path) { return ::_chdir(path); }
char *vpb::getCurrentWorkingDirectory(char *path, int nbyte) { return ::_getcwd(path, nbyte); }

#else // WIN32

#include <sys/stat.h>

int vpb::access(const char *path, int amode) { return ::access(path, amode); }
int vpb::open(const char *path, int oflag) { return ::open(path, oflag); }
FILE *vpb::fopen(const char *filename, const char *mode) { return ::fopen(filename, mode); }
ssize_t vpb::read(int fildes, void *buf, size_t nbyte) { return ::read(fildes, buf, nbyte); }
ssize_t vpb::write(int fildes, const void *buf, size_t nbyte) { return ::write(fildes, buf, nbyte); }
int vpb::close(int fildes) { return ::close(fildes); }
int vpb::fclose(FILE *stream) { return ::fclose(stream); }
int vpb::fchmod(int fildes, mode_t mode) { return ::fchmod(fildes, mode); }
off_t vpb::lseek(int fildes, off_t offset, int whence) { return ::lseek(fildes, offset, whence); }
int vpb::lockf(int fildes, int function, off_t size) { return ::lockf(fildes, function, size); }
int vpb::ftruncate(int fildes, off_t length) { return ::ftruncate(fildes, length); }
void vpb::sync() { ::sync(); }
int vpb::fsync(int fildes) { return ::fsync(fildes); }
int vpb::getpid() { return ::getpid(); }
int vpb::gethostname(char *name, size_t namelen) { return ::gethostname(name, namelen); }
int vpb::getdtablesize() { return ::getdtablesize(); }
int vpb::mkdir(const char *path, int mode) { return ::mkdir(path, mode); }
int vpb::chdir(const char *path) { return ::chdir(path); }
char *vpb::getCurrentWorkingDirectory(char *path, int nbyte) { return ::getcwd(path, nbyte); }

#endif // WIN32

int vpb::mkpath(const char *path, int mode)
{
    if (path == 0)
        return 0;

    vpb::log(osg::NOTICE, "mkpath(%s)", path);

    // first create a list of paths that needs to be checked/created.
    std::string fullpath(path);
    typedef std::list<std::string> Directories;
    Directories directories;
    int pos_start = 0;
    for (std::string::size_type pos_current = 0; pos_current < fullpath.size(); ++pos_current)
    {
        if (fullpath[pos_current] == '\\' || fullpath[pos_current] == '/')
        {
            int size = pos_current - pos_start;
            if (size > 1)
            {
                if (pos_current == 2 && fullpath[1] == ':')
                    directories.push_back(std::string(fullpath, 0, pos_current + 1));
                else
                    directories.push_back(std::string(fullpath, 0, pos_current));

                pos_start = pos_current + 1;
            }
        }
    }
    int size = fullpath.size() - pos_start;
    if (size > 1)
    {
        directories.push_back(fullpath);
    }

    // now check the diretories and create the onces that are required in turn.
    for (Directories::iterator itr = directories.begin();
         itr != directories.end();
         ++itr)
    {
        std::string &path = (*itr);
        int result = 0;
        osgDB::FileType type = osgDB::fileType(path);
        if (type == osgDB::REGULAR_FILE)
        {
            log(osg::NOTICE, "Error cannot create directory %s as a conventional file already exists with that name", path.c_str());
            return 1;
        }
        else if (type == osgDB::FILE_NOT_FOUND)
        {
            // need to create directory.
            result = vpb::mkdir(path.c_str(), mode);
            if (result)
                log(osg::NOTICE, "Error could not create directory %s", path.c_str());
            else
                log(osg::NOTICE, "   created directory %s", path.c_str());

            if (result)
                return result;
        }
    }

    return 0;
}

bool vpb::hasWritePermission(const std::string &filename)
{
    log(osg::NOTICE, "vpb::access(%s, W_OK)=%i", filename.c_str(), vpb::access(filename.c_str(), W_OK));

    if (vpb::access(filename.c_str(), W_OK) == 0)
        return true;

    std::string path = osgDB::getFilePath(filename);
    if (path.empty())
        path = ".";

    log(osg::NOTICE, "vpb::access(%s, W_OK)=%i", path.c_str(), vpb::access(path.c_str(), W_OK));

    return (vpb::access(path.c_str(), W_OK) == 0);
}

std::string vpb::simplifyFileName(const std::string &filename)
{
    typedef std::list<std::string> Directories;
    Directories directories;
    int pos_start = 0;
    for (std::string::size_type pos_current = 0; pos_current < filename.size(); ++pos_current)
    {
        if (filename[pos_current] == '\\' || filename[pos_current] == '/')
        {
            std::string directory = filename.substr(pos_start, pos_current - pos_start);

            if (directory != "..")
                directories.push_back(directory);
            else if (!directories.empty())
                directories.pop_back();

            pos_start = pos_current + 1;
        }
    }

    std::string simplifiedName;
    for (Directories::iterator itr = directories.begin();
         itr != directories.end();
         ++itr)
    {
        simplifiedName += *itr;
        simplifiedName += "/";
    }

    simplifiedName += filename.substr(pos_start, std::string::npos);

    return simplifiedName;
}
