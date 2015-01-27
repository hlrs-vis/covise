/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
//**************************************************************************

#define USE_AFS // afs overrides unix file permissions

enum
{
    FALSE = 0,
    TRUE = 1
};

#include <iostream>
using namespace std;

#include "coFileUtil.h"
#include <sys/types.h>
#include <sys/stat.h>

namespace covise
{

//==========================================================================
//
//==========================================================================
class coDirectoryImpl
{

    friend class coDirectory;

private:
#ifdef _WIN32
    coDirectoryImpl(intptr_t, struct _finddata_t *, const char *);
#else
    coDirectoryImpl(DIR *, const char *);
#endif
    ~coDirectoryImpl();

    char *name_;
#ifdef _WIN32
    intptr_t dirHandle;
    struct _finddata_t *findData;
#else
    DIR *dir_;
#endif
    coDirectoryEntry *entries_;
    int count_;
    int used_;
    int filled_;

    coDirectoryImpl &filled();
    void do_fill();

    static int dot_slash(const char *);
    static int dot_dot_slash(const char *);
    static const char *home(const char *);
    static const char *eliminate_dot(const char *);
    static int collapsed_dot_dot_slash(char *, char *&start);
    static const char *eliminate_dot_dot(const char *);
    static const char *replace_backslash(const char *);
    static const char *replace_cwd(const char *);
    static const char *collapse_slash_slash(const char *);
    static const char *interpret_tilde(const char *);
    static const char *expand_tilde(const char *, int);
    static const char *real_path(const char *);
    static const char *chop_slash(const char *);
    static const char *getwd();
    static const char *check_covise_path(const char *);
    static const char *check_cwd(const char *file);
    static const char *check_path(const char *file, const char *path);
    static bool is_absolute(const char *pathname);
    static int ifdir(const char *);
};
}

using namespace covise;

//==========================================================================
//
//==========================================================================
coDirectory::coDirectory()
{
    impl_ = NULL;
}

//==========================================================================
//
//==========================================================================
coDirectory::~coDirectory()
{
    close();
    if (NULL != impl_)
    {
        delete impl_;
    }
}

//==========================================================================
//
//==========================================================================
coDirectory *coDirectory::current()
{
    return open(".");
}

//==========================================================================
//
//==========================================================================
coDirectory *coDirectory::open(const char *name)
{
    char *s = canonical(name);

#ifdef _WIN32
    struct _finddata_t *c_file = new struct _finddata_t;
    char pattern[path_buffer_size];
    strcpy(pattern, s);
    strcat(pattern, "/*");
    intptr_t handle = _findfirst(pattern, c_file);
    if (handle == -1)
    {
        delete[] s;
        return NULL;
    }
    coDirectory *d = new coDirectory;
    d->impl_ = new coDirectoryImpl(handle, c_file, s);
#else
    DIR *dir = opendir(s);
    if (dir == NULL)
    {
        delete[] s;
        return NULL;
    }
    coDirectory *d = new coDirectory;
    d->impl_ = new coDirectoryImpl(dir, s);
#endif
    return d;
}

//==========================================================================
//
//==========================================================================
void coDirectory::close()
{
    coDirectoryImpl &d = *impl_;
#ifdef _WIN32
    if (d.dirHandle != -1)
#else
    if (d.dir_ != NULL)
#endif
    {
#ifdef _WIN32
        _findclose(d.dirHandle);
        delete d.findData;
#else
        closedir(d.dir_);
#endif
        coDirectoryEntry *end = &d.entries_[d.used_];
        for (coDirectoryEntry *e = &d.entries_[0]; e < end; e++)
        {
            delete[] e -> name_;
            //#ifndef _WIN32
            delete e->info_;
            //#endif
        }
        delete[] d.entries_;
#ifdef _WIN32
        d.dirHandle = -1;
#else
        d.dir_ = NULL;
#endif
    }
}

//==========================================================================
//
//==========================================================================
const char *coDirectory::path() const
{
    coDirectoryImpl &d = *impl_;
    return d.name_;
}

//==========================================================================
//
//==========================================================================
int coDirectory::count() const
{
    coDirectoryImpl &d = impl_->filled();
    return d.used_;
}

//==========================================================================
//
//==========================================================================
const char *coDirectory::name(int i) const
{
    coDirectoryImpl &d = impl_->filled();
    if (i < 0 || i >= d.count_)
        return NULL; // raise exception -- out of range

    return d.entries_[i].name_;
}

//==========================================================================
//
//==========================================================================
char *coDirectory::full_name(int i)
{
    coDirectoryImpl &d = impl_->filled();
    if (i < 0 || i >= d.count_)
        return NULL; // raise exception -- out of range

    char *tmp = new char[strlen(d.name_) + strlen(d.entries_[i].name_) + 2];
    sprintf(tmp, "%s%s", d.name_, d.entries_[i].name_);
    return (tmp);
}

//==========================================================================
//
//==========================================================================
int coDirectory::index(const char *fname) const
{
    coDirectoryImpl &d = impl_->filled();
    int i = 0, j = d.used_ - 1;
    while (i <= j)
    {
        int k = (i + j) / 2;
        int cmp = strcmp(fname, d.entries_[k].name_);

        if (cmp == 0)
            return k;

        if (cmp > 0)
            i = k + 1;
        else
            j = k - 1;
    }
    return -1;
}

//==========================================================================
//
//==========================================================================
int coDirectory::is_exe(int i) const
{
    coDirectoryImpl &d = impl_->filled();
    if (i < 0 || i >= d.count_)
        return FALSE; // raise exception -- out of range

#if defined(_WIN32) || defined(USE_AFS)
    return 1;
#else
    coDirectoryEntry &e = d.entries_[i];
    if (e.info_ == NULL)
    {
        char *tmp = new char[strlen(d.name_) + strlen(e.name_) + 2];
        sprintf(tmp, "%s/%s", d.name_, e.name_);
        e.info_ = new (struct stat);
        int ret = stat(tmp, e.info_);
        delete[] tmp;
        if (ret < 0)
        {
            delete[] e.info_;
            e.info_ = NULL;
            return FALSE;
        }
    }
    uid_t uid = getuid();
    gid_t gid = getgid();
    if (e.info_->st_uid == uid)
        return ((e.info_->st_mode) & (S_IXUSR | S_IXGRP | S_IXOTH));
    if (e.info_->st_gid == gid)
        return ((e.info_->st_mode) & (S_IXGRP | S_IXOTH));
    return ((e.info_->st_mode) & (S_IXOTH));
#endif
}

//==========================================================================
//
//==========================================================================
time_t coDirectory::getDate(int i) const
{
    coDirectoryImpl &d = impl_->filled();
    if (i < 0 || i >= d.count_)
    {
        // raise exception -- out of range
        return FALSE;
    }

    coDirectoryEntry &e = d.entries_[i];
    if (e.info_ == NULL)
    {
        char *tmp = new char[strlen(d.name_) + strlen(e.name_) + 2];
        sprintf(tmp, "%s/%s", d.name_, e.name_);
#ifdef _WIN32
        e.info_ = new (struct _stat);
        int ret = _stat(tmp, e.info_);
#else
        e.info_ = new (struct stat);
        int ret = stat(tmp, e.info_);
#endif
        delete[] tmp;
        if (ret < 0)
        {
            delete e.info_;
            e.info_ = NULL;

            return (time_t)0;
        }
    }

#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__) || defined(__hpux)
    return (e.info_->st_mtime);
#else
    return (e.info_->st_mtim.tv_sec);
#endif
}
//==========================================================================
//
//==========================================================================
int coDirectory::getSize(int i) const
{
    coDirectoryImpl &d = impl_->filled();
    if (i < 0 || i >= d.count_)
    {
        // raise exception -- out of range
        return FALSE;
    }

    coDirectoryEntry &e = d.entries_[i];
    if (e.info_ == NULL)
    {
        char *tmp = new char[strlen(d.name_) + strlen(e.name_) + 2];
        sprintf(tmp, "%s/%s", d.name_, e.name_);
#ifdef _WIN32
        e.info_ = new (struct _stat);
        int ret = _stat(tmp, e.info_);
#else
        e.info_ = new (struct stat);
        int ret = stat(tmp, e.info_);
#endif
        delete[] tmp;
        if (ret < 0)
        {
            delete e.info_;
            e.info_ = NULL;

            return -1;
        }
    }
    return (e.info_->st_size);
}

//==========================================================================
//
//==========================================================================
int coDirectory::is_directory(int i) const
{
    coDirectoryImpl &d = impl_->filled();
    if (i < 0 || i >= d.count_)
        return FALSE; // raise exception -- out of range

    coDirectoryEntry &e = d.entries_[i];
    if (!e.info_)
    {
        char *tmp = new char[strlen(d.name_) + strlen(e.name_) + 2];
        sprintf(tmp, "%s/%s", d.name_, e.name_);
#ifdef _WIN32
        int ret = 0;
        DWORD attr = GetFileAttributes(tmp);

        if (attr == -1)
        {
            return 0;
        }
        else if (attr & FILE_ATTRIBUTE_DIRECTORY)
        {
            return 1;
        }
        return 0;
#else
        e.info_ = new (struct stat);
        int ret = stat(tmp, e.info_);
#endif
        delete[] tmp;
        if (ret < 0)
        {
            delete e.info_;
            e.info_ = NULL;

            return FALSE;
        }
    }
    return S_ISDIR(e.info_->st_mode);
}

//==========================================================================
//
//==========================================================================
inline int coDirectoryImpl::dot_slash(const char *path)
{
    return path[0] == '.' && (path[1] == '/' || path[1] == '\0');
}

//==========================================================================
//
//==========================================================================
inline int coDirectoryImpl::dot_dot_slash(const char *path)
{
    return ((path[0] == '.' && path[1] == '.')
            && (path[2] == '/' || path[2] == '\0'));
}

//==========================================================================
//
//==========================================================================
bool coDirectoryImpl::is_absolute(const char *pathname)
{
    if (pathname[0] == '/')
        return true;

#ifdef _WIN32
    if (pathname[0] && pathname[1] == ':' && pathname[2] == '/')
        return true;
#endif

    return false;
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::check_covise_path(const char *path)
{
    if (is_absolute(path))
        return path;

    char *covisepath = getenv("COVISE_PATH");

    if (!covisepath)
        return path;

    return check_path(path, covisepath);
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::check_cwd(const char *path)
{
    if (is_absolute(path))
        return path;

    return check_path(path, getwd());
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::check_path(const char *pathname, const char *path)
{
    static char newpath[path_buffer_size];

    char *pathbuf = new char[strlen(path) + 1];
    strcpy(pathbuf, path);

#ifdef _WIN32
    char *dirname = strtok(pathbuf, ";");
#else
    char *dirname = strtok(pathbuf, ":");
#endif

    while (dirname)
    {
#ifdef _WIN32
        char *s = dirname;
        while (*s)
        {
            if (*s == '\\')
                *s = '/';
            s++;
        }
#endif

        sprintf(newpath, "%s/%s", dirname, pathname);
#ifdef _WIN32
        struct _stat statbuf;
        if (!_stat(newpath, &statbuf))
#else
        struct stat statbuf;
        if (!stat(newpath, &statbuf))
#endif
        {
            delete[] pathbuf;
            return newpath;
        }

#ifdef CHECK_COVISE_PATH_UP
// end loop if / or c: is reached
#ifdef _WIN32
        if (strlen(dirname) == 2)
#else
        if (strlen(dirname) == 1)
#endif
            break;

        // also check one directory higher
        for (int i = strlen(dirname) - 2; i > 0; i--)
        {
            if ((dirname[i] == '/')
#ifdef _WIN32
                || (dirname[i] == '\\')
#endif
                    )
            {
                dirname[i] = '\0';
                break;
            }
        }

        sprintf(newpath, "%s/%s", dirname, pathname);
#ifdef _WIN32
        if (!_stat(newpath, &statbuf))
#else
        if (!stat(newpath, &statbuf))
#endif
        {
            delete[] pathbuf;
            return newpath;
        }
#endif

#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }

    delete[] pathbuf;
    return pathname;
}

//==========================================================================
//
//==========================================================================
char *coDirectory::canonical(const char *name)
{
    const char *path = name;
    static char newpath[path_buffer_size];
    const char *s = (path);
    s = coDirectoryImpl::chop_slash(s);
    s = coDirectoryImpl::interpret_tilde(s);
    s = coDirectoryImpl::replace_backslash(s);
    s = coDirectoryImpl::check_cwd(s);
    s = coDirectoryImpl::check_covise_path(s);
    s = coDirectoryImpl::replace_backslash(s);
    s = coDirectoryImpl::replace_cwd(s);
    s = coDirectoryImpl::replace_backslash(s);
    s = coDirectoryImpl::collapse_slash_slash(s);
    s = coDirectoryImpl::eliminate_dot(s);
    s = coDirectoryImpl::eliminate_dot_dot(s);
    s = coDirectoryImpl::collapse_slash_slash(s);

    if (s[0] == '\0')
        sprintf(newpath, "./");

    else if (!coDirectoryImpl::dot_slash(s)
             && !coDirectoryImpl::dot_dot_slash(s)
             && !coDirectoryImpl::is_absolute(s))
        sprintf(newpath, "./%s", s);

    else if (coDirectoryImpl::ifdir(s)
             && s[strlen(s) - 1] != '/')
        sprintf(newpath, "%s/", s);

    else
        sprintf(newpath, "%s", s);

#ifdef _WIN32
    if ((newpath[0] == '/') && (newpath[1] != '/')) // if it is an absolute path and not a UNC name, add a drive letter
    {
        memmove(newpath + 2, newpath, strlen(newpath) + 1);
        newpath[0] = 'A' + _getdrive() - 1;
        newpath[1] = ':';
    }
#endif

    size_t n = strlen(newpath);
    char *r = new char[n + 1];
    strcpy(r, newpath);
    return r;
}

char *coDirectory::fileOf(const char *name)
{
    size_t n = strlen(name);
    char *r = new char[n + 1];
    const char *c = name + n;
    while (c >= name)
    {
        if ((*c == '/') || (*c == ':'))
        {
            strcpy(r, c + 1);
            return r;
        }
        c--;
    }
    strcpy(r, name);
    return r;
}
char *coDirectory::dirOf(const char *name)
{
    size_t n = strlen(name);
    char *r = new char[n + 1];
    strcpy(r, name);
    char *c = r + n;
    while (c >= name)
    {
        if (*c == ':')
        {
            c++;
            *c = '\0';
            return r;
        }
        if (*c == '/')
        {
            *c = '\0';
            return r;
        }
        c--;
    }
    strcpy(r, ".");
    return r;
}

//==========================================================================
//
//==========================================================================
#undef FILENAME_MAX
#define FILENAME_MAX 1024

int coDirectory::match(const char *name, const char *patternToMatch)
{
    if (!patternToMatch || *patternToMatch == '\0')
        return FALSE;

    // aw: support multiple patterns for pattern-matching
    //     need second \0 after last oart for end of matching loop
    char patBuf[FILENAME_MAX];
    strncpy(patBuf, patternToMatch, FILENAME_MAX - 2);
    patBuf[FILENAME_MAX - 1] = '\0';
    patBuf[FILENAME_MAX - 2] = '\0';
    if (strlen(patternToMatch) < FILENAME_MAX - 2)
    {
        patBuf[strlen(patternToMatch) + 1] = '\0';
    }
    else
        fprintf(stderr, "truncated pattern: larger than %d chars", FILENAME_MAX - 2);

    /// loop over multiple patterns
    const char *pattern = patBuf;
    char *colon;

    do
    { // make the colo to a \0 -> we later jump behind it
        if ((colon = strchr((char *)pattern, ';')) != NULL)
            *colon = '\0';

        // try to match the pattern
        const char *s = name;
        const char *end_s = s + strlen(name);
        const char *p = pattern;
        const char *end_p = p + strlen(pattern);

        int go = 1; // set to 0 to quit
        for (; (p < end_p && go); p++, s++)
        {
            if ((*p == '?') && (*s))
            {
                // ? matches all characters
            }

            else if (*p == '*')
            {
                const char *pp = p + 1;
                const char *tpp = pp;
                int patlen = 0;
                if (pp == end_p)
                    return TRUE;
                while ((*tpp) && (*tpp != '*') && (*tpp != '?'))
                {
                    tpp++;
                    patlen++;
                }
                for (; s < end_s && (strncmp(s, pp, patlen) != 0); s++)
                    ;
                p = pp;
                if (s == end_s)
                {
                    go = 0; // leave inner loop and do NOT return TRUE
                    s = NULL;
                    continue; // try next pattern
                }
            }

            else if (s == end_s || (s && (*p != *s)))
            {
                go = 0;
                s = NULL;
                continue; // try next pattern
            }
        }
        if (s == end_s)
            return TRUE;

        // try to match next pattern now
        pattern += strlen(pattern) + 1;
    } while (*pattern);

    // if we get here, all matching failed
    return FALSE;
}

// class coDirectoryImpl

//==========================================================================
//
//==========================================================================
#ifdef _WIN32
coDirectoryImpl::coDirectoryImpl(intptr_t handle, struct _finddata_t *data, const char *name)
#else
coDirectoryImpl::coDirectoryImpl(DIR *d, const char *name)
#endif
{
    size_t n = strlen(name);
    name_ = new char[n + 1];
    strcpy(name_, name);
#ifdef _WIN32
    dirHandle = handle;
    findData = data;
#else
    dir_ = d;
#endif
    entries_ = NULL;
    count_ = 0;
    used_ = 0;
    filled_ = FALSE;
}

//==========================================================================
//
//==========================================================================
coDirectoryImpl::~coDirectoryImpl()
{
    delete[] name_;
}

//==========================================================================
//
//==========================================================================
coDirectoryImpl &coDirectoryImpl::filled()
{
    if (!filled_)
    {
        do_fill();
        filled_ = TRUE;
    }
    return *this;
}

//==========================================================================
//
//==========================================================================
namespace covise
{
static int compare_entries(const void *k1, const void *k2)
{
    coDirectoryEntry *e1 = (coDirectoryEntry *)k1;
    coDirectoryEntry *e2 = (coDirectoryEntry *)k2;
    return strcmp(e1->name(), e2->name());
}
}

//==========================================================================
//
//==========================================================================
void coDirectoryImpl::do_fill()
{
    unsigned int overflows = 0;

#ifdef _WIN32
    int contFind = 1;
    for (/* already initialized in open */; contFind; contFind = _findnext(dirHandle, findData) == 0)
#else
    for (struct dirent *d = readdir(dir_); d != NULL; d = readdir(dir_))
#endif
    {
        if (used_ >= count_)
        {
            ++overflows;
            int new_count = count_ + 50 * overflows;
            coDirectoryEntry *new_entries = new coDirectoryEntry[new_count];
            memmove(new_entries, entries_, count_ * sizeof(coDirectoryEntry));
            delete[] entries_;
            entries_ = new_entries;
            count_ = new_count;
        }
        coDirectoryEntry &e = entries_[used_];
#ifdef _WIN32
        size_t n = strlen(findData->name);
#else
        int n = strlen(d->d_name);
#endif
        e.name_ = new char[n + 1];
#ifdef _WIN32
        strcpy(e.name_, findData->name);
        e.attrib_ = findData->attrib;
        e.info_ = NULL;
#else
        strcpy(e.name_, d->d_name);
        e.info_ = NULL;
#endif
        ++used_;
    }
    qsort(entries_, used_, sizeof(coDirectoryEntry), &compare_entries);
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::home(const char *name)
{
    const char *retval = NULL;
#ifdef _WIN32
    retval = getenv("USERPROFILE");
#else
    struct passwd *pw;
    if (NULL == name)
    {
        char *n = getlogin();
        if (NULL != n)
        {
            pw = getpwnam(n);
            retval = pw->pw_dir;
        }
        else
        {
            retval = getenv("HOME");
        }
    }
    else
    {
        pw = getpwnam(name);
        if (NULL != pw)
        {
            retval = pw->pw_dir;
        }
    }
#endif
    return retval;
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::getwd()
{
    static char cwd[path_buffer_size];
#ifdef _WIN32
    /* ganz tolle implementierung für getcwd... wer macht denn so nen sch... 
#ifdef YAC
   const char *cwd = getenv("YACDIR");
#else
   const char *cwd = getenv("COVISEDIR");
#endif
   if(cwd)
   {
      return cwd;
   }*/
    if (_getcwd(cwd, path_buffer_size) != NULL)
    {
        return cwd;
    }
#else
    if (getcwd(cwd, path_buffer_size) != NULL)
    {
        return cwd;
    }
#endif

    return ".";
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::chop_slash(const char *path)
{
    static char newpath[path_buffer_size];
    char *dest = newpath;

    const char *end = &path[strlen(path) - 1];
    if (*end == '/' && strlen(path) > 3)
    {
        strcpy(newpath, path);
        dest[strlen(path) - 1] = '\0';
        return newpath;
    }
    else
    {
        return path;
    }
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::eliminate_dot(const char *path)
{
    static char newpath[path_buffer_size];
    const char *src;
    char *dest = newpath;

    const char *end = &path[strlen(path)];
    for (src = path; src < end; src++)
    {
        if (dot_slash(src) && dest > newpath && *(dest - 1) == '/')
            src++;
        else
            *dest++ = *src;
    }
    *dest = '\0';
    return newpath;
}

//==========================================================================
//
//==========================================================================
int coDirectoryImpl::collapsed_dot_dot_slash(char *path, char *&start)
{
    if (path == start || *(start - 1) != '/')
        return FALSE;
    if (path == start - 1 && *path == '/')
        return TRUE;
    if (path == start - 2) // doesn't handle double-slash correctly
    {
        start = path;
        return *start != '.';
    }
    if (path < start - 2 && !dot_dot_slash(start - 3))
    {
        for (start -= 2; path <= start; --start)
        {
            if (*start == '/')
            {
                ++start;
                return TRUE;
            }
        }
        start = path;
        return TRUE;
    }
    return FALSE;
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::eliminate_dot_dot(const char *path)
{
    static char newpath[path_buffer_size];
    const char *src;
    char *dest = newpath;

    const char *end = &path[strlen(path)];
    for (src = path; src < end; src++)
    {
        if (dot_dot_slash(src) && collapsed_dot_dot_slash(newpath, dest))
            src += 2;
        else
            *dest++ = *src;
    }
    *dest = '\0';
    return newpath;
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::replace_backslash(const char *path)
{
#ifdef _WIN32
    static char buf[path_buffer_size];
    const char *s = path;
    char *d = buf;
    while (*s)
    {
        if (*s == '\\')
            *d = '/';
        else
            *d = *s;
        s++;
        d++;
    }
    *d = '\0';
    return buf;
#else
    return path;
#endif
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::replace_cwd(const char *path)
{
    static char buf[path_buffer_size];
    if (!strcmp(path, ".") || !strncmp(path, "./", 2))
    {
        strcpy(buf, getwd());
        strcat(buf, path + 1);
        return buf;
    }
    else if (!strcmp(path, "..") || !strncmp(path, "../", 3) || !is_absolute(path))
    {
        strcpy(buf, getwd());
        strcat(buf, "/");
        strcat(buf, path);
        return buf;
    }
    else
        return path;
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::collapse_slash_slash(const char *path)
{
    static char buf[path_buffer_size];
    char *d = buf;
    const char *s = path;
    if (*s) // do not collapse double slashes at the beginning of a filename, it might be a UNC filename
    {
        *d = *s;
        d++;
        s++;
    }
    bool slash = false;
    while (*s)
    {
        if (*s == '/')
        {
            if (!slash)
            {
                *d = *s;
                d++;
            }
            slash = true;
        }
        else
        {
            slash = false;
            *d = *s;
            d++;
        }
        s++;
    }
    *d = '\0';
    return buf;
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::interpret_tilde(const char *path)
{
    static char realpath[path_buffer_size];
    const char *beg = strrchr(path, '~');
    int valid = (beg != NULL && (beg == path || *(beg - 1) == '/'));
    if (valid)
    {
        const char *end = strchr(beg, '/');
        size_t length = (end == NULL) ? strlen(beg) : (end - beg);
        const char *expanded = expand_tilde(beg, (int)length);
        if (expanded == NULL)
            valid = FALSE;
        else
        {
            strcpy(realpath, expanded);
            if (end != NULL)
            {
                strcat(realpath, end);
            }
        }
    }
    return valid ? realpath : path;
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::expand_tilde(const char *tilde, int length)
{
    const char *name = NULL;
    if (length > 1)
    {
        static char buf[path_buffer_size];
        strncpy(buf, tilde + 1, length - 1);
        buf[length - 1] = '\0';
        name = buf;
    }
    return home(name);
}

//==========================================================================
//
//==========================================================================
const char *coDirectoryImpl::real_path(const char *path)
{
    const char *realpath;
    if (*path == '\0')
    {
        realpath = "./";
    }
    else
    {
        realpath = interpret_tilde((path)); /*interpret_slash_slash*/
        realpath = replace_backslash(realpath);
        realpath = check_covise_path(realpath);
        realpath = replace_cwd(realpath);
        realpath = replace_backslash(realpath);
        realpath = collapse_slash_slash(realpath);
    }
    return realpath;
}

//==========================================================================
//
//==========================================================================
int coDirectoryImpl::ifdir(const char *path)
{
#ifdef _WIN32
    //struct _stat st;
    //return _stat(path, &st) == 0 && S_ISDIR(st.st_mode);
    DWORD attr = GetFileAttributes(path);

    if (attr == -1)
    {
        return 0;
    }
    else if (attr & FILE_ATTRIBUTE_DIRECTORY)
    {
        return 1;
    }
    return 0;
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

#ifdef __sgi
#include <sys/mman.h>
#endif

// coFileInfo Stuff

//==========================================================================
//
//==========================================================================
coFileInfo::coFileInfo(const char *s, int fd)
{
    size_t n = strlen(s);
    name_ = new char[n + 1];
    strcpy(name_, s);
    fd_ = fd;
    pos_ = 0;
    limit_ = 0;
    map_ = NULL;
    buf_ = NULL;
}

coFileInfo::~coFileInfo()
{
    delete[] name_;
}

//==========================================================================
//
//==========================================================================
coFile::coFile(coFileInfo *i)
{
    rep_ = i;
}

//==========================================================================
//
//==========================================================================
coFile::~coFile()
{
    close();
    delete rep_;
}

//==========================================================================
//
//==========================================================================
const char *coFile::name() const
{
    return rep_->name_;
}

//==========================================================================
//
//==========================================================================
long coFile::length() const
{
    return rep_->info_.st_size;
}

bool coFile::exists(const char *fname)
{
#if defined(WIN32) || defined(WIN64)
    // windows specific
    HANDLE h;
    h = CreateFile(fname, STANDARD_RIGHTS_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h != INVALID_HANDLE_VALUE)
    {
        CloseHandle(h);
        return TRUE;
    }
    return FALSE;

#else
    // unix
    if (access(fname, F_OK) == 0)
        return TRUE;

    return FALSE;
#endif
}

//==========================================================================
//
//==========================================================================
void coFile::close()
{
    coFileInfo *i = rep_;
    if (i->fd_ >= 0)
    {
        if (i->map_ != NULL)
        {
#ifdef __sgi
            munmap(i->map_, int(i->info_.st_size));
#endif
        }
        if (i->buf_ != NULL)
        {
            delete i->buf_;
        }
        ::close(i->fd_);
        i->fd_ = -1;
    }
}

//==========================================================================
//
//==========================================================================
void coFile::limit(unsigned int buffersize)
{
    rep_->limit_ = buffersize;
}

//==========================================================================
//
//==========================================================================
coFileInfo *coFile::rep() const
{
    return rep_;
}

#if 0
// class coInputFile

//==========================================================================
//
//==========================================================================
coInputFile::coInputFile(coFileInfo* i) : coFile(i)
{

}


//==========================================================================
//
//==========================================================================
coInputFile::~coInputFile()
{

}


//==========================================================================
//
//==========================================================================
coInputFile* coInputFile::open(const char* name)
{

   int fd = ::open((char*)name, O_RDONLY);
   if (fd < 0)
   {
      return NULL;
   }
   coFileInfo* i = new coFileInfo(name, fd);
   if (fstat(fd, &i->info_) < 0)
   {
      delete i;
      return NULL;
   }
   return new coInputFile(i);
}


//==========================================================================
//
//==========================================================================
long coInputFile::read(const char*& start)
{
#ifdef __sgi
   coFileInfo* i = rep();
   long len = i->info_.st_size;
   if (i->pos_ >= len)
   {
      return 0;
   }
   if (i->limit_ != 0 && ((unsigned int)len) > i->limit_)
   {
      len = i->limit_;
   }
   i->map_ = (char*)mmap(
         0, (int)len, PROT_READ, MAP_PRIVATE, i->fd_, i->pos_
         );
   if (i->map_ == MAP_FAILED)
   {
      return -1;
   }
   start = i->map_;
   i->pos_ += len;
   return len;
#else
   (void) start;
   return(-1);
#endif
}
#endif

#if 0
// class StdInput

//==========================================================================
//
//==========================================================================
coStdInput::coStdInput() : coInputFile(new coFileInfo("-stdin", 0))
{

}


//==========================================================================
//
//==========================================================================
coStdInput::~coStdInput()
{

}


//==========================================================================
//
//==========================================================================
long coStdInput::length() const
{
   return -1;

}


//==========================================================================
//
//==========================================================================
long coStdInput::read(const char*& start)
{
   coFileInfo* i = rep();
   if (i->buf_ == NULL)
   {
      if (i->limit_ == 0)
         i->limit_ = BUFSIZ;
      i->buf_ = new char[i->limit_];
   }
   long nbytes = ::read(i->fd_, (char*)i->buf_, i->limit_);
   if (nbytes > 0)
      start = (const char*)(i->buf_);
   return nbytes;
}
#endif

//
// Hash table support
//

namespace covise
{
//==========================================================================
//
//==========================================================================
unsigned long key_to_hash(const char *str)
{
    unsigned long v = 0;
    for (const char *p = str; *p != '\0'; p++)
    {
        v = (v << 1) ^ (*p);
    }
    unsigned long t = v >> 10;
    t ^= (t >> 10);
    return v ^ t;
}

//==========================================================================
//
//==========================================================================
unsigned long key_to_hash(const char *str, unsigned int /*length*/)
{
    unsigned long v = 0;
    //const char* q = &str[length];
    for (const char *p = str; *p != '\0'; p++)
    {
        v = (v << 1) ^ (*p);
    }
    unsigned long t = v >> 10;
    t ^= (t >> 10);
    return v ^ t;
}

//==========================================================================
//
//==========================================================================
int key_equal(const char *k1, const char *k2)
{
    return strcmp(k1, k2) == 0;
}
}
