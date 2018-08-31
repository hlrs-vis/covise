#ifndef ARCHIVE_MODEL_H
#define ARCHIVE_MODEL_H

#include <map>
#include <string>
#include <vector>
#include <streambuf>

namespace fs {

class Entry;
class Directory;
class File;
class Model;
class Path;

class Path {
    friend class Model;
    friend class DirectoryIterator;
public:
    Path(const Model &model);
    Path(const Model *model);
    Path(const Model *model, const std::string &path);
    Path(const Entry &entry);

    Path filename() const;
    Path stem() const;
    Path extension() const;
    std::string string() const;

    Path operator+(const std::string &name) const;
    Path operator/(const std::string &name) const;
    Path &operator/=(const std::string &name);

    bool exists() const;
    bool is_directory() const;

    const Model *getModel() const;
private:

    const Model *model = nullptr;
    std::vector<std::string> components;
    bool absolute = true; // prepend '/'
};

class Entry {
    friend class Path;
public:
   Entry(const Directory *parent, std::string name);
   virtual ~Entry();

   bool operator<(const Entry &other) const;
   bool operator<(const std::string &other) const;

   const std::string &string() const;
   Path path() const;
   operator Path() const;

protected:
    Entry(const Model *model);
    const Model *model = nullptr;
    const Directory *parent = nullptr;
    std::string name;
};

class File: public Entry {
    friend class Model;
public:
   File(const Directory *dir, const std::string &name);
   size_t size = 0;
   size_t offset = 0;
};

class Directory: public Entry {
    friend class DirectoryIterator;
    friend class Model;
public:
   Directory(const Directory *dir, const std::string &name);
   Directory(const Model *model);

   Directory *addDirectory(const std::string &name);
   File *addFile(const std::string &name);

   Directory *findDirectory(const std::string &name);
   const Directory *findDirectory(const std::string &name) const;
   File *findFile(const std::string &name);
   const File *findFile(const std::string &name) const;
   Entry *find(const std::string &name);
   const Entry *find(const std::string &name) const;

private:
   std::map<std::string,Directory> dirs;
   std::map<std::string,File> files;
};

class DirectoryIterator {
public:
    DirectoryIterator();
    DirectoryIterator(const Directory &dir);
    DirectoryIterator(const Path &path);
    DirectoryIterator &operator=(const DirectoryIterator &other);
    bool operator==(const DirectoryIterator &other) const;
    bool operator!=(const DirectoryIterator &other) const;
    DirectoryIterator &operator++();
    DirectoryIterator operator++(int);
    const Entry &operator*() const;
    const Entry *operator->() const;

private:
    const Directory *dir = nullptr;
    std::map<std::string,Directory>::const_iterator dit;
    std::map<std::string,File>::const_iterator fit;
};

class Model {
   friend class Path;
public:

   Model(const std::string &archiveOrDirectory);

   bool isDirectory(const std::string &path) const;
   bool isDirectory(const Path &path) const;
   bool exists(const Path &path, bool requireDirectory=false) const;
   const Entry *findEntry(const Path &path) const;
   const Directory *findDirectory(const Path &path) const;
   operator Directory() const;
   const File *findFile(const std::string &pathname) const;
   const std::string &getContainer() const;

private:
   const Entry *findEntry(const std::vector<std::string> &pathcomponents) const;
   Entry *addPath(const std::string &path);

   bool archive = false;
   std::string container;
   Directory root;
};

bool is_directory(const Entry &entry);
bool is_directory(const Path &path);
bool exists(const Path &path);

} // namespace fs

#endif
