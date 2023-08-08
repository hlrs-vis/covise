#include "archivemodel.h"

#ifdef HAVE_LIBARCHIVE
#include <archive.h>
#include <archive_entry.h>
#endif

#ifdef HAVE_LIBZIP
#include <zip.h>
#endif

#include <boost/algorithm/string/split.hpp>

#include <algorithm>
#include <exception>
#include <iostream>

namespace {

struct Cleaner {
    Cleaner(std::function<void()> task): task(std::move(task)) {}

    virtual ~Cleaner() { clean(); }

    void clean()
    {
        if (task)
            task();
        reset();
    }

    void reset(std::function<void()> ntask = std::function<void()>()) { task = std::move(ntask); }
    std::function<void()> task;
};

} // namespace

namespace fs {

Directory::Directory(const Directory *dir, const std::string &name): Entry(dir, name)
{}

Directory::Directory(const Model *m): Entry(m)
{}


Directory *Directory::findDirectory(const std::string &name)
{
    auto it = dirs.find(name);
    if (it == dirs.end())
        return nullptr;
    return &it->second;
}
const Directory *Directory::findDirectory(const std::string &name) const
{
    auto it = dirs.find(name);
    if (it == dirs.end())
        return nullptr;
    return &it->second;
}

File *Directory::findFile(const std::string &name)
{
    auto it = files.find(name);
    if (it == files.end())
        return nullptr;
    return &it->second;
}
const File *Directory::findFile(const std::string &name) const
{
    auto it = files.find(name);
    if (it == files.end())
        return nullptr;
    return &it->second;
}

Entry *Directory::find(const std::string &name)
{
    if (auto ent = findDirectory(name))
        return ent;
    return findFile(name);
}
const Entry *Directory::find(const std::string &name) const
{
    if (auto ent = findDirectory(name))
        return ent;
    return findFile(name);
}

Directory *Directory::addDirectory(const std::string &name)
{
    if (auto d = findDirectory(name))
        return d;
    if (findFile(name))
        return nullptr;
    auto result = dirs.emplace(name, Directory(this, name));
    return &result.first->second;
}

File *Directory::addFile(const std::string &name)
{
    if (auto f = findFile(name))
        return f;
    if (findDirectory(name))
        return nullptr;
    auto result = files.emplace(name, File(this, name));
    return &result.first->second;
}

File::File(const Directory *dir, const std::string &name): Entry(dir, name)
{}

Entry::Entry(const Directory *parent, std::string name): model(parent->model), parent(parent), name(std::move(name))
{}

Entry::Entry(const Model *model): model(model)
{}

Entry::~Entry() = default;

bool Entry::operator<(const std::string &other) const
{
    return name < other;
}

const std::string &Entry::string() const
{
    return name;
}

Path Entry::path() const
{
    if (!parent)
        return Path(model, name);

    return parent->path() / name;
}

fs::Entry::operator Path() const
{
    return path();
}

bool Entry::operator<(const Entry &other) const
{
    return name < other.name;
}

Path::Path(const Model &model): model(&model)
{}

Path::Path(const Model *model): model(model)
{}

Path::Path(const Model *model, const std::string &path): model(model)
{
    if (path.empty()) {
        absolute = false;
        return;
    }
    if (path.front() != '/')
        absolute = false;
    std::vector<std::string> c;
    boost::split(
        c, path, [](char c) { return c == '/'; }, boost::token_compress_on);
    std::copy(c.begin(), c.end(), std::back_inserter(components));
}

Path::Path(const Entry &entry): model(entry.model)
{
    std::vector<std::string> c;
    for (const Entry *e = &entry; e; e = e->parent) {
        c.emplace_back(e->string());
        if (e->parent == &model->root) {
            absolute = true;
            break;
        }
    }

    std::copy(c.rbegin(), c.rend(), std::back_inserter(components));
}

Path Path::filename() const
{
    if (components.empty())
        return Path(model);
    return Path(model, components.back());
}

Path Path::stem() const
{
    auto fn = filename().string();
    auto pos = fn.find('.');
    if (pos == std::string::npos)
        return Path(model, fn);

    return Path(model, fn.substr(0, pos));
}

Path Path::extension() const
{
    auto fn = filename().string();
    auto pos = fn.find('.');
    if (pos == std::string::npos)
        return Path(model);

    return Path(model, fn.substr(pos));
}

std::string Path::string() const
{
    std::string path;
    for (auto it = components.begin(); it != components.end(); ++it) {
        if (it != components.begin())
            path += "/";
        path += *it;
    }
    return path;
}

Path Path::operator+(const std::string &name) const
{
    Path path(*this);
    if (path.components.empty()) {
        path /= name;
    } else {
        path.components.back() += name;
    }
    return path;
}

Path Path::operator/(const std::string &name) const
{
    Path path(*this);
    return path /= name;
}

Path &Path::operator/=(const std::string &name)
{
    if (!name.empty())
        components.emplace_back(name);
    return *this;
}

bool Path::is_directory() const
{
    return model->isDirectory(*this);
}

const Model *Path::getModel() const
{
    return model;
}

bool Path::exists() const
{
    return model->exists(*this);
}

class ModelPrivate {
    friend class Model;
    friend class ::archive_streambuf;

public:
    ~ModelPrivate()
    {
#ifdef HAVE_LIBZIP
        if (zipfile)
            zip_close(zipfile);
#endif
    }

#ifdef HAVE_LIBZIP
    struct zip *zipfile = nullptr;
#endif
};

Model::Model(const std::string &archiveOrDirectory, Format format)
: format(format), container(archiveOrDirectory), root(this)
{
    d.reset(new ModelPrivate);
#ifdef HAVE_LIBZIP
    d->zipfile = zip_open(archiveOrDirectory.c_str(), 0, nullptr);
    if (d->zipfile) {
        archive = true;
        format = FormatZip;
        int64_t nument = zip_get_num_entries(d->zipfile, 0);
        for (int64_t idx = 0; idx < nument; ++idx) {
            std::string pathname = zip_get_name(d->zipfile, idx, 0);
            if (auto file = dynamic_cast<File *>(addPath(pathname))) {
                file->index = idx;
            }
        }
    } else if (format == FormatZip) {
        std::cerr << "failed to open archive with libzip: " << archiveOrDirectory << std::endl;
    }
#else
    if (format == FormatZip) {
        throw std::runtime_error("failed to open archive " + archiveOrDirectory +
                                 " in zip format: not compiled with libzip");
    }
#endif

    if (!archive) {
#ifndef HAVE_LIBARCHIVE
        throw std::runtime_error("failed to open archive " + archiveOrDirectory + ": not compiled with libarchive");
#else
        archive = true;
        struct archive *a = archive_read_new();
        Cleaner archiveCleaner([a]() { archive_read_free(a); });

        archive_read_support_format_zip(a);
        archive_read_support_format_tar(a);
        int r = archive_read_open_filename(a, archiveOrDirectory.c_str(), 102400);
        if (r != ARCHIVE_OK) {
            throw std::runtime_error("failed to open archive " + archiveOrDirectory);
        }

        File *prev = nullptr;
        struct archive_entry *entry = nullptr;
        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            if (prev) {
                prev->offset = (archive_read_header_position(a) - prev->size) & ~(int64_t)0x1ff;
                //std::cerr << prev->path().string() << " OFFSET: " << prev->offset << std::endl;
            }
            std::string pathname = archive_entry_pathname(entry);
            auto type = archive_entry_filetype(entry);
            prev = nullptr;
            if (type == AE_IFDIR) {
                /* auto dir = dynamic_cast<Directory *>(addPath(pathname)); */
            } else if (type == AE_IFREG) {
                auto file = dynamic_cast<File *>(addPath(pathname));
                if (file) {
                    size_t sz = archive_entry_size(entry);
                    file->size = sz;
                    int64_t off = archive_read_header_position(a);
                    file->offset = off;
                    //std::cerr << file->path().string() << " OFFSET: " << file->offset  << "+" << file->size << std::endl;
                }
                prev = file;
            }
            //archive_read_data_skip(a); // not necessary, and might be detrimental to performance
        }
        if (prev) {
            prev->offset = (archive_read_header_position(a) - prev->size) & ~(int64_t)0x1ff;
            //std::cerr << prev->path().string() << " OFFSET: " << prev->offset << std::endl;
        }
#endif
    }
}

Model::operator Directory() const
{
    return root;
}

Entry *Model::addPath(const std::string &path)
{
    std::vector<std::string> components;
    boost::split(
        components, path, [](char c) { return c == '/'; }, boost::token_compress_on);
    Directory *dir = &root;
    for (size_t i = 0; i + 1 < components.size(); ++i) {
        File *f = dir->findFile(components[i]);
        if (f) {
            std::cerr << "already have file of same name" << std::endl;
            return nullptr;
        }
        Directory *d = dir->findDirectory(components[i]);
        if (!d) {
            d = dir->addDirectory(components[i]);
        }
        dir = d;
    }
    if (isDirectory(path)) {
        Directory *d = dir->addDirectory(components.back());
        d->pathname = path;
        return d;
    }

    File *f = dir->addFile(components.back());
    f->pathname = path;
    return f;
}

bool Model::isDirectory(const std::string &path) const
{
    if (path.empty())
        return true;

    return path.back() == '/';
}

bool Model::exists(const Path &path, bool requireDirectory) const
{
    if (path.components.empty())
        return true;

    const Directory *dir = &root;
    for (size_t i = 0; i + 1 < path.components.size(); ++i) {
        const auto &c = path.components[i];
        auto d = dir->findDirectory(c);
        if (!d)
            return false;
        dir = d;
    }

    return requireDirectory ? dir->findDirectory(path.components.back()) : dir->find(path.components.back());
}

const Entry *Model::findEntry(const Path &path) const
{
    return findEntry(path.components);
}

const Entry *Model::findEntry(const std::vector<std::string> &pathcomponents) const
{
    if (pathcomponents.empty())
        return &root;

    const Directory *dir = &root;
    for (size_t i = 0; i + 1 < pathcomponents.size(); ++i) {
        const auto &c = pathcomponents[i];
        auto d = dir->findDirectory(c);
        if (!d)
            return nullptr;
        dir = d;
    }

    return dir->find(pathcomponents.back());
}

const Directory *Model::findDirectory(const Path &path) const
{
    return dynamic_cast<const Directory *>(findEntry(path));
}

const File *Model::findFile(const std::string &pathname) const
{
    std::vector<std::string> components;
    boost::split(
        components, pathname, [](char c) { return c == '/'; }, boost::token_compress_on);
    auto e = findEntry(components);
    return dynamic_cast<const File *>(e);
}

const std::string &Model::getContainer() const
{
    return container;
}

bool Model::isDirectory(const Path &path) const
{
    return exists(path, true);
}


DirectoryIterator::DirectoryIterator() = default;

DirectoryIterator::DirectoryIterator(const Directory &dir): dir(&dir), dit(dir.dirs.begin()), fit(dir.files.begin())
{}

DirectoryIterator::DirectoryIterator(const Path &path)
{
    dir = path.model->findDirectory(path);
    if (dir) {
        dit = dir->dirs.begin();
        fit = dir->files.begin();
    }
}

DirectoryIterator &DirectoryIterator::operator=(const DirectoryIterator &other) = default;

bool DirectoryIterator::operator==(const DirectoryIterator &other) const
{
    if (dir != other.dir)
        return false;
    if (dir == nullptr)
        return true;
    if (dit != other.dit)
        return false;
    if (fit != other.fit)
        return false;
    return true;
}

bool DirectoryIterator::operator!=(const DirectoryIterator &other) const
{
    return !(*this == other);
}

DirectoryIterator DirectoryIterator::operator++(int)
{
    auto ret = *this;
    ++*this;
    return ret;
}

const Entry &DirectoryIterator::operator*() const
{
    assert(dir);
    if (dit != dir->dirs.end())
        return dit->second;
    assert(fit != dir->files.end());
    return fit->second;
}

const Entry *DirectoryIterator::operator->() const
{
    assert(dir);
    if (dit != dir->dirs.end())
        return &dit->second;
    assert(fit != dir->files.end());
    return &fit->second;
}

DirectoryIterator &DirectoryIterator::operator++()
{
    assert(dir);
    if (dit != dir->dirs.end()) {
        ++dit;
        if (dit == dir->dirs.end() && fit == dir->files.end())
            dir = nullptr;
    } else if (fit != dir->files.end()) {
        ++fit;
        if (fit == dir->files.end())
            dir = nullptr;
    }
    return *this;
}

bool is_directory(const Entry &e)
{
    return dynamic_cast<const Directory *>(&e) != nullptr;
}

bool is_directory(const Path &path)
{
    return path.is_directory();
}

bool exists(const Path &path)
{
    return path.exists();
}

} // namespace fs


archive_streambuf::archive_streambuf(const fs::File *file)
{
    auto end = buf + sizeof(buf);
    setg(end, end, end);

    auto &container = file->model->container;

#ifdef HAVE_LIBZIP
    if (file->index >= 0) {
        auto &zipfile = file->model->d->zipfile;
        zip = zip_fopen_index(zipfile, file->index, 0);
    } else
#endif
    {
#ifdef HAVE_LIBARCHIVE
        auto a = archive_read_new();
        archive = a;
        archive_read_support_format_tar(a);
        archive_read_support_format_zip(a);
        int r = archive_read_open_filename(a, container.c_str(), sizeof(buf));
        if (r != ARCHIVE_OK) {
            archive_read_free(a);
            throw std::runtime_error("failed to open archive " + container);
        }

        struct archive_entry *entry = nullptr;
        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            std::string pathname = archive_entry_pathname(entry);
            if (pathname == file->pathname) {
#ifndef NDEBUG
                auto type = archive_entry_filetype(entry);
                assert(type == AE_IFREG);
#endif
                /* size_t sz = archive_entry_size(entry); */
                return;
            }
        }

        archive_read_free(a);
        archive = nullptr;
        throw std::runtime_error("did not find " + file->pathname + " in archive " + container);
#else
        throw std::runtime_error("cannot read from " + file->pathname + " from " + container +
                                 ": not compiled with libarchive");
#endif
    }
}

archive_streambuf::~archive_streambuf()
{
#ifdef HAVE_LIBZIP
    if (zip) {
        auto z = static_cast<zip_file *>(zip);
        zip_fclose(z);
    } else
#endif
    {
#ifdef HAVE_LIBARCHIVE
        auto a = static_cast<struct archive *>(archive);
        archive_read_close(a);
        archive_read_free(a);
        archive = nullptr;
#endif
    }
}

std::streambuf::int_type archive_streambuf::underflow()
{
#ifdef HAVE_LIBZIP
    if (zip) {
        auto z = static_cast<zip_file *>(zip);
        auto n = zip_fread(z, buf, sizeof(buf));
        if (n < 0) {
            return EOF;
        }
        if (n == 0)
            return EOF;
        nread += n;
        setg(buf, buf, buf + n);
    } else
#endif
    {
#ifdef HAVE_LIBARCHIVE
        auto a = static_cast<struct archive *>(archive);
        auto n = archive_read_data(a, buf, sizeof(buf));
        if (n == 0)
            return EOF;
        nread += n;
        setg(buf, buf, buf + n);
#endif
    }
    return traits_type::to_int_type(*gptr());
}
