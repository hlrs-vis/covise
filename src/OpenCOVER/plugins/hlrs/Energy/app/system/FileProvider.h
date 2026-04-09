#pragma once
#include "Provider.h"
#include "DataPackage.h"
#include "Scenario.h"

#include <vector>

class FileProvider : public Provider<DataPackages>
{
public:
    FileProvider(const std::string &dirPath) : m_dirPath(dirPath) {};
    virtual ~FileProvider() = default;
    FileProvider(const FileProvider &) = delete;
    FileProvider(FileProvider &&) = delete;
    FileProvider &operator=(const FileProvider &) = delete;
    FileProvider &operator=(FileProvider &&) = delete;

protected:
    std::string createFilePath(const Scenario &scenario, EnergyType type, const std::string &filename = "") const;
    std::vector<std::string> createFilePaths(const Scenario &scenario, EnergyType type, const std::string &extension) const;

private:
    std::vector<std::string> discoverFiles(const std::string &dirPath, const std::string &extension) const;

    std::string m_dirPath;
};
