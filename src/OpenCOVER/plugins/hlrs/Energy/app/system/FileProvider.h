#pragma once
#include "Provider.h"
#include "DataPackage.h"

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
    std::string createFilePath(int scenarioId, EnergyType type, const std::string &filename = "");
    std::vector<std::string> createFilePaths(int scenarioID, EnergyType type, const std::string &extension);

private:
    std::vector<std::string> discoverFiles(const std::string &dirPath, const std::string &extension);

    std::string m_dirPath;
};
