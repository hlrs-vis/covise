#include "FileProvider.h"

#include <filesystem>
#include <algorithm>

std::string FileProvider::createFilePath(const Scenario &scenario, EnergyType type, const std::string &filename) const
{
    auto folder = (scenario.id == -1) ? "static" : scenario.name;
    std::string typeFolder(EnergyTypeToString(type));
    std::string filePath = m_dirPath + "/" + folder + "/" + typeFolder;

    if (filename != "")
        filePath += "/" + filename;
    return filePath;
}

std::vector<std::string> FileProvider::createFilePaths(const Scenario &scenario, EnergyType type, const std::string &extension) const
{
    auto filePath = createFilePath(scenario, type);
    auto files = discoverFiles(filePath, extension);
    std::for_each(files.begin(), files.end(), [&](auto &file){ createFilePath(scenario, type, file); });
    return files;
}

std::vector<std::string> FileProvider::discoverFiles(const std::string &dirPath, const std::string &extension) const
{
    std::vector<std::string> files(0);
    if (std::filesystem::exists(dirPath))
        for (const auto &entry : std::filesystem::directory_iterator(dirPath))
            if (entry.is_regular_file() && entry.path().extension() == extension)
                files.emplace_back(entry.path().filename().string());
    return files;
}
