#ifndef _READ_CSV_H
#define _READ_CSV_H

#include "export.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

namespace opencover::utils::read {

class CSVUTIL CSVStream_Exception: public std::exception {
public:
    CSVStream_Exception(const std::string &msg): msg(msg) {}
    [[nodiscard]] const char *what() const noexcept override { return msg.c_str(); }

private:
    const std::string msg;
};

/**
 * @class CSV
 * @brief A utility class for reading CSV files.
 */
class CSVUTIL CSVStream {
public:
    CSVStream(const std::string &filename, char delimiter = ','): m_filename(filename), m_delimiter(delimiter)
    {
        m_inputFileStream.open(filename.c_str());
        if (!m_inputFileStream.is_open())
            throw CSVStream_Exception("Could not open file " + filename);
        readHeader();
    }

    ~CSVStream()
    {
        if (m_inputFileStream.is_open())
            m_inputFileStream.close();
    }

    CSVStream(const CSVStream &) = delete;
    CSVStream &operator=(const CSVStream &) = delete;

    CSVStream &operator>>(std::map<std::string, std::string> &row)
    {
        readLine(row);
        return *this;
    }

    explicit operator bool() const { return m_inputFileStream.good(); }

    const std::vector<std::string> &getHeader() { return m_header; }

private:
    void readHeader()
    {
        std::string colName("");
        auto ss = getLine();
        while (std::getline(ss, colName, m_delimiter))
            m_header.push_back(colName);
    }

    void readLine(std::map<std::string, std::string> &row)
    {
        std::string value("");
        auto ss = getLine();
        size_t currentColNameIdx = 0;
        while (std::getline(ss, value, m_delimiter) && currentColNameIdx < m_header.size()) {
            row[m_header[currentColNameIdx]] = value;
            ++currentColNameIdx;
        }
    }

    std::stringstream getLine()
    {
        std::getline(m_inputFileStream, m_currentline);
        std::stringstream ss(m_currentline);
        return ss;
    }

    std::string m_filename;
    std::ifstream m_inputFileStream;
    std::vector<std::string> m_header;
    std::string m_currentline;

    char m_delimiter;
};
} // namespace opencover::utils::read

#endif