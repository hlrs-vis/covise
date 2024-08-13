#ifndef _READ_CSV_H
#define _READ_CSV_H

#include "export.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <array>

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
    typedef std::map<std::string, std::string> CSVRow;
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

    CSVStream &operator>>(CSVRow &row)
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

const std::array<std::string, 2> INVALID_CELL_CONTENT = {"", "NULL"};

/**
 * @brief A utility struct for accessing CSV rows.
 */
struct CSVUTIL AccessCSVRow {
    template<typename T>
    void operator()(const CSVStream::CSVRow &row, const std::string &colName, T &value) const
    {
        try {
            auto value_str = row.at(colName);
            if (value_str.empty() ||
                std::any_of(INVALID_CELL_CONTENT.begin(), INVALID_CELL_CONTENT.end(),
                            [&value_str](const std::string &invalid) { return value_str == invalid; }))
                value_str = "0";
            convert(value_str, value);
        } catch (const std::out_of_range &ex) {
            throw CSVStream_Exception("Column " + colName + " not found in CSV file");
        } catch (const std::invalid_argument &ex) {
            throw CSVStream_Exception("Invalid argument for column " + colName);
        } catch (const std::exception &ex) {
            throw CSVStream_Exception(ex.what());
        }
    }

private:
    template<typename T>
    void convert(const std::string &value_str, T &value) const
    {
        std::istringstream ss(value_str);
        ss >> value;
    }
};

// convert string to T if possible
inline constexpr AccessCSVRow access_CSVRow{};
} // namespace opencover::utils::read

#endif