#include "csv.h"

namespace opencover::utils::read {

CSVStream::CSVStream(const std::string &filename, char delimiter): m_filename(filename), m_delimiter(delimiter)
{
    m_inputFileStream.open(filename.c_str());
    if (!m_inputFileStream.is_open())
        throw CSVStream_Exception("Could not open file " + filename);
    readHeader();
}

void CSVStream::readHeader()
{
    std::string colName("");
    auto ss = getLine();
    while (std::getline(ss, colName, m_delimiter))
        m_header.push_back(colName);
}

void CSVStream::readLine(CSVStream::CSVRow &row)
{
    std::string value("");
    auto ss = getLine();
    size_t currentColNameIdx = 0;
    for  (auto &header : m_header) {
        std::getline(ss, value, m_delimiter);
        row[header] = value;
    }
}

std::stringstream CSVStream::getLine()
{
    std::getline(m_inputFileStream, m_currentline);
    std::stringstream ss(m_currentline);
    return ss;
}
} // namespace opencover::utils::read
