#include "DataTable.h"
#include <iostream>
#include <cassert>
#include <cover/coVRMSController.h>
#include <util/string_util.h>

size_t sizeTmp = 0; //set by file readers and used to initialize const m_size

std::string readString(std::ifstream &f)
{
    std::string str;
    size_t size;
    f.read((char *)&size, sizeof(size));
    str.resize(size);
    f.read(&str[0], size);
    return str;
}

void writeString(std::ofstream &f, const std::string &s)
{
    size_t size = s.size();
    f.write((const char *)&size, sizeof(size));
    f.write(&s[0], size);
}

DataTable::DataTable(const std::string &filename, const std::string &timeScaleIndicator, char delimiter, int headerOffset)
    : DataTable(readFile(filename, timeScaleIndicator, delimiter, headerOffset))
{
}

DataTable::DataTable(const std::string &binaryFile)
    : DataTable(readBinaryFile(binaryFile))
{
}

DataTable::DataTable(const DataTable& other)
    :DataTable(other.m_data)
{

}

DataTable::DataTable(const  std::shared_ptr<const std::map<std::string, Vector>> &data)
    : m_data(data), m_size(sizeTmp), m_currentValues(m_data->size())
{
    size_t i = 0;
    for (auto &val : *m_data)
    {
        m_currentValues[i] = val.second.data[0];
        m_symbols.add_variable(val.first, m_currentValues[i]);
        ++i;
    }
}

size_t DataTable::size() const
{
    return m_size;
}

void DataTable::setCurrentValues(size_t index)
{
    size_t i = 0;
    for (auto& val : *m_data)
    {
        m_currentValues[i] = val.second[index];
        ++i;
    }
}


DataTable::symbol_table_t &DataTable::symbols()
{
    return m_symbols;
}

DataTable::Vector::Iterator DataTable::Vector::begin() const
{
    return Iterator(data.data(), stride);
}

DataTable::Vector::Iterator DataTable::Vector::end() const
{
    return Iterator{data.data() + data.size(), stride};
}

float DataTable::Vector::operator[](size_t index) const
{
    size_t pos = std::floor(index / stride);
    auto diff = index % stride;
    auto retval = data[pos];
    if(diff)
        retval +=(data[pos + 1] - data[pos]) * diff / stride;
    return retval;
}


DataTable::Vector::Iterator::Iterator(DataTable::Vector::Iterator::pointer ptr, size_t stride)
    : m_ptr(ptr), m_stride(std::max(size_t(1), stride))
{
}

DataTable::Vector::Iterator::value_type DataTable::Vector::Iterator::operator*() const
{
    float currValue = *m_ptr;
    float nextValue = *(m_ptr + 1);
    return currValue + (nextValue - currValue) * m_currStride / m_stride;
}
DataTable::Vector::Iterator::pointer DataTable::Vector::Iterator::operator->() { return m_ptr; }

DataTable::Vector::Iterator &DataTable::Vector::Iterator::operator++()
{
    ++m_currStride;
    if (m_currStride == m_stride)
    {
        m_currStride = 0;
        m_ptr++;
    }
    return *this;
}

DataTable::Vector::Iterator DataTable::Vector::Iterator::operator++(int)
{
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

std::shared_ptr<const std::map<std::string, DataTable::Vector>> DataTable::readFile(const std::string &filename, const std::string &timeScaleIndicator, char delimiter, int headerOffset)
{
    sizeTmp = 0;
    auto points = std::make_shared<std::map<std::string, Vector>>();
    std::fstream f(filename);
    std::string stringValue;
    for (size_t i = 0; i < headerOffset; i++) // skip header
    {
        if (!std::getline(f, stringValue))
            return points; // bail out
    }
    if (!std::getline(f, stringValue))
        return points; // bail out
    std::vector<std::string> headlines;
    std::stringstream headline(stringValue);
    if (stringValue.find(delimiter) == std::string::npos)
    {
        std::cerr << "CsvPointcloud plugin failed to parse file: delimiter " << delimiter << " not found in file" << std::endl;
        return points;
    }
    while (std::getline(headline, stringValue, delimiter)) // read headline
    {
        if(stringValue != timeScaleIndicator && std::find(headlines.begin(), headlines.end(), stringValue) != headlines.end())
        {
            std::cerr << "CsvPointcloud plugin failed to parse file: some columns have the same headline " << stringValue << std::endl;
            return points;
        }
        headlines.push_back(stringValue);
    }

    std::map<size_t, float> timescales;
    size_t numLines = 0;
    while (std::getline(f, stringValue)) // fill data fields line by line
    {

        std::stringstream dataLine(stringValue);
        size_t column = 0;
        size_t numFullColumns = 0;
        bool empty = true;
        while (std::getline(dataLine, stringValue, delimiter)) // read columns
        {
            if (!(stringValue.empty() || std::isspace(stringValue[0])))
            {
                ++numFullColumns;
                empty = false;
                if (headlines[column] != timeScaleIndicator)
                {
                    auto value = std::stof(stringValue);
                    assert(value == value); // check for NaN
                    (*points)[headlines[column]].data.push_back(value);
                }
                else if (numLines == 0)
                {
                    timescales[column] = std::stof(stringValue);
                }
                else if (numLines == 1)
                {
                    auto &ts = timescales[column];
                    ts = std::stof(stringValue) - ts;
                }
                else if (numLines == 2 && timescales.size() > 1 && column == 0) // we have multiple time scales
                {
                    float min = timescales.begin()->second;
                    for (const auto &ts : timescales)
                        min = std::min(min, ts.second);
                    for (auto tsIt = ++timescales.begin(); tsIt != timescales.end(); tsIt++)
                    {
                        auto lastTsIt = tsIt;
                        --lastTsIt;
                        for (size_t headlineColumn = lastTsIt->first + 1; headlineColumn < tsIt->first; headlineColumn++)
                        {
                            (*points)[headlines[headlineColumn]].stride = lastTsIt->second / min;
                        }
                    }
                    for (size_t headlineColumn = timescales.rbegin()->first + 1; headlineColumn < headlines.size(); headlineColumn++) // add the stride for the rest
                    {
                        (*points)[headlines[headlineColumn]].stride = timescales.rbegin()->second / min;
                    }
                }
            }
            ++column;
        }
        if (!empty)
            ++numLines;
    }
    sizeTmp = numLines;
    for (auto &p : *points)
        sizeTmp = std::min(sizeTmp, p.second.data.size() * p.second.stride);

    for (auto &p : *points)
        p.second.data.push_back(p.second.data[p.second.data.size() - 1]); // add last element twice for interpolation

    return points;
}

std::shared_ptr<const std::map<std::string, DataTable::Vector>> DataTable::readBinaryFile(const std::string &filename)
{
    std::ifstream f(filename, std::ios::binary);
    auto data = std::make_shared<std::map<std::string, Vector>>();
    sizeTmp = read<size_t>(f);
    auto size = read<size_t>(f);
    for (size_t i = 0; i < size; i++)
    {
        auto &column = (*data)[readString(f)];
        column.stride = read<size_t>(f);
        auto size = read<size_t>(f);
        column.data.resize(size);
        f.read((char *)&column.data[0], size * sizeof(float));
    }
    return data;
}

void DataTable::writeToFile(const std::string &filename) const
{
    if (opencover::coVRMSController::instance()->isMaster())
    {
    
        std::ofstream f(filename, std::ios::binary);
        write(f, m_size);
        write(f, m_data->size());
        for (const auto &column : *m_data)
        {
            writeString(f, column.first);
            write(f, column.second.stride);
            write(f, column.second.data.size());
            f.write((const char *)&column.second.data[0], column.second.data.size() * sizeof(float));
        }
    }
}
