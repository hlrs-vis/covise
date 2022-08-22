#include "DataTable.h"
#include <iostream>
#include <cassert>
size_t sizeTmp = 0;

std::map<std::string, DataTable::Vector> DataTable::readFile(const std::string &filename, const std::string& timeScaleIndicator, char delimiter)
{
    sizeTmp = 0;
    constexpr size_t headerOffset = 6;
    std::map<std::string, Vector> points;
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
    if(stringValue.find(delimiter) == std::string::npos)
    {
        std::cerr << "Oct pluging failed to parse file, delimiter " << delimiter << " not found in file" << std::endl;
        return points;
    }
    while (std::getline(headline, stringValue, delimiter)) // read headline
    {
        headlines.push_back(stringValue);
    }

    std::map<size_t, float> timescales;
    size_t numLines = 0;
    while (std::getline(f, stringValue)) // fill data fields line by line
    {

        std::stringstream dataLine(stringValue);
        size_t column = 0;
        bool empty = true;
        while (std::getline(dataLine, stringValue, delimiter)) //read columns
        {
            if(!(stringValue.empty() || std::isspace(stringValue[0])))
            {
                empty = false;
                if (headlines[column] != timeScaleIndicator)
                {
                    auto value = std::stof(stringValue);
                    assert(value == value); //check for NaN
                    points[headlines[column]].data.push_back(value);
                }
                else if(numLines == 0){
                    timescales[column] = std::stof(stringValue);
                }
                else if(numLines == 1){
                    auto &ts = timescales[column];
                    ts = std::stof(stringValue) - ts;
                }
                else if(numLines == 2 && timescales.size() > 1 && column == 0) //we have multiple time scales
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
                            points[headlines[headlineColumn]].stride = lastTsIt->second / min;
                        }
                    }
                    for (size_t headlineColumn = timescales.rbegin()->first + 1; headlineColumn < headlines.size(); headlineColumn++) //add the stride for the rest
                    {
                        points[headlines[headlineColumn]].stride = timescales.rbegin()->second / min;
                    }
                }
            }
            ++column;
        }

        if(!empty)
            ++numLines;
    }
    for(auto &p :points)
        p.second.data.push_back(p.second.data[p.second.data.size() - 1]); //add last element twice for interpolation
    sizeTmp = numLines;
    return points;
}

DataTable::DataTable(const std::string &filename, const std::string& timeScaleIndicator, char delimiter)
: m_data(readFile(filename, timeScaleIndicator, delimiter))
, m_size(sizeTmp)
, m_currentPos(m_data.size())
, m_currentValues(m_data.size())
{
    size_t i = 0;
    for (auto &val : m_data)
    {
        m_currentPos[i] = val.second.begin();
        m_currentValues[i] = *val.second.data.data();
        m_symbols.add_variable(val.first, m_currentValues[i]);
        ++i;
    }
}

size_t DataTable::size() const
{
    return m_size;
}

void DataTable::advance()
{
    for (size_t i = 0; i < m_currentPos.size(); i++)
        m_currentValues[i] = *(++m_currentPos[i]);
}

void DataTable::reset()
{
    m_currentPos.clear();
    m_currentValues.clear();
    for (const auto &field : m_data)
    {
        m_currentPos.push_back(field.second.begin());
        m_currentValues.push_back(field.second.data[0]);
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

DataTable::Vector::Iterator::Iterator(DataTable::Vector::Iterator::pointer ptr, size_t stride)
    : m_ptr(ptr), m_stride(std::max(size_t(1), stride))
{
}

DataTable::Vector::Iterator::value_type DataTable::Vector::Iterator::operator*() const {
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
