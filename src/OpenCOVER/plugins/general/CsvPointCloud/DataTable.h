#ifndef COVISE_OCT_DATA_TABLE_H
#define COVISE_OCT_DATA_TABLE_H

#include "exprtk.hpp"
#include <map>
#include <string>
#include <vector>
#include <iostream>

template <typename T>
T read(std::ifstream &f)
{
    T t;
    f.read((char *)&t, sizeof(T));
    return t;
}

template <typename T>
void write(std::ofstream &f, const T &t)
{
    f.write((const char *)&t, sizeof(T));
}

std::string readString(std::ifstream &f);

void writeString(std::ofstream &f, const std::string &s);


class DataTable
{
public:
    typedef exprtk::symbol_table<float> symbol_table_t;
    DataTable(const std::string &filename, const std::string& timeScaleIndicator, char delimiter, int headerOffset);
    DataTable(const std::string &binaryFile);
    size_t size() const;
    void advance();
    void reset();
    symbol_table_t &symbols();
    void writeToFile(const std::string &filename) const;

private:
    struct Vector
    {

        struct Iterator
        {
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = float;
            using pointer = const value_type *;   // or also value_type*
            using reference = const value_type &; // or also value_type&

            Iterator() = default;
            Iterator(pointer ptr, size_t stride);
            value_type operator*() const;
            pointer operator->();

            // Prefix increment
            Iterator &operator++();

            // Postfix increment
            Iterator operator++(int);

            friend bool operator==(const Iterator &a, const Iterator &b) { return a.m_ptr == b.m_ptr; };
            friend bool operator!=(const Iterator &a, const Iterator &b) { return a.m_ptr != b.m_ptr; };

        private:
            size_t m_stride = 1;
            size_t m_currStride = 0;
            pointer m_ptr = nullptr;
        };

        Iterator begin() const;
        Iterator end() const;

        std::vector<float> data;
        size_t stride = 1;
    };

    const std::map<std::string, Vector> m_data;//keep at top
    const size_t m_size; 
    std::vector<Vector::Iterator> m_currentPos;
    std::vector<float> m_currentValues;
    symbol_table_t m_symbols;

    DataTable(const std::map<std::string, Vector> data);

    //read csv style file filename
    //timeScaleIndicator indicates the timestep in which the following data fields are recorded, overwritetn with the next occurence of such an indicator
    // Vector contains a data field and its iterator is made so that it uses the previous entry if this data field has a wider timescale
    std::map<std::string, Vector> readFile(const std::string &filename, const std::string& timeScaleIndicator, char delimiter, int headerOffset);
    std::map<std::string, Vector> readBinaryFile(const std::string &filename);
};

#endif