#ifndef _READ_CSV_H
#define _READ_CSV_H

#ifdef _WIN32
#include <boost/filesystem.hpp>
#else
#include <boost/filesystem.hpp>
#include <boost/filesystem/directory.hpp>
#endif
#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <mutex>  // Keep mutex for thread-safe line reading if m_parallel is true
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "export.h"

namespace opencover::utils::read {

class CSVUTIL CSVStream_Exception : public std::exception {
 public:
  CSVStream_Exception(const std::string &msg) : msg(msg) {}
  [[nodiscard]] const char *what() const noexcept override { return msg.c_str(); }

 private:
  const std::string msg;
};

/**
 * @class CSVStream
 * @brief A utility class for reading CSV files.
 *
 * Modified for parallel processing using a producer-consumer pattern.
 *
 * Usage for sequential reading (e.g., by a producer thread):
 * auto stream = CSVStream("banane.csv");
 * CSVStream::CSVRow row;
 * while (stream.readNextRow(row)) // Use new method for reading a row
 * std::cout << row["spaltenName"] << "\n";
 *
 * For parallel processing, a producer thread uses `getLine()` to retrieve raw lines,
 * and consumer threads use `parseLine()` to parse them concurrently.
 */
class CSVUTIL CSVStream {
 public:
  typedef std::map<std::string, std::string> CSVRow;
  // 'parallel' in constructor now indicates if `getLine` needs internal mutex
  // protection.
  CSVStream(const std::string &filename, char delimiter = ',',
            bool parallel = false);
  ~CSVStream() {
    if (m_inputFileStream.is_open()) m_inputFileStream.close();
  }

  CSVStream(const CSVStream &) = delete;
  CSVStream &operator=(const CSVStream &) = delete;
  CSVStream(CSVStream &&) = delete;
  CSVStream &operator=(CSVStream &&) = delete;

  // Renamed and modified: Reads the next full row and parses it.
  // Primarily for sequential use or the producer thread.
  // Returns true if a row was successfully read, false otherwise (EOF or error).
  bool readNextRow(CSVRow &row);

  // Get the next raw line from the file. Thread-safe if m_parallel is true.
  // Returns an empty string if EOF or error. Check stream state externally.
  std::string getLine();

  // Parse a raw line string into a CSVRow. This can be called concurrently
  // by multiple worker threads, as it operates on local data (`rawLine`).
  // It requires the `header` to be passed, as `m_header` is a member.
  void parseLine(const std::string &rawLine, CSVRow &row,
                 const std::vector<std::string> &header) const;

  explicit operator bool() const { return m_inputFileStream.good(); }
  bool good() const { return m_inputFileStream.good(); }        // Added utility
  bool is_open() const { return m_inputFileStream.is_open(); }  // Added utility
  bool eof() const { return m_inputFileStream.eof(); }          // Added utility

  const std::vector<std::string> &getHeader() const {
    return m_header;
  }  // Added const
  const std::string &getFilename() const { return m_filename; }

 private:
  void readHeader();  // Reads the header line and populates m_header. Called once.

  std::string m_filename;
  std::ifstream m_inputFileStream;
  std::vector<std::string> m_header;
  std::mutex m_mutex;  // Protects access to m_inputFileStream in getLine()

  char m_delimiter;
  bool m_parallel;  // Controls if m_mutex is used in getLine()
};

const std::array<std::string, 2> INVALID_CELL_CONTENT = {"", "NULL"};
constexpr const char *const INVALID_CELL_VALUE = "INVALID_VALUE";

/**
 * @brief A utility struct for accessing CSV rows.
 */
struct CSVUTIL AccessCSVRow{template <typename T> void operator()(
    const CSVStream::CSVRow &row, const std::string &colName, T &value)
                                const {try {auto value_str = row.at(colName);
if (value_str.empty() ||
    std::any_of(
        INVALID_CELL_CONTENT.begin(), INVALID_CELL_CONTENT.end(),
        [&value_str](const std::string &invalid) { return value_str == invalid; }))
  value_str = INVALID_CELL_VALUE;
convert(value_str, value);
}  // namespace opencover::utils::read
catch (const std::out_of_range &ex) {
  auto err_msg = "Column " + colName + " not found in row ";
  for (auto &[key, val] : row) {
    err_msg += key + ": " + val + ", ";
  }
  throw CSVStream_Exception(err_msg);
}
catch (const std::invalid_argument &ex) {
  throw CSVStream_Exception("Invalid argument for column " + colName);
}
catch (const std::exception &ex) {
  throw CSVStream_Exception(ex.what());
}
}

private:
template <typename T>
void convert(const std::string &value_str, T &value) const {
  std::istringstream ss(value_str);
  ss >> value;
}

void convert(const std::string &value_str, std::string &value) const {
  value = value_str;
}

template <typename... Ts>
void convert(const std::string &value_str, std::variant<Ts...> &value) const {
  std::visit(
      [&value_str](auto &val) {
        std::istringstream ss(value_str);
        ss >> val;
      },
      value);
}
}
;

typedef std::map<std::string, CSVStream> StreamMap;

/**
 * @brief A utility function for converting a string to a value of type T.
 *
 * Usage:
 * double value;
 * CSVStream::CSVRow row;
 *
 * while (stream >> row) { // Or stream.readNextRow(row)
 * access_CSVRow(row, "columnName", value);
 * std::cout << value << "\n";
 * }
 */
inline constexpr AccessCSVRow ACCESS_CSV_ROW{};

StreamMap CSVUTIL getCSVStreams(const boost::filesystem::path &dirPath);
}  // namespace opencover::utils::read

#endif
