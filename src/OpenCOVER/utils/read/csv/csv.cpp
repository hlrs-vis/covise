#include "csv.h"

namespace opencover::utils::read {

CSVStream::CSVStream(const std::string &filename, char delimiter, bool parallel)
    : m_filename(filename), m_delimiter(delimiter), m_parallel(parallel) {
  m_inputFileStream.open(filename.c_str());
  if (!m_inputFileStream.is_open())
    throw CSVStream_Exception("Could not open file " + filename);
  readHeader();
}

void CSVStream::readHeader() {
  std::string headerLine;
  // Read header directly, no mutex needed as it's the first operation on a fresh
  // stream.
  std::getline(m_inputFileStream, headerLine);
  if (headerLine.empty()) {  // Handle potentially empty file or just comments.
    // Check if the stream is good, if not, it means EOF.
    if (!m_inputFileStream.good() && m_inputFileStream.eof()) {
      throw CSVStream_Exception("CSV file is empty or header is missing.");
    }
  }
  // Skip comments if header is empty or starts with #
  while (!headerLine.empty() && headerLine[0] == '#') {
    std::getline(m_inputFileStream, headerLine);
  }

  // Remove carriage return characters
  if (headerLine.find('\r') != std::string::npos)
    headerLine.erase(headerLine.find('\r'), 1);

  std::string colName("");
  std::stringstream ss(headerLine);
  while (std::getline(ss, colName, m_delimiter)) m_header.push_back(colName);

  if (m_header.empty()) {
    throw CSVStream_Exception("CSV header is empty or invalid after parsing.");
  }
}

// Thread-safe line acquisition from file stream
std::string CSVStream::getLine() {
  std::string line;
  // Ensure the entire operation of getting a *valid* line is locked.
  if (m_parallel) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::getline(m_inputFileStream, line);
    // Skip comments within the lock to ensure state consistency
    while (!line.empty() && line[0] == '#') {
      std::getline(m_inputFileStream, line);
    }
  } else {  // Single-threaded access, no lock needed
    std::getline(m_inputFileStream, line);
    while (!line.empty() && line[0] == '#') {
      std::getline(m_inputFileStream, line);
    }
  }

  // Remove carriage return characters
  if (line.find('\r') != std::string::npos) line.erase(line.find('\r'), 1);

  return line;
}

// For sequential processing or by a single producer thread
bool CSVStream::readNextRow(CSVRow &row) {
  std::string rawLine =
      getLine();  // Get a line (thread-safely if m_parallel is true)

  // Check if end of file or bad state after trying to read
  if (rawLine.empty() && (!m_inputFileStream.good() && m_inputFileStream.eof())) {
    return false;  // No more rows and reached EOF
  }
  // If rawLine is empty but stream is still good, it implies an empty data line.
  // You might want to skip it or handle as an error based on your CSV format.
  // For now, we'll try to parse it (parseLine will result in empty values).
  // Or you could loop: while(rawLine.empty() && m_inputFileStream.good()) rawLine
  // =
  // getLine(); For simplicity, we'll process the empty line here.

  // Clear the row to ensure it doesn't contain old data if re-used
  row.clear();
  parseLine(rawLine, row, m_header);  // Parse the line using the class's header
  return true;
}

// Parse a raw line string into a CSVRow. Can be called concurrently by consumers.
// Note: The original quoted string handling logic was problematic for standard
// CSV.
// This refactored version directly reflects the original logic's structure
// operating
// on `rawLine`. For robust CSV parsing (especially complex quoted fields, escaped
// quotes), a dedicated CSV parsing library or a more sophisticated state machine
// is
// recommended.
void CSVStream::parseLine(const std::string &rawLine, CSVRow &row,
                          const std::vector<std::string> &header) const {
  std::string value("");
  std::stringstream ss(rawLine);

  // Clear the row to ensure it doesn't contain old data if re-used
  row.clear();

  for (const auto &colName : header) {
    // Iterate over `header` passed as
    // argument
    if (!std::getline(ss, value, m_delimiter)) {
      // Ran out of values before all headers were filled, implies malformed CSV
      // line. Fill remaining columns with empty strings.
      row[colName] = "";
      continue;  // Go to next header
    }

    // Original problematic logic for quoted strings (replicated here for
    // consistency
    // with original intent): This logic assumes `getline` will split "a,b" into
    // "a"
    // and "b" if delimiter is comma, and then tries to re-combine them if quotes
    // are
    // present. This is non-standard. A proper CSV parser would read "a,b" as one
    // token if quoted.
    if (auto it = value.find('"'); it != std::string::npos) {
      std::string next = value.substr(it + 1);  // Remove leading quote
      value = next;
      // This loop implies reading *more* segments from the stringstream `ss`
      // to find the closing quote, which is unusual for single-cell parsing.
      // It suggests `getline` splits a quoted field containing delimiters.
      while (next.find('"') == std::string::npos &&
             !ss.eof()) {  // Added !ss.eof() to prevent infinite loop on malformed
        // data
        std::string temp_next;
        if (std::getline(ss, temp_next, m_delimiter)) {
          next = temp_next;
          value += "," + next;
        } else {
          // No more delimiters, just take remaining part
          std::streamoff pos = ss.tellg();
          std::streamoff len = static_cast<std::streamoff>(ss.str().length());
          std::streamoff diff = pos != -1 ? pos : 0;  // handle tellg() == -1
          next = ss.str().substr(
              static_cast<std::size_t>(diff));  // Use the current position as
                                                // start
          value += "," + next;
          break;
        }
      }
      if (!value.empty() && value.back() == '"') {
        value = value.substr(0, value.size() - 1);  // Remove trailing quote
      } else if (!value.empty() && it == 0 && value.size() == 1 &&
                 value.front() == '"') {
        // Case of empty quoted field: ""
        value = "";
      } else {
        // Malformed quoted string, could throw an error or just leave as is
        // For simplicity, we'll leave it as is, but it's a parse error.
      }
    }
    row[colName] = value;
  }
}

}  // namespace opencover::utils::read
// #include "csv.h"

// #include <mutex>
// #include <string>

// namespace opencover::utils::read {

// CSVStream::CSVStream(const std::string &filename, char delimiter, bool parallel)
//     : m_filename(filename), m_delimiter(delimiter), m_parallel(parallel) {
//   m_inputFileStream.open(filename.c_str());
//   if (!m_inputFileStream.is_open())
//     throw CSVStream_Exception("Could not open file " + filename);
//   readHeader();
// }

// void CSVStream::readHeader() {
//   std::string colName("");
//   auto ss = getLine();
//   while (std::getline(ss, colName, m_delimiter)) m_header.push_back(colName);
// }

// void CSVStream::readLine(CSVStream::CSVRow &row) {
//   std::string value("");
//   auto ss = getLine();
//   size_t currentColNameIdx = 0;
//   for (auto &header : m_header) {
//     std::getline(ss, value, m_delimiter);
//     // if there is a comma in the value the string will contain \" at the
//     beginning
//     // and end of the cell
//     if (auto it = value.find('\"'); it != std::string::npos) {
//       std::string next = value.substr(it + 1);
//       value = next;
//       while (next.find('\"') == std::string::npos) {
//         std::getline(ss, next, m_delimiter);
//         value += "," + next;
//       }
//       value = value.substr(0, value.size() - 1);
//     }

//     row[header] = value;
//   }
// }

// std::stringstream CSVStream::getLine() {
//   if (m_parallel) std::lock_guard<std::mutex> lock(m_mutex);
//   std::getline(m_inputFileStream, m_currentline);
//   // skip comments
//   while (!m_currentline.empty() && m_currentline[0] == '#')
//     std::getline(m_inputFileStream, m_currentline);

//   // remove carriage return characters
//   if (m_currentline.find('\r') != std::string::npos)
//     m_currentline.erase(m_currentline.find('\r'), 1);
//   std::stringstream ss(m_currentline);
//   return ss;
// }
// }  // namespace opencover::utils::read
