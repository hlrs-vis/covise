#pragma once

#include <arrow/api.h>
#include <arrow/chunked_array.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/status.h>

#include <memory>

#include "export.h"
#include "enums.h"

namespace apache {

/**
 * @class ArrowReader
 * @brief Utility class for reading Apache Arrow files.
 *
 * ArrowReader provides an interface to read Arrow files using either memory-mapped
 * or readable file IO. It allows access to the file's schema, table, and individual
 * columns.
 *
 * @note Requires Apache Arrow C++ library.
 *
 * @param filepath Path to the Arrow file to be read.
 * @param io IO mode for file access (default: memory-mapped).
 *
 * @method getReader() Returns the underlying Arrow RecordBatchFileReader.
 * @method getSchema() Returns the schema of the Arrow file.
 * @method getTable() Returns the entire Arrow table.
 * @method readColumnFromTable() Reads a specific column from the Arrow table by
 * name.
 *
 * @private
 * @method init() Initializes the reader with the specified file and IO mode.
 * @method initMemoryMapped() Initializes the reader using memory-mapped IO.
 * @method initReadableFile() Initializes the reader using readable file IO.
 * @method openFile() Opens the file and prepares the reader.
 */
class ARROWUTIL ArrowReader {
 public:
  ArrowReader(const std::string &filepath, IO io = IO::MEM_MAP);

  std::shared_ptr<arrow::ipc::RecordBatchFileReader> getReader() const {
    return m_reader;
  }
  auto getSchema() const { return m_reader->schema(); }
  std::shared_ptr<arrow::Table> getTable() const;
  std::shared_ptr<arrow::ChunkedArray> readColumnFromTable(
      const std::string &columnName,
      std::shared_ptr<arrow::Table> table = nullptr) const;

 private:
  void init(const std::string &filepath, IO io = IO::MEM_MAP);
  arrow::Status initMemoryMapped(const std::string &filepath);
  arrow::Status initReadableFile(const std::string &filepath);
  arrow::Status openFile(std::shared_ptr<arrow::io::RandomAccessFile> file);

  std::shared_ptr<arrow::ipc::RecordBatchFileReader> m_reader;
};
}  // namespace apache
