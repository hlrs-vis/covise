#pragma once

#include <arrow/api.h>
#include <arrow/chunked_array.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/status.h>

#include <memory>

#include "enums.h"

namespace apache {

class ArrowReader {
 public:
  ArrowReader(const std::string &filepath, IO io = IO::MEM_MAP);

  std::shared_ptr<arrow::ipc::RecordBatchFileReader> getReader() const {
    return m_reader;
  }
  std::shared_ptr<arrow::Schema> getSchema() const { return m_reader->schema(); }
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
