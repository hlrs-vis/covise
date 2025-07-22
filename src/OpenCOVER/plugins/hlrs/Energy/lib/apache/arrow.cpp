#include "arrow.h"

#include <arrow/chunked_array.h>
#include <arrow/io/api.h>
#include <arrow/table.h>

#include <memory>

namespace apache {

ArrowReader::ArrowReader(const std::string &filepath, IO io) { init(filepath); }

arrow::Status ArrowReader::initMemoryMapped(const std::string &filepath) {
  std::shared_ptr<arrow::io::MemoryMappedFile> file;
  arrow::Status status =
      arrow::io::MemoryMappedFile::Open(filepath, arrow::io::FileMode::READ)
          .Value(&file);

  if (!status.ok()) return status;
  return openFile(file);
}

arrow::Status ArrowReader::initReadableFile(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> file;
  arrow::Status status = arrow::io::ReadableFile::Open(filepath).Value(&file);

  if (!status.ok()) return status;
  return openFile(file);
}

arrow::Status ArrowReader::openFile(
    std::shared_ptr<arrow::io::RandomAccessFile> file) {
  return arrow::ipc::RecordBatchFileReader::Open(file).Value(&m_reader);
}

std::shared_ptr<arrow::Table> ArrowReader::getTable() const {
  auto table_status = m_reader->ToTable();
  if (!table_status.ok()) return nullptr;
  return *table_status;
}

std::shared_ptr<arrow::ChunkedArray> ArrowReader::readColumnFromTable(
    const std::string &columnName, std::shared_ptr<arrow::Table> table) const {
  if (!table) {
    table = getTable();
    if (!table) throw std::runtime_error("Table is null, cannot read column.");
  }

  return table->GetColumnByName(columnName);
}

void ArrowReader::init(const std::string &filepath, IO io) {
  arrow::Status status;
  switch (io) {
    case IO::MEM_MAP:
      status = initMemoryMapped(filepath);
      break;
    case IO::FILE:
      status = initReadableFile(filepath);
      break;
  }

  if (!status.ok())
    throw std::runtime_error("Failed to initialize ArrowReader: " +
                             status.ToString());
}

}  // namespace apache
