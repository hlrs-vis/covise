#include <gtest/gtest.h>
#include <lib/apache/arrow.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/chunked_array.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>
#include <cstdio>

using namespace apache;

namespace {

std::string create_test_arrow_file() {
    auto schema = arrow::schema({arrow::field("col1", arrow::float64())});
    arrow::DoubleBuilder builder;
    builder.Append(1.23);
    builder.Append(4.56);
    std::shared_ptr<arrow::Array> arr;
    builder.Finish(&arr);
    auto table = arrow::Table::Make(schema, {arr});

    std::string filename = "test_arrow_reader.arrow";
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    arrow::io::FileOutputStream::Open(filename).Value(&outfile);
    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
    arrow::ipc::MakeFileWriter(outfile, schema).Value(&writer);
    writer->WriteTable(*table);
    writer->Close();
    outfile->Close();
    return filename;
}

TEST(ArrowReader, ThrowsOnMissingFile) {
    EXPECT_THROW(ArrowReader("does_not_exist.arrow"), std::runtime_error);
}

TEST(ArrowReader, CanReadSchema) {
    std::string filename = create_test_arrow_file();
    ArrowReader reader(filename);

    auto schema = reader.getSchema();
    ASSERT_EQ(schema->num_fields(), 1);
    EXPECT_EQ(schema->field(0)->name(), "col1");

    // Clean up
    std::remove(filename.c_str());
}

TEST(ArrowReader, CanReadTable) {
    std::string filename = create_test_arrow_file();
    ArrowReader reader(filename);

    auto schema = reader.getSchema();

    auto table = reader.getTable();
    ASSERT_TRUE(table != nullptr);
    EXPECT_EQ(table->num_columns(), 1);

    // Clean up
    std::remove(filename.c_str());
}

TEST(ArrowReader, CanReadColumnFromTable) {
    std::string filename = create_test_arrow_file();
    ArrowReader reader(filename);

    auto schema = reader.getSchema();

    auto table = reader.getTable();
    auto chunked = reader.readColumnFromTable("col1", table);
    ASSERT_TRUE(chunked != nullptr);
    EXPECT_EQ(chunked->num_chunks(), 1);

    // Clean up
    std::remove(filename.c_str());
}

} // namespace
