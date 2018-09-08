#include <virvo/vvvecmath.h>
#include <virvo/mem/align.h>
#include <virvo/sse/sse.h>

#include <gtest/gtest.h>

#include <iostream>

class MatrixTest : public ::testing::Test
{
protected:
  void SetUp()
  {
    virvomatrix1.identity();
    virvomatrix1(0, 3) = 200.0f;
    virvomatrix1(1, 3) = 4.0f;
    virvomatrix1(2, 3) = 15.0f;

    virvomatrix2.identity();
    virvomatrix2(0, 0) = 10.0f;
    virvomatrix2(1, 1) = 4.0f;
    virvomatrix2(2, 2) = 5.0f;

    ssematrix1 = virvomatrix1;
    ssematrix2 = virvomatrix2;

    CACHE_ALIGN float soax[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    CACHE_ALIGN float soay[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    CACHE_ALIGN float soaz[] = { 0.0f, 0.5f, 0.0f, 0.5f };
    CACHE_ALIGN float soaw[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    ssevec3 = virvo::sse::Vec3(soax, soay, soaz);
    ssevec4 = virvo::sse::Vec4(soax, soay, soaz, soaw);
  }

  virvo::Matrix virvomatrix1;
  virvo::Matrix virvomatrix2;

  virvo::sse::Matrix ssematrix1;
  virvo::sse::Matrix ssematrix2;

  virvo::sse::Vec3 ssevec3;
  virvo::sse::Vec4 ssevec4;
};

TEST_F(MatrixTest, SseInitFromVirvoWorks)
{
  virvo::Matrix tmp = ssematrix1;
  for (size_t i = 0; i < 4; ++i)
  {
    for (size_t j = 0; j < 4; ++j)
    {
      EXPECT_EQ(tmp(i, j), virvomatrix1(i, j));
    }
  }
}

TEST_F(MatrixTest, SseMultWorks)
{
  virvo::Matrix virvomul = virvomatrix1 * virvomatrix2;
  virvo::Matrix ssemul = ssematrix1 * ssematrix2;
  for (size_t i = 0; i < 4; ++i)
  {
    for (size_t j = 0; j < 4; ++j)
    {
      EXPECT_EQ(virvomul(i, j), ssemul(i, j));
    }
  }
}

TEST_F(MatrixTest, SseTransposeWorks)
{
  virvo::Matrix virvotrans = virvomatrix1;
  virvotrans.transpose();
  virvo::sse::Matrix ssetrans = ssematrix1;
  ssetrans.transpose();
  virvo::Matrix tmp = ssetrans;
  for (size_t i = 0; i < 4; ++i)
  {
    for (size_t j = 0; j < 4; ++j)
    {
      EXPECT_EQ(virvotrans(i, j), tmp(i, j));
    }
  }
}

TEST_F(MatrixTest, SseMultVec4Works)
{
}

/* performance tests */

#define NUM_PERF_TESTS 50000000

TEST_F(MatrixTest, VirvoMultPerf)
{
  volatile size_t i;
  for (i = 0; i < NUM_PERF_TESTS; ++i)
  {
    virvo::Matrix tmp = virvomatrix1 * virvomatrix2;
  }
}

TEST_F(MatrixTest, SseMultPerf)
{
  volatile size_t i;
  for (i = 0; i < NUM_PERF_TESTS; ++i)
  {
    virvo::sse::Matrix tmp = ssematrix1 * ssematrix2;
  }
}

TEST_F(MatrixTest, VirvoTransposePerf)
{
  volatile size_t i;
  for (i = 0; i < NUM_PERF_TESTS; ++i)
  {
    virvo::Matrix tmp = virvomatrix1;
    tmp.transpose();
  }
}

TEST_F(MatrixTest, SseTransposePerf)
{
  volatile size_t i;
  for (i = 0; i < NUM_PERF_TESTS; ++i)
  {
    virvo::sse::Matrix tmp = ssematrix1;
    tmp.transpose();
  }
}

TEST_F(MatrixTest, VirvoInvertPerf)
{
  volatile size_t i;
  for (i = 0; i < NUM_PERF_TESTS; ++i)
  {
    virvo::Matrix tmp = virvomatrix1;
    tmp.invert();
  }
}

TEST_F(MatrixTest, SseInvertPerf)
{
  volatile size_t i;
  for (i = 0; i < NUM_PERF_TESTS; ++i)
  {
    virvo::sse::Matrix tmp = ssematrix1;
    tmp.invert();
  }
}

GTEST_API_ int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

