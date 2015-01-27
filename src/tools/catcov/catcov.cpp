/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "catcov.h"
#include <string.h>

using namespace std;

int main(int argc, char *argv[])
{
    int time_step = 0;

    CatCov application(argc, argv);
    if (application.Diagnose(argv) != 0)
    {
        exit(EXIT_FAILURE);
    }
    application.WriteMagic();
    application.WriteSetHeader();
    for (time_step = 0; time_step < application.how_many(); ++time_step)
    {
        application.DumpFile(time_step);
    }
    application.WriteTimeAttrib();
    return 0;
}

extern char *optarg;
extern int optind;

CatCov::CatCov(int argc, char *argv[])
    : BuffSize_(0)
    , Buffer_(0)
    , Magic_(IGNORANCE)
    , InputType_(INPUT_IS_LITTLE_ENDIAN)
{
    int i;
    int option;

    not_ok_ = 0;
    pathname_ = 0;
    fileOutN_ = STDOUT_FILENO;

    while ((option = getopt(argc, argv, "ho:")) != -1)
        switch (option)
        {
        case 'o':
            if ((fileOutN_ = open(optarg, O_WRONLY | O_CREAT, 0660)) < 0)
            {
                cerr << "Could not open " << optarg << endl;
                ++not_ok_;
            }
            break;
        case '?':
        case 'h':
            ++not_ok_;
            break;
        }

    noTimeSteps_ = argc - optind;
    if (noTimeSteps_ <= 0)
        ++not_ok_;

    if (not_ok_ == 0)
    {
        pathname_ = new char *[noTimeSteps_];
        for (i = 0; optind < argc; ++optind, ++i)
        {
            pathname_[i] = argv[optind];
        }
    }
    else
    {
        noTimeSteps_ = 0;
    }
}

void CatCov::WriteMagic()
{
    char magic_string[7];
    int fileInN;
    int i;
    for (i = 0; i < noTimeSteps_; ++i)
    {
        if ((fileInN = open(pathname_[i], O_RDONLY)) < 0)
        {
            cerr << "Error when opening " << pathname_[TimeStep_] << endl;
            exit(EXIT_FAILURE);
        }
        if (read(fileInN, magic_string, 6) != 6)
        {
            cerr << "Could not test for MAGIC in file " << pathname_[i] << endl;
            exit(EXIT_FAILURE);
        }
        magic_string[6] = '\0';
        switch (Magic_)
        {
        case IGNORANCE:
            if (strcmp(magic_string, "COV_BE") == 0)
            {
                Magic_ = IS_BIG_ENDIAN;
            }
            else if (strcmp(magic_string, "COV_LE") == 0)
            {
                Magic_ = IS_LITTLE_ENDIAN;
            }
            else
            {
                Magic_ = NO_INFO;
            }
            break;
        case NO_INFO:
            if (strcmp(magic_string, "COV_BE") == 0)
            {
                Magic_ = NO_INFO_IS_BIG_ENDIAN;
            }
            else if (strcmp(magic_string, "COV_LE") == 0)
            {
                Magic_ = NO_INFO_IS_LITTLE_ENDIAN;
            }
            else
            {
                Magic_ = NO_INFO;
            }
            break;
        case IS_BIG_ENDIAN:
            if (strcmp(magic_string, "COV_BE") == 0)
            {
                Magic_ = IS_BIG_ENDIAN;
            }
            else if (strcmp(magic_string, "COV_LE") == 0)
            {
                Magic_ = IS_BIG_ENDIAN_IS_LITTLE_ENDIAN;
            }
            else
            {
                Magic_ = NO_INFO_IS_BIG_ENDIAN;
            }
            break;
        case IS_LITTLE_ENDIAN:
            if (strcmp(magic_string, "COV_BE") == 0)
            {
                Magic_ = IS_BIG_ENDIAN_IS_LITTLE_ENDIAN;
            }
            else if (strcmp(magic_string, "COV_LE") == 0)
            {
                Magic_ = IS_LITTLE_ENDIAN;
            }
            else
            {
                Magic_ = NO_INFO_IS_LITTLE_ENDIAN;
            }
            break;
        case NO_INFO_IS_BIG_ENDIAN:
            if (strcmp(magic_string, "COV_BE") == 0)
            {
                Magic_ = NO_INFO_IS_BIG_ENDIAN;
            }
            else if (strcmp(magic_string, "COV_LE") == 0)
            {
                Magic_ = IS_BIG_ENDIAN_IS_LITTLE_ENDIAN;
            }
            else
            {
                Magic_ = NO_INFO_IS_BIG_ENDIAN;
            }
            break;
        case NO_INFO_IS_LITTLE_ENDIAN:
            if (strcmp(magic_string, "COV_BE") == 0)
            {
                Magic_ = IS_BIG_ENDIAN_IS_LITTLE_ENDIAN;
            }
            else if (strcmp(magic_string, "COV_LE") == 0)
            {
                Magic_ = NO_INFO_IS_LITTLE_ENDIAN;
            }
            else
            {
                Magic_ = NO_INFO_IS_LITTLE_ENDIAN;
            }
            break;
        case IS_BIG_ENDIAN_IS_LITTLE_ENDIAN:
            break;
        }
        if (close(fileInN) < 0)
        {
            cerr << "Error when closing " << pathname_[TimeStep_] << endl;
            exit(EXIT_FAILURE);
        }
    }

    switch (Magic_)
    {
    case NO_INFO:
        cerr << "The input files have no MAGIC information" << endl;
        InputType_ = INPUT_IS_LITTLE_ENDIAN;
        if (write(fileOutN_, "COV_LE", 6) < 0)
        {
            cerr << "Could not write MAGIC information to output file " << endl;
            exit(EXIT_FAILURE);
        }
        break;
    case IS_BIG_ENDIAN:
        InputType_ = INPUT_IS_BIG_ENDIAN;
        cerr << "All input files begin with COV_BE" << endl;
        if (write(fileOutN_, "COV_BE", 6) < 0)
        {
            cerr << "Could not write MAGIC information to output file " << endl;
            exit(EXIT_FAILURE);
        }
        break;
    case IS_LITTLE_ENDIAN:
        InputType_ = INPUT_IS_LITTLE_ENDIAN;
        cerr << "All input files begin with COV_LE" << endl;
        if (write(fileOutN_, "COV_LE", 6) < 0)
        {
            cerr << "Could not write MAGIC information to output file " << endl;
            exit(EXIT_FAILURE);
        }
        break;
    case NO_INFO_IS_BIG_ENDIAN:
        InputType_ = INPUT_IS_BIG_ENDIAN;
        cerr << "Some input files (but not all of them) begin with COV_BE" << endl;
        cerr << "The rest have no MAGIC information" << endl;
        if (write(fileOutN_, "COV_BE", 6) < 0)
        {
            cerr << "Could not write MAGIC information to output file " << endl;
            exit(EXIT_FAILURE);
        }
        break;
    case NO_INFO_IS_LITTLE_ENDIAN:
        InputType_ = INPUT_IS_LITTLE_ENDIAN;
        cerr << "Some input files (but not all of them) begin with COV_LE" << endl;
        if (write(fileOutN_, "COV_LE", 6) < 0)
        {
            cerr << "Could not write MAGIC information to output file " << endl;
            exit(EXIT_FAILURE);
        }
        cerr << "The rest have no MAGIC information" << endl;
        break;
    case IS_BIG_ENDIAN_IS_LITTLE_ENDIAN:
        cerr << "Some input files with COV_LE and other with COV_BE" << endl;
        cerr << "This is a contradiction: exiting!!!" << endl;
        exit(EXIT_FAILURE);
    case IGNORANCE:
        cerr << "Indeterminate state after looking for MAGIC information " << endl;
        cerr << "in the input files: exiting!!!" << endl;
        exit(EXIT_FAILURE);
    }
}

int CatCov::Diagnose(char *argv[])
{

    int i;
    struct stat filestat;
    for (i = 0; i < noTimeSteps_; ++i)
    {
        // Test for existence and ideal buffer size !!!!
        if (stat(pathname_[i], &filestat) < 0)
        {
            cerr << "Could not 'stat' input file " << pathname_[i] << endl;
            ++not_ok_;
        }
        if (filestat.st_blksize > BuffSize_)
            BuffSize_ = filestat.st_blksize;
    }

    if (BuffSize_)
    {
        try
        {
            Buffer_ = new char[BuffSize_];
        }
        catch (...)
        {
            cerr << "Could not allocate" << BuffSize_ << " bytes for buffer" << endl;
            return -1;
        }
    }
    else
    {
        cerr << "The ideal buffer size could not be calculated... " << endl;
        ++not_ok_;
    }

    // Test if the output file descriptor refers to a terminal device
    if (not_ok_ == 0 && isatty(fileOutN_))
    {
        cerr << "The output file descriptor refers to a terminal device, " << endl;
        cerr << "Do you want to continue? (y/n) ";
        char yn;
        cin >> yn;
        if (yn == 'n' || yn == 'N')
            ++not_ok_;
    }

    if (not_ok_)
    {
        cerr << "Usage is:" << endl;
        cerr << argv[0] << " [-o <output file>] <file1> [file2 ...] " << endl;
        cerr << "Without -o option the output is written to stdout!!!!" << endl;
        return -1;
    }
    return 0;
}

void CatCov::swap_int(int &d)
{
    unsigned int &data = (unsigned int &)d;

#ifdef BYTESWAP
    // Big endian platform
    if (InputType_ == INPUT_IS_LITTLE_ENDIAN)
        data = ((data & 0xff000000) >> 24)
               | ((data & 0x00ff0000) >> 8)
               | ((data & 0x0000ff00) << 8)
               | ((data & 0x000000ff) << 24);
#else
    // Little endian platform
    if (InputType_ == INPUT_IS_BIG_ENDIAN)
        data = ((data & 0xff000000) >> 24)
               | ((data & 0x00ff0000) >> 8)
               | ((data & 0x0000ff00) << 8)
               | ((data & 0x000000ff) << 24);
#endif
}

void CatCov::WriteSetHeader()
{
    if (write(fileOutN_, "SETELE", 6) < 0)
    {
        cerr << "Error when writing SETELE header" << endl;
        exit(EXIT_FAILURE);
    }
    int noTimeSteps_s = noTimeSteps_;
    // In spite of BYTESWAP, we only have to swap if
    // the input files
    // are not in the machine format!!!
    swap_int(noTimeSteps_s);
    if (write(fileOutN_, &noTimeSteps_s, sizeof(int)) < 0)
    {
        cerr << "Error when writing SETELE header" << endl;
        exit(EXIT_FAILURE);
    }
}

void CatCov::DumpFile(int time_step)
{
    int fileInN;
    TimeStep_ = time_step;
    if ((fileInN = open(pathname_[time_step], O_RDONLY)) < 0)
    {
        cerr << "Error when opening " << pathname_[TimeStep_] << endl;
        exit(EXIT_FAILURE);
    }

    // Magic stripping for set elements
    char magic_string[7];
    if (read(fileInN, magic_string, 6) != 6)
    {
        cerr << "Could not test for MAGIC in file " << pathname_[time_step] << endl;
        exit(EXIT_FAILURE);
    }
    magic_string[6] = '\0';

    // If magic_string is "COV_BE" or "COV_LE" we do not rewind
    // and so we prevent the magic info from being written
    if (strcmp(magic_string, "COV_BE") != 0 && strcmp(magic_string, "COV_LE") != 0)
        if (lseek(fileInN, 0, SEEK_SET) != 0)
        {
            cerr << "Could not rewind input file " << pathname_[time_step] << endl;
            exit(EXIT_FAILURE);
        }
    // End of magic stripping

    WriteToOutFile(fileInN);
    if (close(fileInN) < 0)
    {
        cerr << "Error when closing " << pathname_[TimeStep_] << endl;
        exit(EXIT_FAILURE);
    }
}

void CatCov::WriteToOutFile(int fileInN)
{
    int n;

    while ((n = read(fileInN, Buffer_, BuffSize_)) > 0)
        if (write(fileOutN_, Buffer_, n) != n)
        {
            cerr << "Error when writing from " << pathname_[TimeStep_] << endl;
            exit(EXIT_FAILURE);
        }

    if (n < 0)
    {
        cerr << "Error when reading from " << pathname_[TimeStep_] << endl;
        exit(EXIT_FAILURE);
    }
}

void CatCov::WriteTimeAttrib()
{
    char strBuf[64];
    char timeAttr[] = "TIMESTEP";
    int numattrib = 1;

    sprintf(strBuf, "%d %d", 1, noTimeSteps_);
    int size = sizeof(int);
    size += strlen(timeAttr) + strlen(strBuf) + 2;
    swap_int(size);
    swap_int(numattrib);
    if (write(fileOutN_, &size, sizeof(int)) < 0 || write(fileOutN_, &numattrib, sizeof(int)) < 0)
    {
        cerr << "Error when writing size and number of attribute section" << endl;
        exit(EXIT_FAILURE);
    }

    if (write(fileOutN_, timeAttr, strlen(timeAttr) + 1) < 0)
    {
        cerr << "Error when writing TIMESTEP attribute" << endl;
        exit(EXIT_FAILURE);
    }
    if (write(fileOutN_, strBuf, strlen(strBuf) + 1) < 0)
    {
        cerr << "Error when writing TIMESTEP value" << endl;
        exit(EXIT_FAILURE);
    }
}
