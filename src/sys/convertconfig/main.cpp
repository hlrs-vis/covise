/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigImportReader.h>
#include <qapplication.h>
#include <QStringList>
#include <QDir>

#ifndef _WIN32
#include <unistd.h>
#endif

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        bool errflg = false;
        bool resolve = false;
        char c;
        char *ofile = NULL;
        char *tfile = NULL;
        QStringList configFiles;
        extern char *optarg;
        extern int optind;

        while ((c = getopt(argc, argv, "hro:t:")) != -1)
        {
            switch (c)
            {
            case 'o':
                ofile = optarg;
                break;
            case 't':
                tfile = optarg;
                break;
            case 'r':
                resolve = true;
                break;
            case 'h':
                errflg = true;
                break;
            case '?':
                errflg = true;
                break;
            }
        }

        if (optind >= argc)
        {
            errflg = true;
        }
        else
        {
            for (int i = optind; i < argc; i++)
                configFiles += argv[i];
        }

        if (errflg)
        {
            fprintf(stderr, "\n");
            fprintf(stderr, "usage: convertconfig [-h] [-r] [-o <output file>]  \n");
            fprintf(stderr, "                     [-t <translation file>] <list of covise.config files>\n\n");
            fprintf(stderr, "       -r : resolve included files\n");
            fprintf(stderr, "       -o : output file, <covise.config file>.xml if not given\n");
            fprintf(stderr, "       -t : xml including the conversion rules\n");
            fprintf(stderr, "       -h : print this message\n\n");
            return 0;
        }

        for (QStringList::Iterator it = configFiles.begin(); it != configFiles.end(); ++it)
        {
            QString inputFile, outputFile, transformFile;

            if ((*it).endsWith(".xml"))
            {
                inputFile = QString(*it) + ".old";
                QDir::current().rename(*it, inputFile);
            }
            else
            {
                inputFile = *it;
            }

            if (ofile == NULL || configFiles.size() > 1)
            {
                if (configFiles.size() > 1 && ofile != NULL)
                    fprintf(stderr, "\nWARNING: Output filename ignored because you entered a list of input files\n");

                if ((*it).endsWith(".xml"))
                {
                    outputFile = QString(*it);
                }
                else
                {
                    outputFile = QString(*it) + ".xml";
                }
            }
            else
            {
                outputFile = QString(ofile);
            }

            if (tfile == NULL)
            {
#ifdef YAC
                char *covisepath = getenv("YACDIR");
#else
                char *covisepath = getenv("COVISEDIR");
#endif
                transformFile = QString(covisepath) + "/config/transform.xml";
            }
            else
            {
                transformFile = QString(tfile);
            }

            coConfigImportReader *reader = new coConfigImportReader(inputFile,
                                                                    outputFile, transformFile, resolve);

            reader->parse();

            reader->updatev0v1();

            reader->write();
            delete reader;
        }
    }
    return 0;
}
