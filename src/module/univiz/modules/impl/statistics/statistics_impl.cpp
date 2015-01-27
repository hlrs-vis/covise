/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#define rint(x) ((int)(x))
#endif
void statistics_impl(UniSys *us,
                     Unstructured *unst, int comp)
{
    float min = 1e19;
    float max = -1e19;
    double sum = 0;

    int argMin = 0;
    int argMax = 0;

    int n = unst->nNodes * unst->getNodeCompVecLen(comp);

    // Collect and sort data, compute mean
    vector<float> data;
    int pos = 0;
    for (int n = 0; n < unst->nNodes; n++)
    {

        double v[256]; // HACK
        unst->getVectorN(n, comp, v);

        for (int c = 0; c < unst->getNodeCompVecLen(comp); c++)
        {
            float x = v[c];
            data.push_back(x);

            sum += x;
            if (x < min)
            {
                min = x;
                argMin = pos;
            }
            if (x > max)
            {
                max = x;
                argMax = pos;
            }
            pos++;
        }
    }
    sort(data.begin(), data.end());
    float mean = sum / n;

    // Compute stddev
    sum = 0;
    for (int n = 0; n < unst->nNodes; n++)
    {
        double v[256]; // HACK
        unst->getVectorN(n, comp, v);
        for (int c = 0; c < unst->getNodeCompVecLen(comp); c++)
        {
            float x = v[c] - mean;
            sum += x * x;
        }
    }
    float stddev = sqrt(sum) / n;

    float perc01 = data[(int)rint((n - 1) * 0.01)];
    float perc10 = data[(int)rint((n - 1) * 0.10)];
    float perc25 = data[(int)rint((n - 1) * 0.25)];
    float perc50 = data[(int)rint((n - 1) * 0.50)];
    float perc75 = data[(int)rint((n - 1) * 0.75)];
    float perc90 = data[(int)rint((n - 1) * 0.90)];
    float perc99 = data[(int)rint((n - 1) * 0.99)];

    char str[1000];
    sprintf(str,
            "-----------------------\n"
            "nodes:  %d\n"
            "mean:   %g\n"
            "stddev: %g\n"
            "\n"
            "min:    %g (at node %d)\n"
            " 1%%:    %g\n"
            "10%%:    %g\n"
            "25%%:    %g\n"
            "med:    %g\n"
            "75%%:    %g\n"
            "90%%:    %g\n"
            "99%%:    %g\n"
            "max:    %g (at node %d)\n",
            n, mean, stddev, min, argMin, perc01, perc10, perc25, perc50, perc75, perc90, perc99, max, argMax);

    // TODO
    //AVSmodify_parameter("status", AVS_VALUE, str, 0, 0);
    printf("%s", str);
    {
        char *last = str;
        while (last && *last != '\0')
        {
            char *nl;
            nl = strchr(last, '\n');
            if (!nl)
                break;
            *nl = '\0';
            //char line[1000];
            if (strlen(last) > 0)
                us->info(last);
            else
                us->info(" ");
            last = nl + 1;
        }
    }

    //for (int i = 0; i < 100; i++) {
    //  printf("%3d%%: %20.12f\n", i, data[(int)rint((n-1)*i*0.01)]);
    //
}
