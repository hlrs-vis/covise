/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>

/*(+)**********************************************************************************
      Function name:  Ifi_Stp3ReadImageFile

      Synopsis:    int Ifi_Stp3ReadImageFile(char *inputfile, Ifi_SeriesDesc_TP *sh)
         Inputs:   char *inputfile       -  The name of the STP3 image file
         Outputs:  Ifi_SeriesDesc_TP *sh -  The pointer to structure to store
                                            the images informations
         Returns:  the return code of the function
      Description: This function read the STP3 image file and the information
                   about images store to the predefined structure
                   Return value 0 means ok, 1 means error occured
      Comments:

***************************************************************************************/

int Ifi_Stp3ReadImageFile(char *inputfile);

int main(int argc, char *argv[])
{
    if (argc >= 2)
        Ifi_Stp3ReadImageFile(argv[1]);
}

int Ifi_Stp3ReadImageFile(char *inputfile)

/*(-)*/
{
    //Ifi_ImageDesc_TP *ih = NULL, *pih = NULL, ihv;
    int i = 0, j = 0, fpos = 0, image_size = 0;
    double z_pos = 0.0, gantry = 0.0, psiz = 0.0;
    FILE *ifi = NULL, *ofi = NULL;
    int image_type; /* CT, MR, PET */
    int series_header; /* = 512 */
    int image_header; /* = 512 */
    int image_length; /* = byte_per_pixel x resolution x resolution x ??? */
    char patient_name[80];
    char comment[80];
    int resolution; /* = 256 or 512 */
    int byte_per_pixel; /* = 2 */
    int num_of_slices; /* max 200 */
    float pixel_size;
    char date[80];

    ifi = fopen(inputfile, "rb");
    if (ifi == NULL)
    {
        printf("Cannot open file %s\n", inputfile);
        return (1);
    }

    printf("\n==================================\n");
    printf("\n  R E A D I N G\n");
    printf("\n==================================\n");
    printf("\n\n");

    fread(&image_type, 4, 1, ifi);
    fread(&series_header, 4, 1, ifi);
    fread(&image_header, 4, 1, ifi);
    fread(&image_length, 4, 1, ifi);
    fread(patient_name, 1, 80, ifi);
    fread(comment, 1, 80, ifi);
    fread(&resolution, 4, 1, ifi);
    fread(&byte_per_pixel, 4, 1, ifi);
    fread(&num_of_slices, 4, 1, ifi);

    fread(&psiz, 8, 1, ifi);
    pixel_size = (float)psiz;

    fread(date, 1, 80, ifi);
    char **images = NULL;

    switch (image_type)
    {
    case 1:
    case 100:
        image_type = 1;
        printf("Image Type:       CT images\n");
        break;
    case 2:
    case 200:
        image_type = 2;
        printf("Image Type:       MR Images\n");
        break;
    case 3:
    case 410:
        image_type = 3;
        printf("Image Type:       PET images\n");
        break;
    default:
        printf("Image Type:       unknown\n");
    }

    printf("Series Header:    %d\n", series_header);
    printf("Image Header:     %d\n", image_header);
    printf("Image Length:     %d\n", image_length);
    printf("Patient Name:     %s\n", patient_name);
    printf("Comment:          %s\n", comment);
    printf("Resolution:       %d x %d\n", resolution, resolution);
    printf("Byte Per Pixel:   %d\n", byte_per_pixel);
    printf("Number of Slices: %d\n", num_of_slices);
    printf("Pixel Size:       %f\n", pixel_size);
    printf("Date:             %s\n", date);
    printf("\n\n");

    image_size = resolution * resolution * byte_per_pixel;

    printf("  Image            Z-pos       \n");
    printf(" -------------------------\n");

    for (i = 0; i < num_of_slices; i++)
    {

#if 0
        ih = malloc(sizeof(ihv));

        if (ih == NULL)
        {
            printf("Not enough memory\n");
            fclose(ofi);
            return (1);
        }
#endif

        fpos = series_header + i * image_header + i * image_length;
        fseek(ifi, fpos, SEEK_SET);
        int image_type;
        fread(&image_type, 4, 1, ifi);

        fread(&z_pos, 8, 1, ifi);
        //ih->z_position = (float) z_pos;
        fread(&gantry, 8, 1, ifi);
        //ih->gantry_tilt = (float) gantry;

        //ih->next_image = NULL;

        printf("%6d          %8.2f     \n", i + 1, (float)z_pos);

        char *image_matrix = malloc(image_size);

        if (image_matrix == NULL)
        {
            printf("Not enough memory\n");
            fclose(ifi);
            return (1);
        }

        fpos = series_header + (i + 1) * image_header + i * image_length;
        fseek(ifi, fpos, SEEK_SET);
        fread(image_matrix, 1, image_size, ifi);

#if 0
        if (i == 0)
            images = ih;
        else
            pih->next_image = ih;

        pih = ih;
#endif

    } /* end for(i=0; i< sh.nslc; ...) */

    fclose(ifi);
    return (0);
}
