/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



/************************************************************************/

/* main

   The executable exits with either 0 (under all conditions not listed
   below) or 1 (under the following conditions):
   1. A fatal error occurs while interpreting from a file.
   2. Read_tool_file fails.
   3. An error occurs in rs274ngc_init.

   ***********************************************************************

   Here are three ways in which the rs274abc executable may be called.
   Any other sort of call to the executable will cause an error message
   to be printed and the interpreter will not run. Other executables
   may be called similarly.

   1. If the rs274abc stand-alone executable is called with no arguments,
   input is taken from the keyboard, and an error in the input does not
   cause the rs274abc executable to exit.

   EXAMPLE:

   1A. To interpret from the keyboard, enter:

   rs274abc

   ***********************************************************************

   2. If the executable is called with one argument, the argument is
   taken to be the name of an NC file and the file is interpreted as
   described in the documentation of interpret_from_file.

   EXAMPLES:

   2A. To interpret the file "cds.abc" and read the results on the
   screen, enter:

   rs274abc cds.abc

   2B. To interpret the file "cds.abc" and print the results in the file
   "cds.prim", enter:

   rs274abc cds.abc > cds.prim

   ***********************************************************************

   Whichever way the executable is called, this gives the user several
   choices before interpretation starts

   1 = start interpreting
   2 = choose parameter file
   3 = read tool file ...
   4 = turn block delete switch ON
   5 = adjust error handling...

   Interpretation starts when option 1 is chosen. Until that happens, the
   user is repeatedly given the five choices listed above.  Item 4
   toggles between "turn block delete switch ON" and "turn block delete
   switch OFF".  See documentation of adjust_error_handling regarding
   what option 5 does.

   User instructions are printed to stderr (with fprintf) so that output
   can be redirected to a file. When output is redirected and user
   instructions are printed to stdout (with printf), the instructions get
   redirected and the user does not see them.

   */

main(int argc, char **argv)
{
    int status;
    int choice;
    int do_next; /* 0=continue, 1=mdi, 2=stop */
    int block_delete;
    char buffer[80];
    int tool_flag;
    int gees[RS274NGC_ACTIVE_G_CODES];
    int ems[RS274NGC_ACTIVE_M_CODES];
    double sets[RS274NGC_ACTIVE_SETTINGS];
    char default_name[] SET_TO "rs274ngc.var";
    int print_stack;

    if (argc > 3)
    {
        fprintf(stderr, "Usage \"%s\"\n", argv[0]);
        fprintf(stderr, "   or \"%s <input file>\"\n", argv[0]);
        fprintf(stderr, "   or \"%s <input file> <output file>\"\n", argv[0]);
        exit(1);
    }

    do_next SET_TO 2; /* 2=stop */
    block_delete SET_TO OFF;
    print_stack SET_TO OFF;
    tool_flag SET_TO 0;
    strcpy(_parameter_file_name, default_name);
    _outfile SET_TO stdout; /* may be reset below */

    for (;;)
    {
        fprintf(stderr, "enter a number:\n");
        fprintf(stderr, "1 = start interpreting\n");
        fprintf(stderr, "2 = choose parameter file ...\n");
        fprintf(stderr, "3 = read tool file ...\n");
        fprintf(stderr, "4 = turn block delete switch %s\n",
                ((block_delete IS OFF) ? "ON" : "OFF"));
        fprintf(stderr, "5 = adjust error handling...\n");
        fprintf(stderr, "enter choice => ");
        gets(buffer);
        if (sscanf(buffer, "%d", &choice) ISNT 1)
            continue;
        if (choice IS 1)
            break;
        else if (choice IS 2)
        {
            if (designate_parameter_file(_parameter_file_name) ISNT 0)
                exit(1);
        }
        else if (choice IS 3)
        {
            if (read_tool_file("") ISNT 0)
                exit(1);
            tool_flag SET_TO 1;
        }
        else if (choice IS 4)
            block_delete SET_TO((block_delete IS OFF) ? ON : OFF);
        else if (choice IS 5)
            adjust_error_handling(argc, &print_stack, &do_next);
    }
    fprintf(stderr, "executing\n");
    if (tool_flag IS 0)
    {
        if (read_tool_file("rs274ngc.tool_default") ISNT 0)
            exit(1);
    }

    if (argc IS 3)
    {
        _outfile SET_TO fopen(argv[2], "w");
        if (_outfile IS NULL)
        {
            fprintf(stderr, "could not open output file %s\n", argv[2]);
            exit(1);
        }
    }

    if ((status SET_TO rs274ngc_init())ISNT RS274NGC_OK)
    {
        report_error(status, print_stack);
        exit(1);
    }

    if (argc IS 1)
        status SET_TO interpret_from_keyboard(block_delete, print_stack);
    else /* if (argc IS 2 or argc IS 3) */
    {
        status SET_TO rs274ngc_open(argv[1]);
        if (status ISNT RS274NGC_OK) /* do not need to close since not open */
        {
            report_error(status, print_stack);
            exit(1);
        }
        status SET_TO interpret_from_file(do_next, block_delete, print_stack);
        rs274ngc_file_name(buffer, 5); /* called to exercise the function */
        rs274ngc_file_name(buffer, 79); /* called to exercise the function */
        rs274ngc_close();
    }
    rs274ngc_line_length(); /* called to exercise the function */
    rs274ngc_sequence_number(); /* called to exercise the function */
    rs274ngc_active_g_codes(gees); /* called to exercise the function */
    rs274ngc_active_m_codes(ems); /* called to exercise the function */
    rs274ngc_active_settings(sets); /* called to exercise the function */
    rs274ngc_exit(); /* saves parameters */
    exit(status);
}

/***********************************************************************/
