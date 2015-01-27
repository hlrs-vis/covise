/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "parser.h"
#include <unistd.h>

extern ifstream *from; //oeffnet das InputFile

extern list<Name> name_list;
extern list<Curve> curve_list;
extern list<Circle> circle_list;
extern list<Vec3d> point_list;
extern list<Surf> surf_list;
extern list<Cons> cons_list;
extern list<Face> face_list;
extern list<Top> top_list;
extern list<Group> group_list;
extern list<Set> set_list;

void error(char *s)
{
    cerr << s << ' ' << '\n';
    exit(1);
}

int char2int(char &ch)
{
    int wert = ch - 48;
    return (wert);
}

char *str2char(string von)
{
    int slaenge = von.length();
    char *nach = new char[slaenge + 2];
    for (int pos = 0; pos < slaenge + 1; pos++)
    {
        nach[pos] = von[pos];
    }
    nach[slaenge + 1] = 0;
    return nach;
}

void pow10(double &a, signed b)
{

    if (b > 0)
    {
        for (int i = 0; i < b; i++)
        {

            a *= 10;
        }
    }

    if (b < 0)
    {
        for (int i = b; i != 0; i++)
        {

            a /= 10;
        }
    }
}

char skip_crap(int &pos) //gibt das erste verwertbare Zeichen zurueck
{
    char ch = ' ';
    for (; (ch == ' ' || ch == '/' || ch == ',' || ch == '=' || ch == '+');)
    {
        from->get(ch);
        pos++;
    } //Trennzeichen abfangen bis Beginn
    //unnuetzes positives Vorzeichen ignorieren.
    return (ch);
}

void skip_lines(int linien, int &pos) //Skips current line plus n lines
{
    char ch;
    for (; pos <= 81; pos++)
    {
        from->get(ch);
    } //bis ans Zeilenende

    for (; (linien) > 0; linien--)
    {
        for (int i = 80; i >= 0; i--)
            from->get(); //Zeile ueberspringen
    }
    pos = 1;
}

double lese_double(int &pos) //liest eine double-Zahl und ueberspringt
//alle Blancs und Trennzeichen.
{
    double out_double = 0;
    char ch;
    int vorzeichen = 1;

    ch = ' ';

    ch = skip_crap(pos);

    for (; pos > 71;)
    {
        skip_lines(0, pos); //Zeilenende ? dann in die naechste Zeile
        ch = skip_crap(pos);
    }

    if (ch == '-') //eventuelles negatives Vorzeichen pruefen.
    {
        vorzeichen = (-1);
        from->get(ch);
        pos++;
    }

    for (; ch != '.';) //lesen bis zum Komma (.)  !!!!!!!
    {
        out_double *= 10;
        out_double += char2int(ch);
        from->get(ch);
        pos++;
    }

    from->get(ch); // Erste Nachkommastelle...
    pos++;

    double puffer;

    //lesen der Nachkommastellen.
    for (float i = 10; ch != ' ' && ch != ',' && ch != 'E' && ch != 'e'; i *= 10)
    {
        puffer = char2int(ch);
        out_double = out_double + (puffer / i);
        from->get(ch);
        pos++;
    }

    char vorzeichen_exp;

    if (ch == 'E' || ch == 'e')
    {
        from->get(vorzeichen_exp);
        pos++;
        from->get(ch); // xxxxx
        pos++;

        signed exponent = char2int(ch) * 10;

        from->get(ch);
        exponent += char2int(ch);
        pos++;

        if (vorzeichen_exp == '-')
            exponent *= (-1);

        pow10(out_double, exponent);
    }

    out_double *= vorzeichen;
    return (out_double);
}

int lese_int(int &pos) //liest eine int-Zahl und ueberspringt
//alle Blancs und Trennzeichen.
{
    int out_int = 0;
    char ch;

    ch = skip_crap(pos);

    for (; pos > 71;)
    {
        skip_lines(0, pos); //Zeilenende ? dann in die naechste Zeile
        ch = skip_crap(pos);
    }
    for (; ch != ' ' && ch != ',';)
    {
        out_int *= 10;
        out_int += char2int(ch);

        from->get(ch);
        pos++;
    }
    return (out_int);
}

string lese_string(int &pos) //liest einen string und ueberspringt
//alle Blancs und Trennzeichen.
{

    string out_string = "";
    string zeichen;
    char ch;

    ch = skip_crap(pos);

    for (; pos > 71;)
    {
        skip_lines(0, pos); //Zeilenende ? dann in die naechste Zeile
        ch = skip_crap(pos);
    }

    for (; ch != ' ' && ch != ',';)
    {
        zeichen = ch; //der DCC laesst out_string += ch  nicht zu.
        out_string += zeichen;
        from->get(ch);
        pos++;
    }

    return (out_string);
}

string unbekannt(int &pos)
{
    char ch = ' ';
    string zeichen, out_string;
    for (; ch == ' ';)
    {
        skip_lines(0, pos);
        from->get(ch);
        pos++;
    }

    for (; ch != ' ';)
    {
        zeichen = ch; //der DCC laesst out_string += ch  nicht zu.
        out_string += zeichen;
        from->get(ch);
        pos++;
    }

    return (out_string);
}

void punkt(int &pos, string element_id)
{
    double x = lese_double(pos);
    double y = lese_double(pos);
    double z = lese_double(pos);

    skip_lines(0, pos);

    Vec3d new_point(x, y, z);
    point_list.append(new_point);

    Name new_name;
    // Speichert ID,listen-id und position
    new_name.fill(element_id, 2, point_list.length() - 1);
    name_list.append(new_name);
}

void kreis(int &pos, string element_id)
{

    char *char_buffer;
    char_buffer = str2char(element_id);
    Vec3d center_buffer(lese_double(pos), lese_double(pos), lese_double(pos));

    double rad_buffer = lese_double(pos);

    Vec3d v_buffer(lese_double(pos), lese_double(pos), lese_double(pos));
    Vec3d w_buffer(lese_double(pos), lese_double(pos), lese_double(pos));

    double alpha_buffer = lese_double(pos);
    double beta_buffer = lese_double(pos);

    Circle new_circle(char_buffer, center_buffer, rad_buffer, v_buffer, w_buffer, alpha_buffer, beta_buffer);

    skip_lines(0, pos);

    circle_list.append(new_circle); // Speichert in Kreisliste

    Name new_name;
    // Speichert ID,listen-id und position
    new_name.fill(element_id, 3, circle_list.length() - 1);

    name_list.append(new_name);

    delete[] char_buffer;
}

void kurve(int &pos, string element_id)
{
    char *char_buffer;
    char_buffer = str2char(element_id);
    int cn = lese_int(pos);
    double *par = new double[cn + 1];
    //Erzeuge Feld von Curv.Seg. der Laenge n
    Curve_Segment *csgm_buff = new Curve_Segment[cn];
    int *iord = new int[cn];

    for (int i = 0; i <= cn; i++) // lese n+1 mal par.
    {
        par[i] = lese_double(pos);
    }

    for (int u = 0; u < cn; u++) // lese cn Kurvensegmente
    {
        iord[u] = lese_int(pos);
        double *ax = new double[iord[u]];
        double *ay = new double[iord[u]];
        double *az = new double[iord[u]];
        for (int x = 0; x < iord[u]; x++)
            ax[x] = lese_double(pos);
        for (int y = 0; y < iord[u]; y++)
            ay[y] = lese_double(pos);
        for (int z = 0; z < iord[u]; z++)
            az[z] = lese_double(pos);

        Vec3d *vector_buff = new Vec3d[iord[u]];

        for (int v = 0; v < iord[u]; v++)
        {
            vector_buff[v].set(ax[v], ay[v], az[v]);
        }

        //erzeugt Segment
        Curve_Segment csgm(u + 1, iord[u], par[u], par[u + 1], vector_buff);
        csgm_buff[u] = csgm; //Speichere segm in segm.feld
        delete[] ax;
        delete[] ay;
        delete[] az;
        delete[] vector_buff;
    }

    Curve new_curve(char_buffer, cn, iord, csgm_buff);
    curve_list.append(new_curve);
    Name new_name;
    // Speichert ID,listen-id und position
    new_name.fill(element_id, 4, curve_list.length() - 1);

    name_list.append(new_name);

    skip_lines(0, pos);

    delete[] char_buffer;
    delete[] par;
    delete[] csgm_buff;
    delete[] iord;
}

void flaeche(int &pos, string element_id)
{
    char *char_buffer;
    char_buffer = str2char(element_id);

    int i, j, k;
    int ind; // index of row vector storing the control array
    int ps = 0;
    int pt = 0;
    int nps = lese_int(pos); //Groesse der Flaeche
    int npt = lese_int(pos);

    // allocation
    int *iordu = new int[nps * npt];
    int *iordv = new int[nps * npt];

    double *pars = new double[nps + 1];
    double *part = new double[npt + 1];

    Surf_Segment *ssgm_buff = new Surf_Segment[nps * npt];

    for (i = 0; i <= nps; i++) // lese nps+1 mal pars.
    {
        pars[i] = lese_double(pos);
    }

    for (i = 0; i <= npt; i++) // lese npt+1 mal part.
    {
        part[i] = lese_double(pos);
    }

    for (i = 0; i < npt; i++) // lese nps*npt Kurvensegmente
    {
        ind = i;
        ps = 0;
        for (j = 0; j < nps; j++)
        {

            // Notice: Order in VDFAS objects different from that in data file

            iordu[ind] = lese_int(pos);
            iordv[ind] = lese_int(pos);

            int iord_ges = iordu[ind] * iordv[ind];

            // allocation
            //Zeiger auf 2D Feld
            Vec3d **vector_buff = new Vec3d *[iordu[ind]];

            double *ax = new double[iord_ges];
            double *ay = new double[iord_ges];
            double *az = new double[iord_ges];

            for (int x = 0; x < iord_ges; x++)
            {
                ax[x] = lese_double(pos);
            }
            for (int y = 0; y < iord_ges; y++)
            {
                ay[y] = lese_double(pos);
            }
            for (int z = 0; z < iord_ges; z++)
            {
                az[z] = lese_double(pos);
            }

            for (k = 0; k < iordu[ind]; k++)
            {
                // allocation
                vector_buff[k] = new Vec3d[iordv[ind]];
                for (int v = 0; v < iordv[ind]; v++)
                {
                    vector_buff[k][v].set(ax[v * iordu[ind] + k], ay[v * iordu[ind] + k], az[v * iordu[ind] + k]);

                    /*
               cerr << k << "," << v << ": ";
               vector_buff[k][v].output();
               */
                }
            }

            Surf_Segment *ssgm = new Surf_Segment(ind + 1,
                                                  iordu[ind],
                                                  iordv[ind],
                                                  pars[ps],
                                                  pars[ps + 1],
                                                  part[pt],
                                                  part[pt + 1],
                                                  vector_buff); //erzeugt Segment

            ssgm_buff[ind] = *ssgm; //Speichere segm in segm.feld

            //		for ( int uu=0; uu < iordu[ind]  ; uu++ )
            //		{
            //			for ( int vv=0; vv < iordv[ind]  ; vv++ )
            //			{
            //				vector_buff[uu][vv].output();
            //			}
            //		}

            // free
            for (k = 0; k < iordu[ind]; k++)
                delete[] vector_buff[k];
            delete[] vector_buff;
            delete[] ax;
            delete[] ay;
            delete[] az;
            delete ssgm;

            ind += npt;
            ps++;
        }
        pt++;
    }

    //	for ( int yy=0; yy < npt*nps ; yy++ )
    //	{
    //		cout << ("iu,iv: ") << iordu[yy] << (" , ") << iordv[yy];
    //	}

    Surf new_surf(char_buffer, nps, npt, iordu, iordv, ssgm_buff);
    surf_list.append(new_surf);

    Name new_name;
    // Speichert ID,listen-id und position
    new_name.fill(element_id, 5, surf_list.length() - 1);
    name_list.append(new_name);

    skip_lines(0, pos);

    // free
    delete[] char_buffer;
    delete[] pars;
    delete[] part;
    delete[] iordu;
    delete[] iordv;
    delete[] ssgm_buff;
}

void kurveauflaeche(int &pos, string element_id)
{
    char *char_buffer;
    char_buffer = str2char(element_id);

    char *snam; // Surf-Name
    char *cnam; // Curve-Name

    snam = str2char(lese_string(pos));
    cnam = str2char(lese_string(pos));

    double s1 = lese_double(pos);
    double s2 = lese_double(pos);
    int np = lese_int(pos);

    double *parp = new double[np + 1];
    //Erzeuge Feld von Curv.Seg. der Laenge n
    Cons_Segment *cosgm_buff = new Cons_Segment[np];
    int *iordp = new int[np];

    for (int i = 0; i <= np; i++) // lese n+1 mal par.
    {
        parp[i] = lese_double(pos);
    }

    for (int u = 0; u < np; u++) // lese cn Kurvensegmente
    {
        iordp[u] = lese_int(pos);
        double *as = new double[iordp[u]];
        double *at = new double[iordp[u]];
        for (int s = 0; s < iordp[u]; s++)
            as[s] = lese_double(pos);
        for (int t = 0; t < iordp[u]; t++)
            at[t] = lese_double(pos);

        Vec2d *vector_buff = new Vec2d[iordp[u]];

        for (int v = 0; v < iordp[u]; v++)
        {
            vector_buff[v].set(as[v], at[v]);
        }

        //erzeugt Segment
        Cons_Segment cosgm(u + 1, iordp[u], parp[u], parp[u + 1], vector_buff);
        cosgm_buff[u] = cosgm; //Speichere segm in segm.feld
        delete[] as;
        delete[] at;
        delete[] vector_buff;
    }

    Cons new_cons(char_buffer, snam, cnam, s1, s2, np, iordp, cosgm_buff);
    cons_list.append(new_cons);
    Name new_name;
    // Speichert ID,listen-id und position
    new_name.fill(element_id, 6, cons_list.length() - 1);

    name_list.append(new_name);

    skip_lines(0, pos);

    delete[] char_buffer;
    delete[] snam;
    delete[] cnam;
    delete[] parp;
    delete[] cosgm_buff;
    delete[] iordp;
}

void begflaeche(int &pos, string element_id)
{
    char *char_buffer;
    char_buffer = str2char(element_id);

    char *surfname;
    surfname = str2char(lese_string(pos));

    int m = lese_int(pos);

    Cons_Ensemble *conen = new Cons_Ensemble[m];

    for (int i = 0; i < m; i++) // lese m mal cons-name ein.
    {
        int j;
        int n = lese_int(pos);

        double *w1 = new double[n];
        double *w2 = new double[n];
        char **consname = new char *[n];

        for (j = 0; j < n; j++)
        {
            //consname[j] = new char[20];
            consname[j] = str2char(lese_string(pos));
            w1[j] = lese_double(pos);
            w2[j] = lese_double(pos);
        }
        //erzeugt Ensemble
        Cons_Ensemble conenbuff(n, consname, w1, w2);
        conen[i] = conenbuff; //Speichere Ens. ind Ens.Feld
        delete[] w1;
        delete[] w2;
        for (j = 0; j < n; j++)
            delete[] consname[j];
        delete[] consname;
    }

    Face new_face(char_buffer, surfname, m, conen);
    face_list.append(new_face);
    Name new_name;
    // Speichert ID,listen-id und position
    new_name.fill(element_id, 7, face_list.length() - 1);

    name_list.append(new_name);

    skip_lines(0, pos);

    delete[] conen;
    delete[] char_buffer;
}

void flaechenverband(int &pos, string element_id)
{
    int i, j, k;
    int m, n;

    char **consname;
    int *icont;
    double *w1;
    double *w2;

    Name new_name;

    char *char_buffer = str2char(element_id);
    m = lese_int(pos);

    char **fsname = new char *[2 * m];
    Cons_Ensemble *consEns = new Cons_Ensemble[2 * m];
    icont = new int[m];

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < 2; j++)
        {
            fsname[j * m + i] = str2char(lese_string(pos));
            n = lese_int(pos);
            consname = new char *[n];
            w1 = new double[n];
            w2 = new double[n];

            for (k = 0; k < n; k++)
            {
                consname[k] = str2char(lese_string(pos));
                w1[k] = lese_double(pos);
                w2[k] = lese_double(pos);
            }
            Cons_Ensemble consEnsBuff(n, consname, w1, w2);
            consEns[j * m + i] = consEnsBuff;

            for (k = 0; k < n; k++)
                delete[] consname[k];
            delete[] consname;
            delete[] w1;
            delete[] w2;
        }

        icont[i] = lese_int(pos);
    }

    Top new_top(char_buffer, m, fsname, consEns, icont);
    top_list.append(new_top);

    // store ID,list-ID and position
    new_name.fill(element_id, 10, top_list.length() - 1);
    name_list.append(new_name);

    skip_lines(0, pos);

    delete char_buffer;
    for (i = 0; i < 2 * m; i++)
        delete[] fsname[i];
    delete[] fsname;
    delete[] consEns;
    delete[] icont;
}

void gruppe(int &pos, string element_id)
{
    char *char_buffer = str2char(element_id);
    int n = lese_int(pos);
    char **grpelement = new char *[n];

    //variables must be declared outside 'for'
    //since otheriwse under Linux we get
    //name lookup of `j' changed for new ANSI `for' scoping
    int j;
    for (j = 0; j < n; j++) // einlesen von n IDs
    {
        grpelement[j] = str2char(lese_string(pos));
    }

    Group new_group(char_buffer, n, grpelement);
    group_list.append(new_group);
    Name new_name;
    // Speichert ID,listen-id und position
    new_name.fill(element_id, 8, group_list.length() - 1);

    name_list.append(new_name);

    skip_lines(0, pos);

    delete[] char_buffer;
    for (j = 0; j < n; j++)
        delete[] grpelement[j];
    delete[] grpelement;
}

void vdafs_set(int &pos, string element_id, list<string> &elem_names)
{
    char *char_buffer = str2char(element_id);

    Set new_set(char_buffer, elem_names);
    set_list.append(new_set);

    Name new_name;
    new_name.fill(element_id, 0, set_list.length() - 1);
    name_list.append(new_name);

    skip_lines(0, pos);

    delete[] char_buffer;
}

int typ2int(string typstr) //umordnen nach haeufigkeit xxxxx
{

    if (typstr == "BEGINSET")
        return (0);
    if (typstr == "HEADER")
        return (1);
    if (typstr == "POINT")
        return (2);
    if (typstr == "CIRCLE")
        return (3);
    if (typstr == "CURVE")
        return (4);
    if (typstr == "SURF")
        return (5);
    if (typstr == "CONS")
        return (6);
    if (typstr == "FACE")
        return (7);
    if (typstr == "GROUP")
        return (8);
    if (typstr == "ENDSET")
        return (9);
    if (typstr == "TOP")
        return (10);
    if (typstr == "END")
        return (20);

    return 99;
}

int naechstes_element(Status &setFlag) //liest ID und Typ ein und verzweigt
{

    static list<string> set_elem;
    list_item it;
    Face face;
    Cons cons;
    int pos = 1;

    string element_id = lese_string(pos); //ID lesen

    int inttyp = 99;

    string element_typ;

    for (; inttyp == 99;) //suche bekanntes Objekt(typ)
    {
        element_typ = lese_string(pos);
        inttyp = typ2int(element_typ);
        if (inttyp == 99)
        {
            cerr << ".";
            element_id = unbekannt(pos); //neue ID lesen
        }
    }
    switch (typ2int(element_typ))
    {

    case 0: //BEGINSET
        skip_lines(0, pos);
        setFlag = begin_set;
        cerr << "b";
        break;
    case 1: //HEADER
        skip_lines(lese_int(pos), pos);
        cerr << "h";
        break;
    case 2: //POINT
        punkt(pos, element_id);
        cerr << "p";
        if (setFlag)
            set_elem.append(element_id);
        break;
    case 3: //CIRCLE
        kreis(pos, element_id);
        cerr << "O";
        if (setFlag)
            set_elem.append(element_id);
        break;
    case 4: //CURVE
        kurve(pos, element_id);
        cerr << "K";
        if (setFlag)
            set_elem.append(element_id);
        break;
    case 5: //SURF
        flaeche(pos, element_id);
        cerr << "S";
        if (setFlag)
            set_elem.append(element_id);
        break;
    case 6: //CONS
        kurveauflaeche(pos, element_id);
        cerr << "C";
        if (setFlag)
            set_elem.append(element_id);
        // solve possible conflicts within the set (CONS >-< CURVE)
        cons = cons_list.tail();
        it = set_elem.search(cons.get_curvenme());
        if (it != NULL)
        {
            set_elem.del_item(it);
        }
        break;
    case 7: //FACE
        begflaeche(pos, element_id);
        cerr << "F";
        if (setFlag)
            set_elem.append(element_id);
        // solve possible conflicts within the set (FACE >-< SURF)
        face = face_list.tail();
        it = set_elem.search(face.get_surfnme());
        if (it != NULL)
        {
            set_elem.del_item(it);
        }
        break;
    case 8: //GROUP
        gruppe(pos, element_id);
        cerr << "G";
        if (setFlag)
            set_elem.append(element_id);
        break;
    case 9: //ENDSET
        vdafs_set(pos, element_id, set_elem);
        cerr << "e";
        setFlag = end_set;
        set_elem.clear();
        break;
    case 10: //TOP
        flaechenverband(pos, element_id);
        cerr << "T";
        if (setFlag)
            set_elem.append(element_id);
        break;
    case 20: //END
        return 0;
    }

    return 1;
}

Name oldgetname(string element_id)
{
    int l_length = name_list.length();
    Name name_buffer, foundname;
    string aktname;

    for (int posi = 0; posi < l_length; posi++)
    {
        name_buffer = name_list.contents(name_list.item(posi));
        aktname = name_buffer.getid();

        if (element_id == aktname)
        {
            l_length = 0;
            foundname = name_buffer;
        }
    }

    return foundname;
}

Name getname(string look_for)
{
    int len_name = name_list.length();
    string at_pos;
    int i_start, i_end, i_midd, found_pos;
    Name temp_name;

    i_start = 0;
    i_end = len_name;
    i_midd = (i_start + i_end) / 2;

    found_pos = 0;

    for (; found_pos == 0;)
    {
        // cerr << ".";
        temp_name = name_list.contents(name_list.item(i_midd));
        at_pos = temp_name.getid();

        if (at_pos == look_for)
        {
            found_pos = 1;
        }

        if (at_pos > look_for)
        {
            i_end = i_midd;
            i_midd = (i_start + i_end) / 2;
        }

        if (at_pos < look_for)
        {
            i_start = i_midd;
            i_midd = (i_start + i_end) / 2;
        }
    }

    return temp_name;
}

void sort_name()
{
    list<Name> sorted_list;
    int len_name = name_list.length();
    string look_for, at_pos, bef_pos;
    int i_start, i_end, i_midd, found_pos, len_sort;
    Name akt_name, temp_name;

    for (int el = 0; el < len_name; el++)
    {
        found_pos = 0;
        len_sort = sorted_list.length();
        i_start = 0;
        i_end = len_sort;
        i_midd = (i_start + i_end) / 2;

        akt_name = name_list.contents(name_list.item(el));
        look_for = akt_name.getid();

        /*	
      if ( el == 241 )
      {
              // cerr << "241 : ";
         temp_name = oldgetname(look_for);
         // cerr << temp_name << "\n\n";

         for (int xxx=0; xxx < len_sort ; xxx++)
         {
            temp_name = sorted_list.contents(sorted_list.item(xxx));
            // cerr << xxx << " -   " << temp_name << "\n";
      }
      }
      */

        //		found_pos = 0;
        // cerr << "\nElem.: " << el << " -Laengen der Listen sort, unsort:"
        //      << len_sort << " , " << len_name << "\n";
        if (len_sort == 0)
        {
            // cerr << "ok, initializing list\n";
            found_pos = 1;
            temp_name.fill("0", 1, 1);
            sorted_list.append(temp_name);
            sorted_list.append(akt_name);
            temp_name.fill("ZZZZZZZZZZZZZZZ", 1, 1);
            sorted_list.append(temp_name);
        }

        for (; found_pos == 0;)
        {
            // cerr << ".";
            temp_name = sorted_list.contents(sorted_list.item(i_midd));
            at_pos = temp_name.getid();
            temp_name = sorted_list.contents(sorted_list.item(i_midd - 1));
            bef_pos = temp_name.getid();

            if ((at_pos > look_for) && (bef_pos < look_for))
            {
                found_pos = 1;
            }

            if ((at_pos > look_for) && (bef_pos > look_for))
            {
                i_end = i_midd;
                i_midd = (i_start + i_end) / 2;
            }

            if ((at_pos < look_for) && (bef_pos < look_for))
            {
                i_start = i_midd;
                i_midd = (i_start + i_end) / 2;
            }
        }

        if (len_sort != 0)
        {
            sorted_list.insert(akt_name, sorted_list.item(i_midd), before);
        }
    }

    name_list.clear();
    name_list = sorted_list;
}

void pars(const int fd)
{
    Status setFlag = end_set; // flag for VDAFS element SET
    from = new ifstream(fd);
    cout.precision(12);
    int stp = 1;

    for (int elm = 1; stp != 0; elm++)
    {
        stp = naechstes_element(setFlag);
    }
    //int l = name_list.length();

    //cout << ("\nListe der Namen mit ") << l << (" Elementen.\n\n");
    close(fd);

    delete from;
}
