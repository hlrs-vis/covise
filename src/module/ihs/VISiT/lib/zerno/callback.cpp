#include "callback.h"

sendgeodata_func sendgeodata_ptr;
sendrbedata_func sendrbedata_ptr;


void set_sendgeodata_func(sendgeodata_func f)
{
   sendgeodata_ptr = f;
}

void set_sendrbedata_func(sendrbedata_func f)
{
   sendrbedata_ptr = f;
}

	
void sendgeodata_(int *igeb, int *npoin_ges, int *nelem_ges, int *knmax_num, int *elmax_num,
							 int *nkn, int *nel, int *ncd, int *nkd, float *cov_coord,
							 int *cov_lnods, int *cov_lnods_num, int *cov_lnods_proz,
							 int *cov_coord_num, int *cov_coord_joi, int *cov_lnods_joi, int *cov_coord_mod,
							 int *cov_lnods_mod, int *cov_coord_proz)
{
   if(sendgeodata_ptr)
      sendgeodata_ptr(igeb, npoin_ges, nelem_ges, knmax_num, elmax_num,
            nkn, nel, ncd, nkd, cov_coord,
            cov_lnods, cov_lnods_num, cov_lnods_proz,
            cov_coord_num, cov_coord_joi, cov_lnods_joi, cov_coord_mod,
            cov_lnods_mod, cov_coord_proz);
}

void sendrbedata_(int *igeb,int *nrbpo_geb,int *nwand_geb,int *npres_geb,
							 int *nsyme_geb, int *nconv_geb,int *nrbknie,
							 int *cov_displ_kn,int *cov_displ_typ,
							 int *cov_wand_el,int *cov_wand_kn,int *cov_wand_num,
							 int *cov_pres_el,int *cov_pres_kn,int *cov_pres_num,
							 int *cov_conv_el,int *cov_conv_kn,int *cov_conv_num,
							 float *cov_displ_wert, int *reicheck)
{
   if(sendrbedata_ptr)
      sendrbedata_ptr(igeb, nrbpo_geb, nwand_geb, npres_geb,
            nsyme_geb, nconv_geb, nrbknie,
            cov_displ_kn, cov_displ_typ,
            cov_wand_el, cov_wand_kn, cov_wand_num,
            cov_pres_el, cov_pres_kn, cov_pres_num,
            cov_conv_el, cov_conv_kn, cov_conv_num,
            cov_displ_wert, reicheck);
}


