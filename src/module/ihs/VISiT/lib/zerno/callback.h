typedef void (*sendgeodata_func)(int *igeb, int *npoin_ges, int *nelem_ges, int *knmax_num, int *elmax_num,
							 int *nkn, int *nel, int *ncd, int *nkd, float *cov_coord,
							 int *cov_lnods, int *cov_lnods_num, int *cov_lnods_proz,
							 int *cov_coord_num, int *cov_coord_joi, int *cov_lnods_joi, int *cov_coord_mod,
							 int *cov_lnods_mod, int *cov_coord_proz);

typedef void (*sendrbedata_func)(int *igeb,int *nrbpo_geb,int *nwand_geb,int *npres_geb,
							 int *nsyme_geb, int *nconv_geb,int *nrbknie,
							 int *cov_displ_kn,int *cov_displ_typ,
							 int *cov_wand_el,int *cov_wand_kn,int *cov_wand_num,
							 int *cov_pres_el,int *cov_pres_kn,int *cov_pres_num,
							 int *cov_conv_el,int *cov_conv_kn,int *cov_conv_num,
							 float *cov_displ_wert, int *reicheck);


extern "C" void set_sendgeodata_func(sendgeodata_func f);
extern "C" void set_sendrbedata_func(sendrbedata_func f);


extern "C" void sendgeodata_(int *igeb, int *npoin_ges, int *nelem_ges, int *knmax_num, int *elmax_num,
							 int *nkn, int *nel, int *ncd, int *nkd, float *cov_coord,
							 int *cov_lnods, int *cov_lnods_num, int *cov_lnods_proz,
							 int *cov_coord_num, int *cov_coord_joi, int *cov_lnods_joi, int *cov_coord_mod,
							 int *cov_lnods_mod, int *cov_coord_proz);

extern "C" void sendrbedata_(int *igeb,int *nrbpo_geb,int *nwand_geb,int *npres_geb,
							 int *nsyme_geb, int *nconv_geb,int *nrbknie,
							 int *cov_displ_kn,int *cov_displ_typ,
							 int *cov_wand_el,int *cov_wand_kn,int *cov_wand_num,
							 int *cov_pres_el,int *cov_pres_kn,int *cov_pres_num,
							 int *cov_conv_el,int *cov_conv_kn,int *cov_conv_num,
							 float *cov_displ_wert, int *reicheck);


