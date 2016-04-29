C**************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE PRINT_RBE_COV(elra_kno,elra_ele,elra_num,elra_wer,
     *                     elra_adr,
     *                     knra_kno,knra_mat,knra_wer,knra_adr,
     *                     lnods_num,coord_num,
     *                     kern_kn,kern_kn_adr,
     *                     lapp_kn,lapp_kn_adr,
     *                     dopp_kn,dopp_kn_adr,
     *                     kern_el,kern_el_adr,
     *                     lapp_el,lapp_el_adr,
     *                     dopp_el,dopp_el_adr,
     *                     nkern_max,nlapp_el,nlapp_kn,
     *                     ndopp_el,ndopp_kn,
     *                     zeig_kno,zeig_ele,
     *                     rand_dim,ndat_max,rbe_pfad,rbe_name,
     *                     dopplapp,
     *                     covise_run,
     *                     write_files,
     *                     cov_displ_wert,cov_displ_kn,cov_displ_typ,
     *                     cov_wand_el,cov_wand_kn,cov_wand_num,
     *                     cov_syme_el,cov_syme_kn,
     *                     cov_pres_el,cov_pres_kn,cov_pres_num, 
     *                     cov_conv_el,cov_conv_kn,cov_conv_num)

      implicit none 

      include 'common.zer'
      include 'mpif.h'

      integer   elra_kno,elra_ele,elra_num,elra_adr,
     *          knra_kno,knra_mat,knra_adr,
     *          lnods_num,coord_num,
     *          zeig_kno,zeig_ele

      integer   rand_dim,ndat_max

C ----- COVISE
      integer   covise_run, write_files, reicheck
      integer   cov_displ_kn,cov_displ_typ,
     *          cov_wand_el,cov_wand_kn,cov_wand_num,
     *          cov_syme_el,cov_syme_kn,
     *          cov_pres_el,cov_pres_kn,cov_pres_num,
     *          cov_conv_el,cov_conv_kn,cov_conv_num
      real      cov_displ_wert
      dimension cov_displ_kn(nrbpoi),cov_displ_wert(nrbpoi),
     *          cov_displ_typ(nrbpoi),
     *          cov_wand_el(nwand),cov_wand_kn(nwand*nrbknie),
     *          cov_wand_num(nwand),
     *          cov_syme_el(nsyme),cov_syme_kn(nsyme*nrbknie),
     *          cov_pres_el(npres),cov_pres_kn(npres*nrbknie),
     *          cov_pres_num(npres),
     *          cov_conv_el(nconv),cov_conv_kn(nconv*nrbknie),
     *          cov_conv_num(nconv)
      


      integer  kern_kn,kern_kn_adr,
     *         lapp_kn,lapp_kn_adr,
     *         dopp_kn,dopp_kn_adr,
     *         kern_el,kern_el_adr,
     *         lapp_el,lapp_el_adr,
     *         dopp_el,dopp_el_adr,
     *         nkern_max,nlapp_el,nlapp_kn,
     *         ndopp_el,ndopp_kn

      integer   nrbpo_geb,nwand_geb,npres_geb,nsyme_geb,
     *          nzykl_geb,nconv_geb,ntemp_geb

      real      elra_wer,knra_wer

      integer i,igeb,lu,luerr,
     *        ip1,ip2,ip3,ip4,ipfad,lentb

      character*80 rbe_name,rbe_pfad,file_name,comment
      character*4   otto  

      logical  schreiben,dopplapp

      parameter (lu=50)

      dimension elra_kno(nelra_max,nrbknie),elra_ele(nelra_max),
     *          elra_num(nelra_max),elra_wer(nelra_max),
     *          elra_adr(ntyp_elra+1)

      dimension knra_kno(nknra_max),knra_wer(nknra_max,nwer_knra),
     *          knra_mat(nknra_max,nwer_knra),
     *          knra_adr(ntyp_knra+1)

      dimension lnods_num(nelem_max),coord_num(npoin_max),
     *          zeig_kno(npoin_max),zeig_ele(npoin_max),
     *          rand_dim(ngebiet,ndat_max)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1),
     *          lapp_kn(nlapp_kn),lapp_kn_adr(ngebiet+1)

      dimension dopp_el(ndopp_el),dopp_el_adr(ngebiet+1),
     *          dopp_kn(ndopp_kn),dopp_kn_adr(ngebiet+1)
c     ****************************************************************


c     ****************************************************************
c     DIMENSIONSKONTROLLE:

      if (ntyp_knra+ntyp_elra.gt.ndat_max) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PRINT_RBE'
         write(luerr,*)'Die Dimenion des Hilfsfeldes rand_dim '
         write(luerr,*)'ist zu klein.          '
         write(luerr,*)'Dimensioniert:',ndat_max   
         write(luerr,*)'Benoetigt    :',ntyp_knra+ntyp_elra
         call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************

	  
c     **********************************************************
c     BESCHRIFTEN DER FILENAMEN:

      file_name=rbe_pfad
      ipfad=lentb(rbe_pfad)
      ip1=ipfad+1
      ip2=ipfad+4
      ip3=ip2+1
      ip4=ip3+3
      file_name(ip1:ip2)='RBE_'
c     **********************************************************


c     **********************************************************
c     GEBIETSWEISER AUSDRUCK DER RANDBEDINGUNGEN: 

      do 101 i=1,npoin_max
         zeig_kno(i)=0
         zeig_ele(i)=0
 101  continue

      do 100 igeb=1,ngebiet
         if (write_files.eq.1) then
            write(otto,'(i4.4)') igeb
            file_name(ip3:ip4)=otto(1:4)
            open(lu,file=file_name,status='unknown',err=777)

            comment='# Partition von:'    
            write(lu,*)'#                                      '
            write(lu,*)'#                                      '
            write(lu,*)'#                                      '
            write(lu,*)'#                                      '
            call char_druck(comment,rbe_name,lu)
            write(lu,*)'#                                      '
            write(lu,*)'#                                      '
            write(lu,*)'#                                      '
            write(lu,*)'#                                      '
            write(lu,*)'#                                      '
         endif

c        Markieren der Kern-Knoten von Gebiet igeb:
         do 103 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
            zeig_kno(kern_kn(i))=1
 103     continue

c        Markieren der Halo-Knoten von Gebiet igeb:
         do 104 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            zeig_kno(lapp_kn(i))=1
 104     continue

c        Markieren der Kern-Elemente von Gebiet igeb:
         do 105 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            zeig_ele(kern_el(i))=1
 105     continue

c        Markieren der Halo-Elemente von Gebiet igeb:
         do 106 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            zeig_ele(lapp_el(i))=1
 106     continue
         
         if (dopplapp) then
c        Markieren der doppelt ueberlappenden Knoten von Gebiet igeb:
            do 107 i=dopp_kn_adr(igeb),dopp_kn_adr(igeb+1)-1
               zeig_kno(dopp_kn(i))=1
 107        continue

c        Markieren der DOPPLAPP-Elemente von Gebiet igeb:
            do 108 i=dopp_el_adr(igeb),dopp_el_adr(igeb+1)-1
               zeig_ele(dopp_el(i))=1
 108        continue
         endif

         schreiben=.false.
         CALL EXTRA_RBE_COV(elra_kno,elra_ele,elra_num,elra_wer,
     *                  elra_adr,
     *                  knra_kno,knra_mat,knra_wer,knra_adr,
     *                  lnods_num,coord_num,
     *                  zeig_kno,zeig_ele,lu,schreiben,
     *                  nrbpo_geb,nwand_geb,npres_geb,nsyme_geb,
     *                  nzykl_geb,nconv_geb,ntemp_geb,
     *                  cov_displ_wert,cov_displ_kn,cov_displ_typ,
     *                  cov_wand_el,cov_wand_kn,cov_wand_num,
     *                  cov_syme_el,cov_syme_kn,
     *                  cov_pres_el,cov_pres_kn,
     *                  cov_pres_num, cov_conv_el,cov_conv_kn,
     *                  cov_conv_num)
         
		 if (write_files.eq.1) then
            write(lu,199) nrbpo_geb,nwand_geb,npres_geb,nsyme_geb,
     *                 nzykl_geb,0,nconv_geb,ntemp_geb
	     endif

         write(*,*) 'Dimensionen berechnet ...'

 199     format(10(i6,1x))

         rand_dim(igeb,1)=nrbpo_geb
         rand_dim(igeb,2)=nwand_geb
         rand_dim(igeb,3)=npres_geb
         rand_dim(igeb,4)=nsyme_geb
         rand_dim(igeb,5)=nzykl_geb
         rand_dim(igeb,6)=nconv_geb
         rand_dim(igeb,7)=ntemp_geb

         schreiben=.true.
         CALL EXTRA_RBE_COV(elra_kno,elra_ele,elra_num,elra_wer,
     *                  elra_adr,
     *                  knra_kno,knra_mat,knra_wer,knra_adr,
     *                  lnods_num,coord_num,
     *                  zeig_kno,zeig_ele,lu,schreiben,
     *                  nrbpo_geb,nwand_geb,npres_geb,nsyme_geb,
     *                  nzykl_geb,nconv_geb,ntemp_geb,
     *                  cov_displ_wert,cov_displ_kn,cov_displ_typ,
     *                  cov_wand_el,cov_wand_kn,cov_wand_num,
     *                  cov_syme_el,cov_syme_kn,
     *                  cov_pres_el,cov_pres_kn,
     *                  cov_pres_num, cov_conv_el,cov_conv_kn,
     *                  cov_conv_num)

         reicheck = 999
         write(*,*) 'Sende Geometrie an COVISE (Gebiet:',igeb,')'
         CALL TEST_PRINT_RBE(cov_displ_kn,cov_displ_typ,
     *                          cov_displ_wert,
     *                          cov_wand_el,cov_wand_kn,cov_wand_num,
     *                          cov_syme_el,cov_syme_kn,
     *                          cov_pres_el,cov_pres_kn,cov_pres_num,
     *                          cov_conv_el,cov_conv_kn,cov_conv_num,
     *                          nrbpo_geb,nwand_geb,npres_geb,
     *                          nsyme_geb,nconv_geb,
     *                          nrbknie,ncd,igeb,
     *                          myid,parallel)
         CALL sendrbedata(igeb,nrbpo_geb,nwand_geb,npres_geb,
     *           nsyme_geb,
     *           nconv_geb,nrbknie,
     *           cov_displ_kn,cov_displ_typ,
     *           cov_wand_el,cov_wand_kn,cov_wand_num,
     *           cov_pres_el,cov_pres_kn,cov_pres_num,
     *           cov_conv_el,cov_conv_kn,cov_conv_num,
     *           cov_displ_wert,reicheck)

c        Loeschen der markierten Kern-Knoten von Gebiet igeb:
         do 113 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
            zeig_kno(kern_kn(i))=0
 113     continue

c        Loeschen der markierten Halo-Knoten von Gebiet igeb:
         do 114 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            zeig_kno(lapp_kn(i))=0
 114     continue

c        Loeschen der markierten Kern-Elemente von Gebiet igeb:
         do 115 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            zeig_ele(kern_el(i))=0
 115     continue

c        Loeschen der markierten Halo-Elemente von Gebiet igeb:
         do 116 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            zeig_ele(lapp_el(i))=0
 116     continue
         
         if (dopplapp) then
c        Loeschen der doppelt ueberlappenden Knoten von Gebiet igeb:
            do 117 i=dopp_kn_adr(igeb),dopp_kn_adr(igeb+1)-1
               zeig_kno(dopp_kn(i))=0
 117        continue

c        Loeschen der DOPPLAPP-Elemente von Gebiet igeb:
            do 118 i=dopp_el_adr(igeb),dopp_el_adr(igeb+1)-1
               zeig_ele(dopp_el(i))=0
 118        continue
         endif

         if (write_files.eq.1) then
            close(lu) 
            comment='File geschrieben:'
            call char_druck(comment,file_name,6)
         endif

 100  continue
      write(6,*)
c     **********************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (print_rbe):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************


      return
      end
