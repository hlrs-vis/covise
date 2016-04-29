C**************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE EXTRA_RBE_COV(elra_kno,elra_ele,elra_num,elra_wer,
     *                     elra_adr,
     *                     knra_kno,knra_mat,knra_wer,knra_adr,
     *                     lnods_num,coord_num,
     *                     zeig_kno,zeig_ele,lu,schreiben,
     *                     nrbpo_geb,nwand_geb,npres_geb,nsyme_geb,
     *                     nzykl_geb,nconv_geb,ntemp_geb,
     *                     cov_displ_wert,cov_displ_kn,cov_displ_typ,
     *                     cov_wand_el,cov_wand_kn,cov_wand_num,
     *                     cov_syme_el,cov_syme_kn,
     *                     cov_pres_el,cov_pres_kn,
     *                     cov_pres_num, cov_conv_el,cov_conv_kn,
     *                     cov_conv_num)
C
      implicit none 

      include 'common.zer'
      include 'mpif.h'

      integer   elra_kno,elra_ele,elra_num,elra_adr,
     *          knra_kno,knra_mat,knra_adr,
     *          lnods_num,coord_num,
     *          zeig_kno,zeig_ele

      integer   nrbpo_geb,nwand_geb,npres_geb,nsyme_geb,
     *          nzykl_geb,nconv_geb,ntemp_geb

C ----- COVISE
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
      integer   nnn
      integer   cnrbpo_geb,cnwand_geb,cnpres_geb,cnsyme_geb,
     *          cnzykl_geb,cnconv_geb,cntemp_geb

      real      elra_wer,knra_wer,wer

      integer i,j,ityp,lu,ele,num,help(8)
      integer   luerr

      logical  schreiben

      character*4 rbe_com

      dimension elra_kno(nelra_max,nrbknie),elra_ele(nelra_max),
     *          elra_num(nelra_max),elra_wer(nelra_max),
     *          elra_adr(ntyp_elra+1)

      dimension knra_kno(nknra_max),knra_wer(nknra_max,nwer_knra),
     *          knra_mat(nknra_max,nwer_knra),
     *          knra_adr(ntyp_knra+1)

      dimension lnods_num(nelem_max),coord_num(npoin_max),
     *          zeig_kno(npoin_max),zeig_ele(npoin_max)
c     ****************************************************************


      if (schreiben) then
          cnrbpo_geb=nrbpo_geb
          cnwand_geb=nwand_geb
c mb 03.12.2003: jetzt gibt es Druckrandbedingungen auch mit Covise
          cnpres_geb=npres_geb
c alt     cnpres_geb=nwand_geb
          cnsyme_geb=nsyme_geb
          cnzykl_geb=nzykl_geb
          cnconv_geb=nconv_geb
          cntemp_geb=ntemp_geb
      endif

c     *****************************************************************
c     DISP-RANDBEDINGUNGEN:

      nrbpo_geb=0
      nnn=0
      do 100 i=knra_adr(1),knra_adr(2)-1
         IF (zeig_kno(knra_kno(i)).ne.0) THEN

            do 110 ityp=1,nwer_knra
               if (knra_mat(i,ityp).ne.0) then
                  nrbpo_geb=nrbpo_geb+1
                  
                  if (schreiben) then
                     write(lu,101) coord_num(knra_kno(i)),ityp,
     *                             knra_wer(i,ityp)
                     nnn=nnn+1
                     cov_displ_kn(nnn)=coord_num(knra_kno(i))
                     cov_displ_typ(nnn)=ityp
                     cov_displ_wert(nnn)=knra_wer(i,ityp)
                  endif

               endif
 110        continue

         ENDIF                                    
 100  continue
c     *****************************************************************


c     *****************************************************************
c     WAND-RANDBEDINGUNGEN:

      rbe_com='wand'
      nwand_geb=0
      nnn=0
      do 200 i=elra_adr(1),elra_adr(2)-1
         IF (zeig_ele(elra_ele(i)).ne.0) THEN
            nwand_geb=nwand_geb+1

            ele=lnods_num(elra_ele(i))
            num=elra_num(i)
            wer=elra_wer(i)
            do 210 j=1,nrbknie
               help(j)=coord_num(elra_kno(i,j))
 210        continue

            if (ncd.eq.2.and.schreiben) then
               write(lu,202) (help(j),j=1,nrbknie),ele,num,wer,rbe_com    
            else if (ncd.eq.3.and.schreiben) then
               write(lu,203) (help(j),j=1,nrbknie),ele,num,wer,rbe_com   
               nnn=nnn+1
               cov_wand_el(nnn) =ele
               cov_wand_num(nnn)=num
               do 211 j=1,nrbknie
C                  if (nwand_geb.lt.3) then
C                      write(*,*) '??????: ',cnwand_geb,
C    *                 j, nnn,
C    *                 (((j-1)*cnwand_geb)+nnn),help(j)
C                  endif
                   cov_wand_kn(((j-1)*cnwand_geb)+nnn)=help(j)
 211           continue
            endif
         ENDIF                                    
 200  continue
c     *****************************************************************


c     *****************************************************************
c     DRUCK-RANDBEDINGUNGEN:

      rbe_com='pres'
      npres_geb=0
      nnn=0
      do 300 i=elra_adr(2),elra_adr(3)-1
         IF (zeig_ele(elra_ele(i)).ne.0) THEN
            npres_geb=npres_geb+1

            ele=lnods_num(elra_ele(i))
            num=elra_num(i)
            wer=elra_wer(i)
            do 310 j=1,nrbknie
               help(j)=coord_num(elra_kno(i,j))
 310        continue

            if (ncd.eq.2.and.schreiben) then
               write(lu,202) (help(j),j=1,nrbknie),ele,num,wer,rbe_com
            else if (ncd.eq.3.and.schreiben) then
               write(lu,203) (help(j),j=1,nrbknie),ele,num,wer,rbe_com
               nnn=nnn+1
               cov_pres_el(nnn)=ele
               cov_pres_num(nnn)=num
               do 311 j=1,nrbknie
                   cov_pres_kn(((j-1)*cnpres_geb)+nnn)=help(j)
 311           continue
            endif
         ENDIF                                    
 300  continue
c     *****************************************************************


c     *****************************************************************
c     SYMMETRIE-RANDBEDINGUNGEN:

      nsyme_geb=0
      nnn=0
      do 400 i=elra_adr(3),elra_adr(4)-1
         IF (zeig_ele(elra_ele(i)).ne.0) THEN
            nsyme_geb=nsyme_geb+1

            ele=lnods_num(elra_ele(i))
            num=elra_num(i)
            wer=elra_wer(i)
            do 410 j=1,nrbknie
               help(j)=coord_num(elra_kno(i,j))
 410        continue

            if (ncd.eq.2.and.schreiben) then
               write(lu,402) (help(j),j=1,nrbknie),ele
            else if (ncd.eq.3.and.schreiben) then
               write(lu,403) (help(j),j=1,nrbknie),ele
               nnn=nnn+1
               cov_syme_el(nnn)=ele
               do 411 j=1,nrbknie
                  cov_syme_kn(((j-1)*cnsyme_geb)+nnn)=help(j)
 411           continue
            endif
         ENDIF                                    
 400  continue
c     *****************************************************************



c     *****************************************************************
c     PERIODISCHE-RANDBEDINGUNGEN:

      nzykl_geb=0
      nnn=0
      do 500 i=elra_adr(4),elra_adr(5)-1
         IF (zeig_ele(elra_ele(i)).ne.0) THEN
            nzykl_geb=nzykl_geb+1

            ele=lnods_num(elra_ele(i))
            num=elra_num(i)
            wer=elra_wer(i)
            do 510 j=1,nrbknie
               help(j)=coord_num(elra_kno(i,j))
 510        continue

            if (ncd.eq.2.and.schreiben) then
               write(lu,502) (help(j),j=1,nrbknie),wer,ele
            else if (ncd.eq.3.and.schreiben) then
               write(lu,503) (help(j),j=1,nrbknie),wer,ele
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'COVISE-version:'
               write(luerr,*)'Periodische RB nicht moeglich'
               call erro_ende(myid,parallel,luerr)
            endif
         ENDIF                                    
 500  continue
c     *****************************************************************


c     *****************************************************************
c old     CONV-RANDBEDINGUNGEN:
c      BILA-RANDBEDINGUNGEN:

      rbe_com='bila'
      nconv_geb=0
      nnn=0
      do 600 i=elra_adr(5),elra_adr(6)-1
         IF (zeig_ele(elra_ele(i)).ne.0) THEN
            nconv_geb=nconv_geb+1

            ele=lnods_num(elra_ele(i))
            num=elra_num(i)
            wer=elra_wer(i)
            do 610 j=1,nrbknie
               help(j)=coord_num(elra_kno(i,j))
 610        continue

            if (ncd.eq.2.and.schreiben) then
               write(lu,202) (help(j),j=1,nrbknie),ele,num,wer,rbe_com
            else if (ncd.eq.3.and.schreiben) then
               write(lu,203) (help(j),j=1,nrbknie),ele,num,wer,rbe_com
               nnn=nnn+1
               cov_conv_el(nnn)=ele
               cov_conv_num(nnn)=num
               do 611 j=1,nrbknie
c                     cov_conv_kn(nnn,j)=help(j)
                  cov_conv_kn(((j-1)*cnconv_geb)+nnn)=help(j)
 611           continue 
            endif
         ENDIF                                    
 600  continue
c     *****************************************************************


c     *****************************************************************
c     TEMP-RANDBEDINGUNGEN:

      ntemp_geb=0
      nnn=0
      do 700 i=knra_adr(2),knra_adr(3)-1
         IF (zeig_kno(knra_kno(i)).ne.0) THEN
            ntemp_geb=ntemp_geb+1
            if (schreiben) then
               write(lu,701) coord_num(knra_kno(i)),knra_wer(i,1)
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'Fehler in Routine EXTRA_RBE'
               write(luerr,*)'COVISE: Keine temp-RB moeglich'
               call erro_ende(myid,parallel,luerr)
            endif
         ENDIF                                    
 700  continue
c     *****************************************************************


c     ****************************************************************
c     FORMATE:
 
 101  format (I8,1x,I3,1x,F15.6)

 202  format (2(I8,1x),1x,I8,1x,I4,1x,f12.4,1x,A4)
 203  format (4(I8,1x),1x,I8,1x,I4,1x,f12.4,1x,A4)

 302  format (2(I7,1x),4x,i3,4x,f15.6,4x,I7)
 303  format (4(I7,1x),4x,i3,4x,f15.6,4x,I7)

 402  format (2(I8,1x),I8)
 403  format (4(I8,1x),I8)

 502  format (2(I8,1x),F15.6,1x,I8)
 503  format (4(I8,1x),F15.6,1x,I8)

 602  format (2(I8,1x),I8,1x,F15.6)
 603  format (4(I8,1x),I8,1x,F15.6)

 701  format(i8,1x,f15.6)
c     ****************************************************************

      return
      end






