C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE TEST_PRINT_RBE(displ_kn,displ_typ,displ_wert,
     *                          wand_el,wand_kn,wand_num,
     *                          syme_el,syme_kn,
     *                          pres_el,pres_kn,pres_num,
     *                          conv_el,conv_kn,conv_num,
     *                          nrbpoi,nwand,npres,nsyme,nconv,
     *                          nrbknie,ncd,igeb,
     *                          myid,parallel)
C
      implicit none 

      include 'mpif.h'

      integer  displ_kn,displ_typ,
     *         wand_el,wand_kn,wand_num,
     *         syme_el,syme_kn,
     *         pres_el,pres_kn,pres_num,
     *         conv_el,conv_kn,conv_num

      integer  nrbpoi,nwand,npres,nsyme,nconv,
     *         nrbknie,ncd,myid,igeb

      integer  i,j,lu,luerr

      real  displ_wert

      logical parallel

      character*4  otto     
      character*80 file_name,comment

      parameter(lu=49)

      dimension displ_kn(nrbpoi),displ_typ(nrbpoi),displ_wert(nrbpoi),
     *          wand_el(nwand),wand_kn(nwand,nrbknie),wand_num(nwand),
     *          syme_el(nsyme),syme_kn(nsyme,nrbknie),
     *          pres_el(npres),pres_kn(npres,nrbknie),pres_num(npres),
     *          conv_el(nconv),conv_kn(nconv,nrbknie),conv_num(nconv)
c     ****************************************************************


c     ****************************************************************
c     OEFFNEN DER FILES:

      write(otto,'(i4.4)') igeb
      file_name='zer/TEST_RBE_0000'
      file_name(14:17)=otto(1:4)
      open(lu,file=file_name,err=777)

      write(lu,*)'#                                      '
      write(lu,*)'#                                      '
      write(lu,*)'#                                      '
      write(lu,*)'# TEST_PRINT_RBE()                     '
      write(lu,*)'# ZERNO                                '
      write(lu,*)'#                                      '
      write(lu,*)'#                                      '
      write(lu,*)'#                                      '
      write(lu,*)'#                                      '
      write(lu,*)'#                                      '
      write(lu,99) nrbpoi,nwand,npres,nsyme,0,0,nconv,0     
 99   format(10(i6,1x))
c     ****************************************************************


c     ****************************************************************
c     DISPL-RANDBEDINGUNGEN:

      do 100 i=1,nrbpoi
         write(lu,101) displ_kn(i),displ_typ(i),displ_wert(i)
 100  continue
c     ****************************************************************


c     ****************************************************************
c     WAND-RANDBEDINGUNGEN:

      do 200 i=1,nwand                    

         if (ncd.eq.2) then
            write(lu,202) (wand_kn(i,j),j=1,nrbknie),
     *                     wand_el(i),wand_num(i)
         else if (ncd.eq.3) then
            write(lu,203) (wand_kn(i,j),j=1,nrbknie),
     *                     wand_el(i),wand_num(i)
         endif

 200  continue
c     ****************************************************************

c     ****************************************************************
      do 300 i=1,npres

         if (ncd.eq.2) then
            write(lu,302) (pres_kn(i,j),j=1,nrbknie),
     *                     pres_el(i),pres_num(i)
         else if (ncd.eq.3) then
            write(lu,303) (pres_kn(i,j),j=1,nrbknie),
     *                     pres_el(i),pres_num(i)
         endif

 300  continue
c     ****************************************************************


c     ****************************************************************
c     SYMMETRIE-RANDBEDINGUNGEN:          

      do 400 i=1,nsyme       
         if (ncd.eq.2) then
            write(lu,402) (syme_kn(i,j),j=1,nrbknie),
     *                     syme_el(i)
         else if (ncd.eq.3) then
            write(lu,403) (syme_kn(i,j),j=1,nrbknie),
     *                     syme_el(i)
         endif
 400  continue
c     ****************************************************************


c     ****************************************************************
c     CONV-RANDBEDINGUNGEN:          

      do 600 i=1,nconv
         if (ncd.eq.2) then
            write(lu,602) (conv_kn(i,j),j=1,nrbknie),
     *                     conv_el(i),REAL(conv_num(i))
         else if (ncd.eq.3) then
            write(lu,603) (conv_kn(i,j),j=1,nrbknie),
     *                     conv_el(i),REAL(conv_num(i))
         endif

 600  continue
c     ****************************************************************


      close(lu)
      comment='File geschrieben'
      call char_druck(comment,file_name,6)

c     *****************************************************************
c     FEHLERMELDUNG WENN EIN FILE FEHLERHAFT GEOEFFNET WURDE:

      goto 888

 777  continue

      comment='Fehler beim Oeffnen von File (test_print_rbe):'
      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine GEO_PRINT'
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)

 888  continue
c     *****************************************************************


c     ****************************************************************
c     FORMATE:
 
 101  format (I8,1x,I3,1x,F15.6)


 202  format (3(I8,1x),1x,I3)
 203  format (5(I8,1x),1x,I3)

 302  format (2(I7,1x),4x,i3,4x,f15.6,4x,I7)
 303  format (4(I7,1x),4x,i3,4x,i15,4x,I7)

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
 

