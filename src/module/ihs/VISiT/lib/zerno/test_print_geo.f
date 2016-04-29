C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE TEST_PRINT_GEO(coord,lnods,lnods_joi,
     *                          lnods_num,lnods_proz,
     *                          coord_num,coord_joi,
     *                          coord_mod,lnods_mod,
     *                          coord_proz,
     *                          npoin,nelem,nelem_kern,nelem_lapp,
     *                          npoin_ges,nelem_ges,
     *                          knmax_num,elmax_num,
     *                          nkd,ncd,
     *                          igeb,myid,parallel)

      implicit none

      integer lnods,lnods_joi,
     *        lnods_num,lnods_proz,
     *        coord_num,coord_joi,
     *        coord_mod,lnods_mod,
     *        coord_proz

      integer  npoin,nelem,nelem_kern,nelem_lapp,
     *         npoin_ges,nelem_ges,
     *         knmax_num,elmax_num,
     *         nkd,ncd,igeb,myid

      real    coord,vek(3)

      integer i,k,lu,luerr

      logical  parallel

      character*80 file_name,comment

      character*4  otto       

      parameter(lu=80)

      dimension lnods(nelem,nkd),coord(npoin,ncd),
     *          coord_num(npoin),lnods_num(nelem),
     *          coord_joi(npoin),lnods_joi(nelem),lnods_proz(nelem),
     *          coord_mod(npoin),lnods_mod(nelem),
     *          coord_proz(npoin)
c     **********************************************************


c     **********************************************************
c     BESCHRIFTEN DER FILENAMEN:

      file_name='zer/GEO_0000'
      write(otto,'(i4.4)') igeb
      file_name(9:12)=otto(1:4)
      open(lu,file=file_name,err=777)
c     **********************************************************


c     **********************************************************
c     GEBIETSWEISER AUSDRUCK DER GEOMETRIE:

         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '

c        Ausdruck der Dimensionen:
c old         write(lu,911) npoin,nelem,nelem_kern,nelem_lapp,
c old     *                 npoin_ges,nelem_ges,
c old     *                 knmax_num,elmax_num
         write(lu,911) npoin,nelem,0,0,
     *                 npoin_ges,nelem_ges,
     *                 knmax_num,elmax_num


         vek(1)=0.0
         vek(2)=0.0
         vek(3)=0.0
         do 100 i=1,npoin
            do 110 k=1,ncd
               vek(k)=coord(i,k)
 110        continue
            write(lu,901) coord_num(i),(vek(k),k=1,3),coord_mod(i),
     *                   coord_proz(i), coord_joi(i)
 100     continue

         do 200 i=1,nelem
            if (ncd.eq.2) then
               write(lu,902) lnods_num(i),
     *                       (lnods(i,k),k=1,nkd),
     *                       lnods_proz(i),lnods_joi(i)
            else if (ncd.eq.3) then
               write(lu,903) lnods_num(i),
     *                       (lnods(i,k),k=1,nkd),
     *                     lnods_mod(i),lnods_proz(i),lnods_joi(i)
            endif 
 200     continue

         close(lu)
         comment='File geschrieben:'
         call char_druck(comment,file_name,6)
c     **********************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (test_print_geo):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************

 911  format(10(i7,1x))
 902  format(i7,1x,4(i8,1x),2x,i3,1x,i3)
 903  format(i7,1x,8(i8,1x),2x,i3,1x,i3,1x,i7)
 901  format(i8,3x,3(f15.6,1x),2x,i3,1x,i3,1x,i7)

      return
      end

