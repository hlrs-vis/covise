C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE PRINT_ELPAR(lnods_num,
     *                      lapp_el,lapp_el_adr,lapp_el_proz,
     *                      nlapp_el,zeig,farb_geb,geo_name)

      implicit none

      include 'common.zer'

      integer lnods_num,
     *        lapp_el,lapp_el_adr,lapp_el_proz,
     *        nlapp_el,zeig,farb_geb

      integer  i,ielem,lu,igeb,luerr

      character*80 file_name,comment,geo_name

      parameter(lu=80)

      dimension lnods_num(nelem_max)

      dimension zeig(npoin_max),farb_geb(ngebiet)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1)

      dimension lapp_el_proz(nlapp_el)
c     **********************************************************

c     **********************************************************
c     AUSDRUCK DE ELEMENTPARTITION:


      write(6,*)'ngebiet=',ngebiet

      do 731 i=1,nelem_max
         zeig(i)=0
 731  continue

      do 730 igeb=1,ngebiet
        
c        write(6,776) ngebiet,igeb,lapp_el_adr(igeb),
c    *                                lapp_el_adr(igeb+1)-1
c776     format(4(i7,1x))

         do 740 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
c           zeig(ielem)=farb_geb(igeb)
            zeig(ielem)=igeb
 740     continue

 730  continue


      file_name='ELE.PAR'
      open(lu,file=file_name,status='unknown',err=777)

      comment='# Elementpartition von:'    
      call char_druck(comment,geo_name,lu)
      write(lu,*) nelem,nelem_ges

      do 750 i=1,nelem
          if (zeig(i).eq.0) then
             call erro_init(myid,parallel,luerr)
             write(luerr,*)'Fehler in Routine PRINT_ELPAR'
             write(luerr,*)'Das Zeiger-Feld ist Null     '
             write(luerr,*)'i         =',i          
             write(luerr,*)'zeig(i)   =',zeig(i)    
             call erro_ende(myid,parallel,luerr)
          endif
          write(lu,771) lnods_num(i),zeig(i)
 750  continue
 771  format(2(i7,1x))

      close(lu)
      comment='File geschrieben:'
      call char_druck(comment,file_name,6)
c     **********************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (print_elpar):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************


 911  format(10(i7,1x))
 902  format(i7,1x,4(i8,1x),2x,i3,1x,i7)
 903  format(i7,1x,8(i8,1x),2x,i3,1x,i7)
 901  format(i8,3x,3(f15.6,1x),2x,i3,1x,i7)

 801  format(i7,1x,6(e14.7,1x),i7)
 802  format(i7,1x,1(e14.7,1x),i7)

      return
      end

