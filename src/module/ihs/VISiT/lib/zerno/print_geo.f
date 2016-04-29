C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE PRINT_GEO(lnods,lnods_num,lnods_mod,
     *                     coord,coord_num,coord_mod,
     *                     elpar,
     *                     kern_kn,kern_kn_adr,
     *                     lapp_kn,lapp_kn_adr,lapp_kn_proz,
     *                     dopp_kn,dopp_kn_adr,dopp_kn_proz,
     *                     kern_el,kern_el_adr,
     *                     lapp_el,lapp_el_adr,lapp_el_proz,
     *                     dopp_el,dopp_el_adr,dopp_el_proz,
     *                     nkern_max,nlapp_el,nlapp_kn,
     *                     ndopp_el,ndopp_kn,
     *                     zeig,folg,error_kno,error_ele,
     *                     geom_dim,ndat_max,geo_pfad,geo_name,
     *                     dopplapp)

      implicit none

      include 'common.zer'

      integer  lnods,lnods_num,lnods_mod,coord_num,coord_mod,
     *         kern_kn,kern_kn_adr,
     *         lapp_kn,lapp_kn_adr,lapp_kn_proz,
     *         dopp_kn,dopp_kn_adr,dopp_kn_proz,
     *         kern_el,kern_el_adr,
     *         lapp_el,lapp_el_adr,lapp_el_proz,
     *         dopp_el,dopp_el_adr,dopp_el_proz,
     *         nkern_max,nlapp_el,nlapp_kn,
     *         ndopp_el,ndopp_kn,elpar

      integer   nkern,nlapp

      integer  zeig,folg,error_kno,error_ele,
     *         geom_dim,ndat_max

      integer  i,k,ielem,el_num,proz_num,mod_num,nnn,
     *         igeb,kn_num,kn_joi,el_joi,help(8),luerr,
     *         nkn,nkn_kern,nkn_halo,
     *         nel,nel_kern,nel_halo

      integer  ipfad,lentb,lu,ip1,ip2,ip3,ip4

      real     coord,co(3)

      logical  fehler,dopplapp

      character*80 file_name,geo_pfad,geo_name,comment

      character*4  otto       

      parameter(lu=80)

      dimension lnods_num(nelem_max),lnods(nelem_max,nkd),
     *          lnods_mod(nelem_max)
      dimension coord_num(npoin_max),coord(npoin_max,ncd),
     *          coord_mod(npoin_max)
      dimension geom_dim(ngebiet,ndat_max)

      dimension zeig(npoin_max),folg(npoin_max)

      dimension error_kno(ngebiet),error_ele(ngebiet)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1),
     *          lapp_kn(nlapp_kn),lapp_kn_adr(ngebiet+1)

      dimension lapp_kn_proz(nlapp_kn),lapp_el_proz(nlapp_el)

      dimension dopp_el(ndopp_el),dopp_el_adr(ngebiet+1),
     *          dopp_kn(ndopp_kn),dopp_kn_adr(ngebiet+1)

      dimension dopp_kn_proz(ndopp_kn),dopp_el_proz(ndopp_el)
      dimension elpar(nelem_max)
c     **********************************************************


c     **********************************************************
c     BESCHRIFTEN DER FILENAMEN:

      file_name=geo_pfad
      ipfad=lentb(geo_pfad)
      ip1=ipfad+1
      ip2=ipfad+4
      ip3=ip2+1
      ip4=ip3+3
      file_name(ip1:ip2)='GEO_'
c     **********************************************************

c     **********************************************************
c     INITIALISIERUNGEN:

      do 350 igeb=1,ngebiet
         error_kno(igeb)=0
         error_ele(igeb)=0
 350  continue
c     **********************************************************


c     **********************************************************
c     GEBIETSWEISER AUSDRUCK DER GEOMETRIE:

      do 11 i=1,npoin_max
         zeig(i)=0
         folg(i)=0
 11   continue

      write(6,*)          

      do 100 igeb=1,ngebiet

         write(otto,'(i4.4)') igeb
         file_name(ip3:ip4)=otto(1:4)
         open(lu,file=file_name,status='unknown',err=777)

         comment='# Partition von:'    
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         call char_druck(comment,geo_name,lu)
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '
         write(lu,*)'#                                      '

c        Ausdruck der Dimensionen:
         nkn_kern=kern_kn_adr(igeb+1)-kern_kn_adr(igeb)
         nkn_halo=lapp_kn_adr(igeb+1)-lapp_kn_adr(igeb)

         nel_kern=kern_el_adr(igeb+1)-kern_el_adr(igeb)
         nel_halo=lapp_el_adr(igeb+1)-lapp_el_adr(igeb)
         
         if(dopplapp)then
            nkn_halo=nkn_halo
     *	            +dopp_kn_adr(igeb+1)-dopp_kn_adr(igeb)
            nel_halo=nel_halo
     *	            +dopp_el_adr(igeb+1)-dopp_el_adr(igeb)
         endif
	 
         nkn=nkn_kern+nkn_halo
         nel=nel_kern+nel_halo

         write(lu,911) nkn,nel,0,0,npoin_ges,nelem_ges,
     *                 knmax_num,elmax_num

         geom_dim(igeb,1)=nkn
         geom_dim(igeb,2)=nkn_kern
         geom_dim(igeb,3)=nkn_halo
         geom_dim(igeb,4)=nel
         geom_dim(igeb,5)=nel_kern
         geom_dim(igeb,6)=nel_halo

c        AUSDRUCK DER KERN-KNOTEN:
         nnn=0
         do 150 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
            if (ncd.eq.2) then
               co(1)=coord(kern_kn(i),1)
               co(2)=coord(kern_kn(i),2)
               co(3)=0.0
            else if (ncd.eq.3) then
               co(1)=coord(kern_kn(i),1)
               co(2)=coord(kern_kn(i),2)
               co(3)=coord(kern_kn(i),3)
            endif 
            kn_num=coord_num(kern_kn(i))
            kn_joi=kern_kn(i)
            proz_num=igeb 
            mod_num=coord_mod(kern_kn(i))

            if (zeig(kn_joi).eq.0) then
               nnn=nnn+1
               zeig(kn_joi)=nnn 
               folg(nnn)=kn_joi
               write(lu,901) kn_num,(co(k),k=1,3),mod_num,
     *                       proz_num,kn_joi
            else
               error_kno(igeb)=error_kno(igeb)+1
            endif
 150     continue

c        AUSDRUCK DER HALO-KNOTEN:
c         write(lu,*)'halo'
         do 160 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            if (ncd.eq.2) then
               co(1)=coord(lapp_kn(i),1)
               co(2)=coord(lapp_kn(i),2)
               co(3)=0.0
            else if (ncd.eq.3) then
               co(1)=coord(lapp_kn(i),1)
               co(2)=coord(lapp_kn(i),2)
               co(3)=coord(lapp_kn(i),3)
            endif 
            kn_num=coord_num(lapp_kn(i))
            kn_joi=lapp_kn(i)
            proz_num=lapp_kn_proz(i)
            mod_num=coord_mod(lapp_kn(i))

            if (zeig(kn_joi).eq.0) then
               nnn=nnn+1
               zeig(kn_joi)=nnn
               folg(nnn)=kn_joi
               write(lu,901) kn_num,(co(k),k=1,3),mod_num,
     *                                           proz_num,kn_joi
            else
               error_kno(igeb)=error_kno(igeb)+1
            endif
 160     continue

c        AUSDRUCK DER DOPPLAPP-KNOTEN:
         if(dopplapp)then
c         write(lu,*)'dopp'
            do 170 i=dopp_kn_adr(igeb),dopp_kn_adr(igeb+1)-1
               if (ncd.eq.2) then
                  co(1)=coord(dopp_kn(i),1)
                  co(2)=coord(dopp_kn(i),2)
                  co(3)=0.0
               else if (ncd.eq.3) then
                  co(1)=coord(dopp_kn(i),1)
                  co(2)=coord(dopp_kn(i),2)
                  co(3)=coord(dopp_kn(i),3)
               endif 
               kn_num=coord_num(dopp_kn(i))
               kn_joi=dopp_kn(i)
               proz_num=dopp_kn_proz(i)
               mod_num=coord_mod(dopp_kn(i))

               if (zeig(kn_joi).eq.0) then
                  nnn=nnn+1
                  zeig(kn_joi)=nnn
                  folg(nnn)=kn_joi
                  write(lu,901) kn_num,(co(k),k=1,3),mod_num,
     *                                       proz_num,kn_joi
               else
                  error_kno(igeb)=error_kno(igeb)+1
               endif
 170        continue
         endif

         if (nnn.ne.nkn) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine PRINT_GEO'
            write(luerr,*)'Die tatsaechlich geschriebe Knotenanzahl '
            write(luerr,*)'von Gebiet ',igeb,' stimmt mit der       '
            write(luerr,*)'berechneten Anzahl nicht ueberein.'
            write(luerr,*)'Tatsaechlich geschriebe Anzahl:',nnn
            write(luerr,*)'Berechnete Anzahl             :',nkn      
            call erro_ende(myid,parallel,luerr)
         endif

c        Initialisierung:
         do 151 i=1,nnn 
            zeig(folg(i))=0
 151     continue



c        AUSDRUCK DER KERN-ELEMENTE:
         nnn=0
         do 130 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            ielem=kern_el(i)
            do 131 k=1,nkd
               help(k)=coord_num(lnods(ielem,k))
 131        continue
            el_num=lnods_num(ielem)
            el_joi=ielem
            proz_num=igeb
            mod_num=lnods_mod(ielem)

            if (zeig(el_joi).eq.0) then
               nnn=nnn+1
               zeig(el_joi)=nnn 
               folg(nnn)=el_joi
               if (ncd.eq.2) then
                  write(lu,902) el_num,(help(k),k=1,nkd),mod_num,
     *                                                  proz_num,el_joi
               else if (ncd.eq.3) then
                  write(lu,903) el_num,(help(k),k=1,nkd),mod_num,
     *                                                  proz_num,el_joi
               endif 
            else
               error_ele(igeb)=error_ele(igeb)+1
            endif
 130     continue


c        AUSDRUCK DER HALO-ELEMENTE:
c         write(lu,*)'halo'
         do 140 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
            do 141 k=1,nkd
               help(k)=coord_num(lnods(ielem,k))
 141        continue
            el_num=lnods_num(ielem)
            el_joi=ielem
            proz_num=lapp_el_proz(i)
            mod_num=lnods_mod(ielem)

            if (zeig(el_joi).eq.0) then
               nnn=nnn+1
               zeig(el_joi)=nnn
               folg(nnn)=el_joi
               if (ncd.eq.2) then
                  write(lu,902) el_num,(help(k),k=1,nkd),mod_num,
     *                                                  proz_num,el_joi
               else if (ncd.eq.3) then
                  write(lu,903) el_num,(help(k),k=1,nkd),mod_num,
     *                                                  proz_num,el_joi
               endif 
            else
               error_ele(igeb)=error_ele(igeb)+1
            endif
 140     continue

c        AUSDRUCK DER DOPPLAPP-ELEMENTE:
         if(dopplapp)then
c         write(lu,*)'dopp'
            do 180 i=dopp_el_adr(igeb),dopp_el_adr(igeb+1)-1
               ielem=dopp_el(i)
               do 181 k=1,nkd
                  help(k)=coord_num(lnods(ielem,k))
 181           continue
               el_num=lnods_num(ielem)
               el_joi=ielem
               proz_num=dopp_el_proz(i)
               mod_num=lnods_mod(ielem)

               if (zeig(el_joi).eq.0) then
                  nnn=nnn+1
                  zeig(el_joi)=nnn
                  folg(nnn)=el_joi
                  if (ncd.eq.2) then
                     write(lu,902) el_num,(help(k),k=1,nkd),mod_num,
     *                                              proz_num,el_joi
                  else if (ncd.eq.3) then
                     write(lu,903) el_num,(help(k),k=1,nkd),mod_num,
     *                                              proz_num,el_joi
                  endif 
               else
                  error_ele(igeb)=error_ele(igeb)+1
               endif
 180        continue
         endif


         if (nnn.ne.nel) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine PRINT_GEO'
            write(luerr,*)'Die tatsaechlich geschriebe Elementanzahl '
            write(luerr,*)'von Gebiet ',igeb,' stimmt mit der       '
            write(luerr,*)'berechneten Anzahl nicht ueberein.'
            write(luerr,*)'Tatsaechlich geschriebe Anzahl:',nnn
            write(luerr,*)'Berechnete Anzahl             :',nel      
            call erro_ende(myid,parallel,luerr)
         endif

c        Initialisierung:
         do 161 i=1,nnn      
            zeig(folg(i))=0
 161     continue

         close(lu)
         comment='File geschrieben:'
         call char_druck(comment,file_name,6)

         nkern=kern_kn_adr(igeb+1)-kern_kn_adr(igeb)
         nlapp=lapp_kn_adr(igeb+1)-lapp_kn_adr(igeb)
c         CALL TEST_PRINT_GEO(cov_coord,cov_lnods,cov_lnods_joi,
c     *                       cov_lnods_num,cov_lnods_proz,
c     *                       cov_coord_num,cov_coord_joi,
c     *                       cov_coord_mod, cov_lnods_mod,
c     *                       cov_coord_proz,
c     *                       nkn,nel,nkern,nlapp,
c     *                       npoin_ges,nelem_ges,
c     *                       knmax_num,elmax_num,
c     *                       nkd,ncd,igeb,myid,parallel)                         
 100  continue
      write(6,*)
c     **********************************************************

c     **********************************************************
c     AUSDRUCK DE ELEMENTPARTITION:
c
c     file_name='ELEM.PAR'
c     open(lu,file=file_name,status='unknown',err=777)
c
c     comment='# Elementpartition von:'    
c     call char_druck(comment,geo_name,lu)
c     write(lu,*) nelem
c
c     do 730 igeb=1,ngebiet
c
c        do 740 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
c           ielem=kern_el(i)
c           write(lu,771) lnods_num(i),igeb
c740     continue
c771     format(2(i7,1x))
c
c730  continue
c
c     close(lu)
c     comment='File geschrieben:'
c     call char_druck(comment,file_name,6)
c     **********************************************************

c     **********************************************************
c     KONTROLLE DER FEHLER-FELDER:

      fehler=.false.
      nnn=0
      do 360 igeb=1,ngebiet
         if (error_kno(igeb).ne.0) then
            nnn=nnn+1
            fehler=.true.
         endif
 360  continue

      if (fehler) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PRINT_GEO'
         write(luerr,*)'Es gibt ',nnn,' Gebiete in denen doppelte '
         write(luerr,*)'Knoten enthalten sind.                    '
         write(luerr,*)'Gebiet   Anzahl doppelter Knoten'
         do 310 igeb=1,ngebiet
            if (error_kno(igeb).ne.0) then
               write(luerr,333) igeb,error_kno(igeb)
            endif
 310     continue
         call erro_ende(myid,parallel,luerr)
      endif


      fehler=.false.
      nnn=0
      do 370 igeb=1,ngebiet
         if (error_ele(igeb).ne.0) then
            nnn=nnn+1
            fehler=.true.
         endif
 370  continue

      if (fehler) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PRINT_GEO'
         write(luerr,*)'Es gibt ',nnn,' Gebiete in denen doppelte '
         write(luerr,*)'Elemente enthalten sind.                  '
         write(luerr,*)'Gebiet   Anzahl doppelter Elemente'
         do 320 igeb=1,ngebiet
            if (error_kno(igeb).ne.0) then
               write(luerr,333) igeb,error_ele(igeb)
            endif
 320     continue
         call erro_ende(myid,parallel,luerr)
      endif
 333  format(1x,i4,8x,i7)
c     **********************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (print_geo):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************


 911  format(10(i7,1x))
 902  format(i7,1x,4(i8,1x),2x,i3,1x,i3,1x,i7)
 903  format(i7,1x,8(i8,1x),2x,i3,1x,i3,1x,i7)
 901  format(i8,3x,3(f15.6,1x),2x,i3,1x,i3,1x,i7)

      return
      end


