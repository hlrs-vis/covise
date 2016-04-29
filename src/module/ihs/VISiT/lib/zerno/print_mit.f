C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE PRINT_MIT(lnods_num,coord_num,
     *                     kern_kn,kern_kn_adr,
     *                     lapp_kn,lapp_kn_adr,
     *                     dopp_kn,dopp_kn_adr,
     *                     kern_el,kern_el_adr,
     *                     lapp_el,lapp_el_adr,
     *                     dopp_el,dopp_el_adr,
     *                     nkern_max,nlapp_el,nlapp_kn,
     *                     ndopp_el,ndopp_kn,
     *                     erg_kn,nsp_kn_erg,
     *                     erg_el,nsp_el_erg,
     *                     erg_za,nsp_za_erg,
     *                     transi_erg,mit_pfad,mit_name,
     *                     dopplapp,geo_name,geo_pfad,
     *                     rbe_name,rbe_pfad)

      implicit none

      include 'common.zer'

      integer  lnods_num,coord_num,
     *         nsp_kn_erg,nsp_el_erg,nsp_za_erg

      integer  kern_kn,kern_kn_adr,
     *         lapp_kn,lapp_kn_adr,
     *         dopp_kn,dopp_kn_adr,
     *         kern_el,kern_el_adr,
     *         lapp_el,lapp_el_adr,
     *         dopp_el,dopp_el_adr,
     *         nkern_max,nlapp_el,nlapp_kn,
     *         ndopp_el,ndopp_kn

      integer i,k,ielem,
     *        nel,nel_kern,nel_halo,
     *        nkn,nkn_kern,nkn_halo,
     *        nza,el_num,
     *        igeb,kn_num,kn_joi,el_joi,luerr,inode,
     *        ipfad,lentb,lu,
     *        ip1,ip2,ip3,ip4

      real     erg_kn,erg_el,erg_za

      logical transi_erg,dopplapp

      character*80 file_name,mit_name,mit_pfad,comment,
     *             form_node,form_elem
      character*80 geo_name,comment2,comment3,geo_pfad,
     *             comment4, comment5,rbe_name,rbe_pfad,
     *             comment6

      character*4  otto       
      character*2  hugo       

      parameter(lu=80)

      dimension coord_num(npoin_max),lnods_num(nelem_max),
     *          erg_kn(npoin_max,nsp_kn_erg),
     *          erg_el(npoin_max,nsp_el_erg),
     *          erg_za(npoin_max,nsp_za_erg)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1),
     *          lapp_kn(nlapp_kn),lapp_kn_adr(ngebiet+1)

      dimension dopp_el(ndopp_el),dopp_el_adr(ngebiet+1),
     *          dopp_kn(ndopp_kn),dopp_kn_adr(ngebiet+1)
c     **********************************************************

c      write(*,*)'in print_mit'

c     **********************************************************
c     BESCHRIFTEN DER FILENAMEN:

      file_name=mit_pfad
      ipfad=lentb(file_name)
      ip1=ipfad+1
      ip2=ipfad+4
      ip3=ip2+1
      ip4=ip3+3
      file_name(ip1:ip2)='MIT_'
c     **********************************************************

c     **********************************************************
c     GEBIETSWEISER AUSDRUCK DER ANFANGSNAEHERUNG: 


      do 100 igeb=1,ngebiet

         write(otto,'(i4.4)') igeb
         file_name(ip3:ip4)=otto(1:4)
         open(lu,file=file_name,status='unknown',err=777)
     
         if (rbe_name.eq.' ') then
            rbe_name = 'not specified' 
         endif
         if (rbe_pfad.eq.' ') then
            rbe_pfad = 'not specified' 
         endif

         comment='#Partition of:'    
         comment2='#Results of:'    
         comment3='#GEO decomposed in:'    
         comment4='#with   RBE:'    
         comment5='#RBE decomposed in:'    
         comment6='#RBE used in ERG:'    
c         write(lu,*)'#                                      '
c         write(lu,*)'#                                      '
c         write(lu,*)'#                                      '
c         write(lu,'(A)')'#Result made with ZERNO'
c         write(lu,997)'#version:',fen_version(1:ver_len)
         call char_druck(comment,mit_name,lu)
         call char_druck(comment2,geo_name,lu)
         call char_druck(comment3,geo_pfad,lu)
         call char_druck(comment4,rbe_name,lu)
         call char_druck(comment5,rbe_pfad,lu)
         call char_druck(comment6,rbe_used,lu)
         write (lu,998)'#Dimension: ', ncd
c         write(lu,999)'#Nr.Gebiete:',ngebiet
         write(lu,999)'#No. of parts:',ngebiet
         write(lu,1000)'#Averaging:',text_avg
         write(lu,*)'#                                      '
c         write(lu,*)'#                                      '
c         write(lu,*)'#                                      '
c         write(lu,*)'#                                      '
c         write(lu,*)'#                                      '

c        Ausdruck der Dimensionen:
         nkn_kern=kern_kn_adr(igeb+1)-kern_kn_adr(igeb)
         nkn_halo=lapp_kn_adr(igeb+1)-lapp_kn_adr(igeb)

         nel_kern=kern_el_adr(igeb+1)-kern_el_adr(igeb)
         nel_halo=lapp_el_adr(igeb+1)-lapp_el_adr(igeb)

         if (dopplapp) then
            nkn_halo=nkn_halo
     *        +dopp_kn_adr(igeb+1)-dopp_kn_adr(igeb)
            nel_halo=nel_halo
     *        +dopp_el_adr(igeb+1)-dopp_el_adr(igeb) 
	 endif

         nkn=nkn_kern+nkn_halo
         nel=nel_kern+nel_halo

         if (transi_erg) then
            nza=nkn
         else
            nza=0  
         endif

         write(lu,911) nkn,nel,nza,nsp_kn_erg,nsp_el_erg,nsp_za_erg

         write(hugo,'(i2.2)') nsp_kn_erg
         form_node='(i7,1x,  (e14.7,1x),i7)'
         form_node(8:9)=hugo(1:2)

         write(hugo,'(i2.2)') nsp_el_erg
         form_elem='(i7,1x,  (e14.7,1x),i7)'
         form_elem(8:9)=hugo(1:2)

c        AUSDRUCK DER KERN-KNOTEN:
         do 150 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
            inode=kern_kn(i)
            kn_num=coord_num(inode)
            kn_joi=inode

            write(lu,form_node) kn_num,(erg_kn(inode,k),k=1,nsp_kn_erg),
     *                          kn_joi
 150     continue

c        AUSDRUCK DER HALO-KNOTEN:
         do 151 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            inode=lapp_kn(i)
            kn_num=coord_num(inode)
            kn_joi=inode

            write(lu,form_node) kn_num,(erg_kn(inode,k),k=1,nsp_kn_erg),
     *                          kn_joi
 151     continue

c        AUSDRUCK DER DOPPLAPP-KNOTEN:
         if (dopplapp) then
            do 152 i=dopp_kn_adr(igeb),dopp_kn_adr(igeb+1)-1
               inode=dopp_kn(i)
               kn_num=coord_num(inode)
               kn_joi=inode
               write(lu,form_node) kn_num,(erg_kn(inode,k),
     *                     k=1,nsp_kn_erg),kn_joi
 152        continue
         endif



c        AUSDRUCK DER KERN-ELEMENET:
         do 130 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            ielem=kern_el(i)
            el_num=lnods_num(ielem)
            el_joi=ielem
            write(lu,form_elem) el_num,(erg_el(ielem,k),k=1,nsp_el_erg),
     *                          el_joi
 130     continue

c        AUSDRUCK DER HALO-ELEMENTE:
         do 131 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
            el_num=lnods_num(ielem)
            el_joi=ielem
            write(lu,form_elem) el_num,(erg_el(ielem,k),k=1,nsp_el_erg),
     *                          el_joi
 131     continue

c        AUSDRUCK DER DOPPLAPP-ELEMENTE:
         if (dopplapp) then
            do 132 i=dopp_el_adr(igeb),dopp_el_adr(igeb+1)-1
               ielem=dopp_el(i)
               el_num=lnods_num(ielem)
               el_joi=ielem
               write(lu,form_elem) el_num,
     *          (erg_el(ielem,k),k=1,nsp_el_erg),el_joi
 132        continue
         endif


c        Ausdruck der instationaeren Altwerte:
         if (transi_erg) then

c           AUSDRUCK DER KERN-KNOTEN:
            do 170 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
               inode=kern_kn(i)
               kn_num=coord_num(inode)
               kn_joi=inode
               write(lu,803) (erg_za(inode,k),k=1,nsp_za_erg)
 170        continue

c           AUSDRUCK DER HALO-KNOTEN:
            do 171 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
               inode=lapp_kn(i)
               kn_num=coord_num(inode)
               kn_joi=inode
               write(lu,803) (erg_za(inode,k),k=1,nsp_za_erg)
 171        continue

c           AUSDRUCK DER DOPPLAPP-KNOTEN:
            if (dopplapp) then
               do 172 i=dopp_kn_adr(igeb),dopp_kn_adr(igeb+1)-1
                  inode=dopp_kn(i)
                  kn_num=coord_num(inode)
                  kn_joi=inode
                  write(lu,803) (erg_za(inode,k),k=1,nsp_za_erg)
 172           continue
            endif
         endif


         close(lu)
         comment='File geschrieben:'
         call char_druck(comment,file_name,6)
 100  continue
      write(6,*)
c     **********************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File:'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************

 911  format(10(i7,1x))
 801  format(i7,1x,5(e14.7,1x),i7)
 802  format(i7,1x,1(e14.7,1x),i7)
 803  format(10(e14.7,1x))

 997  format(a24,a5)
 998  format(a13,i1)
c ihs 998  format(a12,i1)
 999  format(a14,i6)
1000  format(a11,a6)


      return
      end

