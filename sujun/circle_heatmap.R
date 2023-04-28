library(circlize)
library(dendextend)
library(openxlsx)
library(stringr)
library(devtools)
library(ComplexHeatmap)
library(grid)
library(RColorBrewer)
library(dendextend)
library(dendsort)



axonF<-read.table("/Users/krystal/Desktop/project/cloud_paper/2023/axon_feature.txt")
colnames(axonF)<-c('name','arbor_id','region','max_density','num_nodes','total_path_length','volume','branch','d_to_soma','d_to_soma2','hub','variance_ratio')

table<-read.table("/Users/krystal/Desktop/project/cloud_paper/2023/1891_type.txt",header=TRUE)
ssp_layer<-read.table("/Users/krystal/Desktop/project/cloud_paper/2023/SSp_layer.txt")
colnames(ssp_layer)<-c('name','L1','L2/3','L4','L5','L6a','L6b','Non-cortical')
ssp_arbor_layer<-read.table("/Users/krystal/Desktop/project/cloud_paper/2023/SSp_arbors_layer2.txt")
colnames(ssp_arbor_layer)<-c('name','L1','L2/3','L4','L5','L6a','L6b','Non-cortical')

LD_order<-read.table("/Users/krystal/Desktop/project/cloud_paper/2023/category.txt",header=TRUE,sep=",")

ssp_table<-matrix(0,nrow = nrow(ssp_arbor_layer),ncol=11)
count<-0
for(i in 1:nrow(ssp_arbor_layer)){
  if(sum(ssp_arbor_layer[i,2:7])>10){
    n<-nchar(ssp_arbor_layer$name[i])
    ssp_table[count,1]<-substr(ssp_arbor_layer$name[i],1,n-7)
    ssp_table[count,2]<-substr(ssp_arbor_layer$name[i],n,n)
    ssp_table[count,3]<-ssp_arbor_layer$L1[i]
    ssp_table[count,4]<-ssp_arbor_layer$`L2/3`[i]
    ssp_table[count,5]<-ssp_arbor_layer$L4[i]
    ssp_table[count,6]<-ssp_arbor_layer$L5[i]
    ssp_table[count,7]<-ssp_arbor_layer$L6a[i]
    ssp_table[count,8]<-ssp_arbor_layer$L6b[i]
    tmp<-LD_order[which((LD_order$name==ssp_table[count,1])&(LD_order$arbor_id)==as.numeric(ssp_table[count,2])),]
    ssp_table[count,9]<-tmp$LD[1]
    tmp2<-table[which(table$Name==ssp_table[count,1]),]
    ssp_table[count,10]<-tmp2$Soma_region[1]
    ssp_table[count,11]<-tmp2$Cortical_layer[1]
    count<-count+1
  }
}
ssp_table<-ssp_table[1:257,]

ssp_table2<-as.data.frame(ssp_table)
colnames(ssp_table2)<-c('name','arbor_id','L1','L2/3','L4','L5','L6a','L6b','LD','S','CL')
ssp_table2$L1<-as.numeric(ssp_table2$L1)
ssp_table2$`L2/3`<-as.numeric(ssp_table2$`L2/3`)
ssp_table2$L4<-as.numeric(ssp_table2$L4)
ssp_table2$L5<-as.numeric(ssp_table2$L5)
ssp_table2$L6a<-as.numeric(ssp_table2$L6a)
ssp_table2$L6b<-as.numeric(ssp_table2$L6b)
ssp_table2<-ssp_table2[which(!((ssp_table2$L1==0)&(ssp_table2$`L2/3`==0)&(ssp_table2$L4==0)&(ssp_table2$L5==0)&(ssp_table2$L6a==0)&(ssp_table2$L6b==0))),]

ssp_proj<-ssp_table2
keep<--1
for(i in 1:nrow(ssp_proj)){
  if(ssp_proj$L1[i]<1000){
    ssp_proj[i,3]<-0
  }
  if(ssp_proj$`L2/3`[i]<1000){
    ssp_proj[i,4]<-0
  }
  if(ssp_proj$L4[i]<1000){
    ssp_proj[i,5]<-0
  }
  if(ssp_proj$L5[i]<1000){
    ssp_proj[i,6]<-0
  }
  if(ssp_proj$L6a[i]<1000){
    ssp_proj[i,7]<-0
  }
  if(ssp_proj$L6b[i]<1000){
    ssp_proj[i,8]<-0
  }
  s=sum(ssp_proj[i,3:8])
  if(s>0){
    keep<-c(keep,i)
  }
}
keep<-keep[-1]
ssp_proj<-ssp_proj[keep,]

### based on stype
ssp_stype_proj<-matrix(0,nrow = 6,ncol = 12)
N<-1
for (i in ssp) {
  lines<-ssp_proj[which(ssp_proj$S==i),]
  local<-lines[which(lines$LD==0),]
  distal<-lines[which(lines$LD==1),]
  if(nrow(local)==0){
    m1<-c(0,0,0,0,0,0)
  }
  else{
    m1<-colMeans(local[3:8])
  }
  if(nrow(distal)==0){
    m2<-c(0,0,0,0,0,0)
  }
  else{
    m2<-colMeans(distal[3:8])
  }
  ssp_stype_proj[N,]<-c(m1,m2)
  N<-N+1
}
### standardize
ssp_stype_proj2<-ssp_stype_proj
for(i in 1:6){
  ssp_stype_proj2[i,]<-(ssp_stype_proj[i,]-min(ssp_stype_proj[i,]))/(max(ssp_stype_proj[i,])-min(ssp_stype_proj[i,]))
}


## based on layers
ssp_layer_proj<-matrix(0,nrow = 6,ncol = 12)
N<-1
for (i in L) {
  lines<-ssp_proj[which(ssp_proj$CL==i),]
  local<-lines[which(lines$LD==0),]
  distal<-lines[which(lines$LD==1),]
  if(nrow(local)==0){
    m1<-c(0,0,0,0,0,0)
  }
  else{
    m1<-colMeans(local[3:8])
  }
  if(nrow(distal)==0){
    m2<-c(0,0,0,0,0,0)
  }
  else{
    m2<-colMeans(distal[3:8])
  }
  ssp_layer_proj[N,]<-c(m1,m2)
  N<-N+1
}
ssp_layer_proj<-ssp_layer_proj[-c(1,6),]
ssp_layer_proj2<-ssp_layer_proj
for(i in 1:4){
  ssp_layer_proj2[i,]<-(ssp_layer_proj[i,]-min(ssp_layer_proj[i,]))/(max(ssp_layer_proj[i,])-min(ssp_layer_proj[i,]))
}
write.table(ssp_stype_proj2,file = "/Users/krystal/Desktop/project/cloud_paper/2023/ssp_stype_proj.txt",col.names = FALSE,row.names = FALSE)
write.table(ssp_layer_proj2,file = "/Users/krystal/Desktop/project/cloud_paper/2023/ssp_layer_proj.txt",col.names = FALSE,row.names = FALSE)

## 统计subtype
ssp<-c("SSp-bfd","SSp-ll","SSp-m","SSp-n","SSp-ul","SSp-un")
L<-c("1","2/3","4","5","6a","6b")
ssp_subtype_table<-matrix(0,nrow = length(ssp)*length(L),ncol=12)
N<-1
del_lines<-list()
for(i in ssp){
  for (j in L) {
    lines<-ssp_table2[which((ssp_table2$S==i)&(ssp_table2$CL==j)),]
    if(nrow(lines)<1){
      for(k in 1:12){
        ssp_subtype_table[N,k]<-0
      }
      del_lines<-c(del_lines,N)
      N<-N+1
      next
    }
    local<-lines[lines$LD==0,]
    distal<-lines[lines$LD==1,]
    if(nrow(local)==0){
      m1<-c(0,0,0,0,0,0)
    }
    else{
      m1<-colMeans(local[3:8])
    }
    if(nrow(distal)==0){
      m2<-c(0,0,0,0,0,0)
    }
    else{
      m2<-colMeans(distal[3:8])
    }
    ssp_subtype_table[N,]<-c(m1,m2)
    s1<-sum(m1)
    s2<-sum(m2)
    if((s1+s2)==0){
      del_lines<-c(del_lines,N)
    }
    N<-N+1
  }
}

rev_del<-rev(del_lines)
ssp_subtype_table2<-ssp_subtype_table
for(i in rev_del){
  ssp_subtype_table2<-ssp_subtype_table2[-i,]
}

## standardize
ssp_subtype_table3<-ssp_subtype_table2
for(i in 1:18){
  ssp_subtype_table3[i,]<-(ssp_subtype_table3[i,]-min(ssp_subtype_table3[i,]))/(max(ssp_subtype_table3[i,])-min(ssp_subtype_table3[i,]))
}

ssp_subtype_table3<-ssp_subtype_table3[-c(17),]
#ssp_subtype_table3<-ssp_subtype_table3[,-6]
colnames(ssp_subtype_table3)<-c("L1","L2/3","L4","L5","L6a","L6b","L1","L2/3","L4","L5","L6a","L6b")
rownames(ssp_subtype_table3)<-c("SSp-bfd-L2/3","SSp-bfd-L4","SSp-bfd-L5","SSp-bfd-L6a",
                                "SSp-ll-L2/3","SSp-ll-L5","SSp-ll-L6a",
                                "SSp-m-L2/3","SSp-m-L4","SSp-m-L5",
                                "SSp-n-L2/3","SSp-n-L4","SSp-n-L5",
                                "SSp-ul-L2/3","SSp-ul-L4","SSp-ul-L5",
                                "SSp-un-L5")

ssp_subtype_R<-c("SSp-ul","SSp-bfd","SSp-un","SSp-m",
                 "SSp-bfd","SSp-n","SSp-ul",
                 "SSp-ll","SSp-ul","SSp-m","SSp-ll",
                 "SSp-n","SSp-bfd","SSp-n","SSp-ll",
                 "SSp-m","SSp-bfd")
ssp_subtype_R<-data.frame(ssp_subtype_R)
rownames(ssp_subtype_R)<-c("SSp-ul\n-L4","SSp-bfd\n-L2/3","SSp-un\n-L5","SSp-m\n-L5",
                           "SSp-bfd\n-L5","SSp-n\n-L5","SSp-ul\n-L5",
                           "SSp-ll\n-L5","SSp-ul\n-L2/3",
                           "SSp-m\n-L2/3","SSp-ll\n-L6a","SSp-n\n-L2/3",
                           "SSp-bfd\n-L6a","SSp-n\n-L4","SSp-ll\n-L2/3","SSp-m\n-L4","SSp-bfd\n-L4")



## figure
circos.clear() 
circos.par(gap.after=c(50))
col_r=structure(brewer.pal(6,"Set1"),names=ssp)
circos.heatmap(ssp_subtype_R,col = col_r,track.height=0.03,
               rownames.side="outside",rownames.col="black",rownames.cex=0.8,rownames.font=2,
               )
col_p=colorRamp2(c(-0.1,0.5,1.1),c("lightseagreen","white", "violet"))
circos.heatmap(ssp_subtype_table3,col = col_p,track.height = 0.65,
               dend.side="inside", cluster = TRUE,dend.track.height=0.2,
               rownames.side="outside",rownames.col="black",rownames.cex=0.4,rownames.font=2,
               dend.callback=function(dend,m,si){color_branches(dend,k=8,col=1:8)})

circos.track(track.index = get.current.track.index(),
             panel.fun = function(x, y) {
               if(CELL_META$sector.numeric.index == 1) { # the last sector
                 cn = colnames(ssp_subtype_table3)
                 n = length(cn)
                 circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
                             1:n /1.77+2, cn, 
                             cex = 0.8, adj = c(0, 0.5), facing = "inside")
               }
             }, bg.border = NA)

lg=Legend(title="Projection Strength",col_fun=col_p,direction = c("horizontal"),
          grid_width = unit(4, "cm"),legend_height = unit(2, "cm"))
circle_size= unit(0.07,"snpc")
grid.draw(lg)

library("lessR")
subtype_names<-c("","SSp-ul-L4","SSp-bfd-L2/3","SSp-un-L5","SSp-m-L5",
  "SSp-bfd-L5","SSp-n-L5","SSp-ul-L5",
  "SSp-ll-L5","SSp-ul-L2/3",
  "SSp-m-L2/3","SSp-ll-L6a","SSp-n-L2/3",
  "SSp-bfd-L6a","SSp-n-L4","SSp-ll-L2/3","SSp-m-L4","SSp-bfd-L4")
#subtype_names_df<-data.frame(variable=tmp)
#PieChart(variable,data = subtype_names_df,values_size = 1,labels_cex = 1.2)
PieChart(subtype_names,c(50,310/17,310/17,310/17,310/17,310/17,310/17,310/17,310/17,310/17,310/17,310/17,310/17,
                         310/17,310/17,310/17,310/17,310/17),main=NULL,labels_cex = 1.2)
