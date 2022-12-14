#!/usr/bin/env Rscript
# This script can be run using:
# ./overall_tpt_comparison.R <data> <pdf prefix>

library(ggplot2)
library(cowplot, warn.conflicts = FALSE)
library(scales)
library(sysfonts)
library(showtext)
library(showtextdb)
source("style.R")
showtext_auto()

args <- commandArgs(trailingOnly=TRUE)
file <- args[1]
plot_filename <- args[2]
data<-read.csv(file)


psnr_plot <- ggplot(data[data$bitrate != 900, ], aes(x=bitrate,y=psnr,color="Codec H.264")) + 
        geom_line(size=0.8, linetype="dashed") +
        geom_point(size=2) +

        geom_point(size=2, x=50, y=29.241, aes(color="Generic Model")) +
        annotate("text", x=95, y=28.241, label="Generic\nModel",  color="#00ba38") +
        
        geom_point(size=2, x=50, y=30.115, aes(color="Personalized Model")) +
        annotate("text", x=150, y=31.115, label="Personalized\nModel",  color="#619cff") +

        labs(x="Bitrate (Kb/s)", y="PSNR (dB)", color="approaches", linetype="approaches")
ggsave(paste(plot_filename, "_psnr.pdf", sep=""), width=5.7,height=5)


ssim_plot <- ggplot(data, aes(x=bitrate,y=ssim,color="Codec H.264")) + 
        geom_line(size=0.8,  linetype="dashed") +
        geom_point(size=2) +
        
        geom_point(size=2, x=50, y=0.918, aes(color="Generic Model")) +
        annotate("text", x=95, y=0.903, label="Generic\nModel",  color="#00ba38") +
        
        geom_point(size=2, x=50, y=0.923, aes(color="Personalized Model")) +
        annotate("text", x=150, y=0.933, label="Personalized\nModel",  color="#619cff") +
        

        labs(x="Bitrate (Kb/s)", y="SSIM", color="approaches", linetype="approaches") 
ggsave(paste(plot_filename, "_ssim.pdf", sep=""), width=5.7,height=5)

lpips_plot <- ggplot(data, aes(x=bitrate,y=lpips,color="Codec H.264")) + 
        geom_line(size=0.8, linetype="dashed") +
        geom_point(size=2) +

        geom_point(size=2, x=50, y=0.103, aes(color="Generic Model")) +
        annotate("text", x=95, y=0.123, label="Generic\nModel",  color="#00ba38") +
        
        geom_point(size=2, x=50, y=0.058, aes(color="Personalized Model")) +
        annotate("text", x=150, y=0.038, label="Personalized\nModel",  color="#619cff") +
        
        labs(x="Bitrate (Kb/s)", y="LPIPS", color="approaches", linetype="approaches") 
ggsave(paste(plot_filename, "_lpips.pdf", sep=""), width=5.7,height=5)

legend <- get_legend(psnr_plot + theme(legend.position="top"))


prow <- plot_grid(psnr_plot + theme(legend.position="none"),
                  ssim_plot + theme(legend.position="none"),
                  lpips_plot + theme(legend.position="none"),
                  ncol = 3, align = "v", axis = "l")

# this tells it what order to put it in
# so basically tells it put legend first then plots with th legend height 20% of the
# plot
#p <- plot_grid(legend, prow, rel_heights=c(.2,1), ncol =1)

ggsave(paste(plot_filename,"_metrics.pdf", sep=""), width=12.2, height=3.5)
