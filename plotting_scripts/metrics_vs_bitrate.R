#!/usr/bin/env Rscript
# This script can be run using:
# ./overall_tpt_comparison.R <data> <pdf prefix>

library(ggplot2)
library(cowplot, warn.conflicts = FALSE)
library(scales)
library(sysfonts)
library(showtext)
library(showtextdb)
showtext_auto()

args <- commandArgs(trailingOnly=TRUE)
file <- args[1]
plot_filename <- args[2]
data<-read.csv(file)


psnr_plot <- ggplot(data[data$bitrate != 900, ], aes(x=bitrate,y=psnr,linetype="Codec H.264",color="Codec H.264")) + 
        geom_line(size=0.8) +
        geom_point(size=2) +

        geom_hline(aes(yintercept=29.241, linetype="Generic Model", color="Generic Model")) +
        geom_hline(aes(yintercept=30.115, linetype="Personalized Model", color="Personalized Model")) +

        labs(x="Bitrate (Kb/s)", y="PSNR (dB)", color="approaches", linetype="approaches") +
        
        theme_minimal(base_size=15) +
        theme(axis.text.x=element_text(size=rel(1.3)), axis.text.y=element_text(size=rel(1.3))) +
        theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="none",
              legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank(),
              legend.margin=margin(c(0,0,0,0)))

ggsave(paste(plot_filename, "_psnr.pdf", sep=""), width=5.7,height=5)


ssim_plot <- ggplot(data, aes(x=bitrate,y=ssim,linetype="Codec H.264",color="Codec H.264")) + 
        geom_line(size=0.8) +
        geom_point(size=2) +
        
        geom_hline(aes(yintercept=0.918, linetype="Generic Model", color="Generic Model")) +
        geom_hline(aes(yintercept=0.923, linetype="Personalized Model", color="Personalized Model")) +

        labs(x="Bitrate (Kb/s)", y="SSIM", color="approaches", linetype="approaches") +
        
        theme_minimal(base_size=15) +
        theme(axis.text.x=element_text(size=rel(1.3)), axis.text.y=element_text(size=rel(1.3))) +
        theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="none",
              legend.box.margin=margin(-10,-10,-10,-10),
              legend.margin=margin(c(0,0,0,0)))

ggsave(paste(plot_filename, "_ssim.pdf", sep=""), width=5.7,height=5)

lpips_plot <- ggplot(data, aes(x=bitrate,y=lpips,linetype="Codec H.264",color="Codec H.264")) + 
        geom_line(size=0.8) +
        geom_point(size=2) +

        geom_hline(aes(yintercept=0.103, linetype="Generic Model", color="Generic Model")) +
        geom_hline(aes(yintercept=0.058, linetype="Personalized Model", color="Personalized Model")) +
        
        labs(x="Bitrate (Kb/s)", y="LPIPS", color="approaches", linetype="approaches") +
        
        theme_minimal(base_size=15) +
        theme(axis.text.x=element_text(size=rel(1.3)), axis.text.y=element_text(size=rel(1.3))) +
        theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="none",
              legend.box.margin=margin(-10,-10,-10,-10),
              legend.margin=margin(c(0,0,0,0)))

ggsave(paste(plot_filename, "_lpips.pdf", sep=""), width=5.7,height=5)

legend <- get_legend(psnr_plot + theme(legend.position="top"))


prow <- plot_grid(psnr_plot + theme(legend.position="none"),
                  ssim_plot + theme(legend.position="none"),
                  lpips_plot + theme(legend.position="none"),
                  ncol = 3, align = "v", axis = "l")

# this tells it what order to put it in
# so basically tells it put legend first then plots with th legend height 20% of the
# plot
p <- plot_grid(legend, prow, rel_heights=c(.2,1), ncol =1)

ggsave(paste(plot_filename,"_metrics.pdf", sep=""), width=12.2, height=5)
