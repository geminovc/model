#!/usr/bin/env Rscript
# This script produces a bar plot of the different metrics
# Usage: ./metrics_bar_plot.R metrics.csv original_bilayer_knobs
# result: original_bilayer_knobs_metrics.pdf with all three plots

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
data$experiment_name <- factor(data$experiment_name, 
                               levels=c("augment_0.5", "augment_0.3", "augment_0.2", "augment_0.1"), 
                               labels=c("0.5", "0.3", "0.2", "0.1"))

psnr_plot <- ggplot(data, aes(x=experiment_name,y=G_PSNR_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 
        geom_errorbar(aes(ymin=G_PSNR_min, ymax=G_PSNR_max), width=0.2) +
        labs(x="Augmentation", y="PSNR (dB)") + 

        theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="none",
              legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank(),
              legend.margin=margin(c(0,0,0,0)))
ggsave(paste(plot_filename, "_psnr.pdf", sep=""), width=5.7,height=5)

lpips_plot <- ggplot(data, aes(x=experiment_name,y=G_LPIPS_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 
        geom_errorbar(aes(ymin=G_LPIPS_min, ymax=G_LPIPS_max), width=0.2) +

        labs(x="Augmentation", y="LPIPS") 
ggsave(paste(plot_filename, "_lpips.pdf", sep=""), width=5.7,height=5)

pme_plot <- ggplot(data, aes(x=experiment_name,y=G_PME_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 
        geom_errorbar(aes(ymin=G_PME_min, ymax=G_PME_max), width=0.2) +

        labs(x="Augmentation", y="PME") 
ggsave(paste(plot_filename, "_pme.pdf", sep=""), width=5.7,height=5)

csim_plot <- ggplot(data, aes(x=experiment_name,y=G_CSIM_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 
        geom_errorbar(aes(ymin=G_CSIM_min, ymax=G_CSIM_max), width=0.2) +

        labs(x="Augmentation", y="CSIM") 
ggsave(paste(plot_filename, "_csim.pdf", sep=""), width=5.7,height=5)

ssim_plot <- ggplot(data, aes(x=experiment_name,y=G_SSIM_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 
        geom_errorbar(aes(ymin=G_SSIM_min, ymax=G_SSIM_max), width=0.2) +

        labs(x="Augmentation", y="SSIM") 
ggsave(paste(plot_filename, "_ssim.pdf", sep=""), width=5.7,height=5)

legend <- get_legend(psnr_plot + theme(legend.position="top"))


prow <- plot_grid(psnr_plot + theme(legend.position="none"),
                  lpips_plot + theme(legend.position="none"),
                  pme_plot + theme(legend.position="none"),
                  ncol = 3, align = "v", axis = "l")
second_row <- plot_grid(csim_plot + theme(legend.position="none"),
                  ssim_plot + theme(legend.position="none"),
                  ncol = 3, align = "v", axis = "l")

# this tells it what order to put it in
# so basically tells it put legend first then plots with th legend height 20% of the
# plot
p <- plot_grid(legend, prow, second_row, rel_heights=c(.2,1, 1), ncol =1)
ggsave(paste(plot_filename,"_metrics.pdf", sep=""), width=12.2, height=6)
