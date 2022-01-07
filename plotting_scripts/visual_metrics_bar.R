#!/usr/bin/env Rscript
# This script produces a group bar plot of the different metrics


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
                               
#PSNR
psnr_plot <- ggplot(data[data$metric == "PSNR",], aes(x=experiment, y=value, fill=experiment)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    labs(x="Experiment", y="PSNR (dB)") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
          axis.text.x=element_blank(),
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

ggsave(paste(plot_filename, "_psnr.pdf", sep=""), width=5.7,height=5)


#LPIPS
lpips_plot <- ggplot(data[data$metric == "LPIPS",], aes(x=experiment, y=value, fill=experiment)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    labs(x="Experiment", y="LPIPS") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
          axis.text.x=element_blank(),
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

ggsave(paste(plot_filename, "_lpips.pdf", sep=""), width=5.7,height=5)


#SSIM
ssim_plot <- ggplot(data[data$metric == "SSIM",], aes(x=experiment, y=value, fill=experiment)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    labs(x="Experiment", y="SSIM") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
          axis.text.x=element_blank(),
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

ggsave(paste(plot_filename, "_ssim.pdf", sep=""), width=5.7,height=5)


legend <- get_legend(psnr_plot + theme(legend.position="top"))


prow <- plot_grid(psnr_plot + theme(legend.position="none"),
                  lpips_plot + theme(legend.position="none"),
                  ssim_plot + theme(legend.position="none"),
                  ncol = 3, align = "v", axis = "l")

# this tells it what order to put it in
# so basically tells it put legend first then plots with th legend height 20% of the
# plot
p <- plot_grid(legend, prow, rel_heights=c(.2,1), ncol =1)
ggsave(paste(plot_filename,"_metrics.pdf", sep=""), width=12.2, height=6)


