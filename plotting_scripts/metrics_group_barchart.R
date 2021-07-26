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
psnr_plot <- ggplot(data, aes(x=pose_name, y = G_PSNR_mean, fill=experiment_name)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    geom_errorbar(aes(ymin=G_PSNR_min, ymax=G_PSNR_max), stat="identity", position=position_dodge(.5) , width=0.2) +
    labs(x="pose_distribution", y="PSNR (dB)") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

ggsave(paste(plot_filename, "_psnr.pdf", sep=""), width=5.7,height=5)


#LPIPS
lpips_plot <- ggplot(data, aes(x=pose_name, y = G_LPIPS_mean, fill=experiment_name)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    geom_errorbar(aes(ymin=G_LPIPS_min, ymax=G_LPIPS_max), stat="identity", position=position_dodge(.5) , width=0.2) +
    labs(x="pose_distribution", y="LPIPS") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

ggsave(paste(plot_filename, "_lpips.pdf", sep=""), width=5.7,height=5)


#PME
pme_plot <- ggplot(data, aes(x=pose_name, y = G_PME_mean, fill=experiment_name)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    geom_errorbar(aes(ymin=G_PME_min, ymax=G_PME_max), stat="identity", position=position_dodge(.5) , width=0.2) +
    labs(x="pose_distribution", y="PME") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

ggsave(paste(plot_filename, "_pme.pdf", sep=""), width=5.7,height=5)


#CSIM
csim_plot <- ggplot(data, aes(x=pose_name, y = G_CSIM_mean, fill=experiment_name)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    geom_errorbar(aes(ymin=G_CSIM_min, ymax=G_CSIM_max), stat="identity", position=position_dodge(.5) , width=0.2) +
    labs(x="pose_distribution", y="CSIM") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

ggsave(paste(plot_filename, "_csim.pdf", sep=""), width=5.7,height=5)


#SSIM
ssim_plot <- ggplot(data, aes(x=pose_name, y = G_SSIM_mean, fill=experiment_name)) +
    geom_bar(stat = "identity", position = "dodge", width=0.5) +
    geom_errorbar(aes(ymin=G_SSIM_min, ymax=G_SSIM_max), stat="identity", position=position_dodge(.5) , width=0.2) +
    labs(x="pose_distribution", y="SSIM") +
    theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="top",
            legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank())

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


