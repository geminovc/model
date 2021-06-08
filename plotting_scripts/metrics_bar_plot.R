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
data$experiment_name <- factor(data$experiment_name, 
                               levels=c("G_inf", "No_frozen"), 
                               labels=c("Inference\ntraining", "End-to-end\ntraining"))

psnr_plot <- ggplot(data, aes(x=experiment_name,y=G_PSNR_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 

        labs(x="Scheme", y="PSNR (dB)") +
        
        theme_minimal(base_size=15) +
        theme(axis.text.x=element_text(size=rel(1.0)), axis.text.y=element_text(size=rel(1.0))) +
        theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="none",
              legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank(),
              legend.margin=margin(c(0,0,0,0)))

ggsave(paste(plot_filename, "_psnr.pdf", sep=""), width=5.7,height=5)

lpips_plot <- ggplot(data, aes(x=experiment_name,y=G_LPIPS_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 

        labs(x="Scheme", y="LPIPS") +
        
        theme_minimal(base_size=15) +
        theme(axis.text.x=element_text(size=rel(1.0)), axis.text.y=element_text(size=rel(1.0))) +
        theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="none",
              legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank(),
              legend.margin=margin(c(0,0,0,0)))

ggsave(paste(plot_filename, "_lpips.pdf", sep=""), width=5.7,height=5)

pme_plot <- ggplot(data, aes(x=experiment_name,y=G_PME_mean,fill=experiment_name)) + 
        geom_bar(stat="identity", width=0.5) + 

        labs(x="Scheme", y="PME") +
        
        theme_minimal(base_size=15) +
        theme(axis.text.x=element_text(size=rel(1.0)), axis.text.y=element_text(size=rel(1.0))) +
        theme(legend.text=element_text(size=rel(0.9)), legend.key.size=unit(15,"points"), legend.position="none",
              legend.box.margin=margin(-10,-10,-10,-10), legend.title=element_blank(),
              legend.margin=margin(c(0,0,0,0)))

ggsave(paste(plot_filename, "_pme.pdf", sep=""), width=5.7,height=5)


legend <- get_legend(psnr_plot + theme(legend.position="top"))


prow <- plot_grid(psnr_plot + theme(legend.position="none"),
                  lpips_plot + theme(legend.position="none"),
                  pme_plot + theme(legend.position="none"),
                  ncol = 3, align = "v", axis = "l")

# this tells it what order to put it in
# so basically tells it put legend first then plots with th legend height 20% of the
# plot
p <- plot_grid(legend, prow, rel_heights=c(.2,1), ncol =1)
ggsave(paste(plot_filename,"_metrics.pdf", sep=""), width=12.2, height=3.5)
