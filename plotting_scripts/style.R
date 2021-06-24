library(ggplot2)
ggplot <- function(...) ggplot2::ggplot(...) + 
                        theme_minimal() +
                        theme(panel.border = element_blank(), 
                              #axis.line = element_line(colour = "black"), 
                              axis.line.x = element_line(), 
                              axis.line.y = element_line(),
			      axis.text = element_text(size = 14, color = "black"),
                              text = element_text(size = 14), 
                              axis.text.x = element_text(size = 10, color = "black"), 
                              axis.text.y = element_text(size = 10, color = "black"),
			      legend.key = element_blank(),
			      strip.background = element_blank())
