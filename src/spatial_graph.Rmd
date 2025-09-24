```{r}
library(ggplot2)

colnames(metadata)[1] <- "ID"
colnames(result)[1] <- "ID"

merged_df <- merge(result, metadata, by = "ID")

colnames(merged_df)[2] <- "cluster"

# Plot
ggplot(merged_df, aes(x = CenterX_global_px, y = CenterY_global_px, color = as.factor(cluster))) +
geom_point(size = 0.1) +
#scale_color_manual(values = scales::hue_pal()(8)) +  # Up to 8 distinct colors
scale_color_manual(values = c("green", "red", "blue", "purple", "orange"))
theme_minimal() +
labs(color = "Cluster")
```