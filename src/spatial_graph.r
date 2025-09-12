```{r}
ggplot(metadata, aes(x = CenterX_global_px, y = CenterY_global_px)) +
  geom_point(color = "blue", size = 0.1) +
  labs(title = "Using Metadata",
       x = "X (pixels)", y = "Y (pixels)") +
  theme_minimal()
```

