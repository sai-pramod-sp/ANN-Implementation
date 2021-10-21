import pandas as pd
import matplotlib.pyplot as plt
import os


def save_plot(df, plots_dir, plots_name):
    df.plot(figsize = (10,8))
    plt.grid(True)
    