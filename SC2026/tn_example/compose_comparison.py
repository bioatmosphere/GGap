#!/usr/bin/env python3
"""
Compose the paper-vs-ours side-by-side for site 978 (S. Appalachia).

Top row : the paper's published panel #7 (cropped from conus_10sites_with_bars.png).
Bottom  : our TN reproduction's panels (tn_site978_panels.png).
Columns line up: size distribution (left) | species composition over time (right).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from plot_tn_site978 import render_panels

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = "/home/cc/GGap/SC2026/figures/figs/conus_10sites_with_bars.png"
PAPER_CROP_BOX = (1620, 632, 2148, 850)  # site 7 (S. Appalachia): bar | comp
SITE_CSV = os.path.join(HERE, "results", "species_timeseries", "site_0978", "species_data.csv")
OURS_NOTITLE = os.path.join(HERE, "figures", "_ours_notitle")
OUT = os.path.join(HERE, "figures", "tn_site978_vs_paper")


def main():
    # Render OUR panels with NO internal title (the row label below identifies the source),
    # so the composite has a single, non-redundant set of labels.
    render_panels(SITE_CSV, 978, OURS_NOTITLE, suptitle=None, save_exts=("png",))

    paper = Image.open(PAPER).convert("RGB").crop(PAPER_CROP_BOX)
    ours = Image.open(f"{OURS_NOTITLE}.png").convert("RGB")

    fig = plt.figure(figsize=(9.5, 6.6))
    fig.suptitle("Reproduction check — site 978 / S. Appalachia (one of the paper's 10 representative sites, inside the TN box)",
                 fontsize=11, fontweight="bold")

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(paper)
    ax1.axis("off")
    ax1.set_title("Paper (full CONUS run) — panel #7:  size distribution @ yr 1000  |  species composition over time",
                  fontsize=9)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(ours)
    ax2.axis("off")
    ax2.set_title("This work (2-GPU TN reproduction) — site 978:  same two panels",
                  fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
        print(f"saved {OUT}.{ext}")
    if os.path.exists(f"{OURS_NOTITLE}.png"):
        os.remove(f"{OURS_NOTITLE}.png")


if __name__ == "__main__":
    main()
