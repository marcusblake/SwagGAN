import torch
import torch.nn.functional as F
from typing import List
from PIL import Image
import seaborn as sns
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_cosine_similarity_heatmap(
        txt_embeds: torch.Tensor,
        txt_descriptions: List[str],
        img_embeds: torch.Tensor,
        images: List[Image.Image]) -> None:
    """
    Generates a heatmap plot to plot the cosine simlarities between the text and image embeddings.
    The index of the text_embeddings must correspond to the image of the text description and same
    with image embeddings. If this is not done, then your text/image pairs may not correspond to the
    embedding scores displayed in the heatmap. 
    """
    assert len(txt_descriptions) == len(images)
    assert txt_embeds.size() == img_embeds.size()
    if len(txt_descriptions) > 10:
        print('WARNING: Using too many text descriptions and images may yield undesirable plot results.')
    txt_embeds = F.normalize(txt_embeds, dim=1)
    img_embeds = F.normalize(img_embeds, dim=1)

    cos_sim = txt_embeds @ img_embeds.T
    print('Cosime similarity matrix:', cos_sim)
    ylabels = ['\n'.join(wrap(text, 90)) for text in txt_descriptions]

    _, ax = plt.subplots(figsize=(10,10)) 
    sns.heatmap(cos_sim.detach().numpy(), yticklabels=ylabels, ax=ax)
    ax.get_xaxis().set_ticklabels([])

    for i, image in enumerate(images):
        imagebox = OffsetImage(image, zoom=0.05)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (i,0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(-0.1, 1),
                            bboxprops={"edgecolor" : "none"})
        ax.add_artist(ab)

    plt.show()


