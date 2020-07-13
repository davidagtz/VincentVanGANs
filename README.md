# Vincent Van GANs

This project hopes to make realistic paintings that look as if painted by celebrated painter Van Gogh. There are multiple parts to this project.

-   `VanImages.py`
    -   This fetchs all of Van Goghs paintings as stated in the Wikipedia page.
    -   To accomplish this, `download_images(outdir)` scrapes the page using Beautiful Soup. It receives 150px wide previews. These are then downloaded to the `outdir` and resized as per the `resize_images(indir, shape)` function.
-   `vincentVanGANS.py`
    -   Trains the generator and discriminator models
-   `generator.py`
    -   The generator model
-   `discriminator.py`
    -   The discriminator model
