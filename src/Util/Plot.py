from matplotlib import pyplot as plt


class PlotHelper:

    def __init__(self) -> None:
        pass

    def subplot(**images):
        n_images = len(images)
        plt.figure(figsize=(20, 8))
        for idx, (name, image) in enumerate(images.items()):
            plt.subplot(1, n_images, idx + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(name.replace('_', ' ').title())
            if (name == 'ground_truth_mask'):
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)

        plt.show()
