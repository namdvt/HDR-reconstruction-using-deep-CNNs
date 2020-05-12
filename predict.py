import torch
import cv2
from model import Model
import matplotlib.pyplot as plt


def itm(model, input_image, tau):
    input_image = input_image / 255
    input_image = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0)
    max_c = input_image[0].max(dim=0).values - tau
    max_c[max_c < 0] = 0
    alpha = (max_c / (1 - tau)).float()

    with torch.no_grad():
        output_image = model(input_image)

    output_image = ((1 - alpha) * (input_image ** 2) + alpha * output_image).squeeze().permute(1, 2, 0).detach().numpy()
    return output_image


if __name__ == '__main__':
    # Load model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model(device).to(device)
    model.eval()
    model.load_state_dict(torch.load('output/separate_loss_false/weight.pth', map_location=device))

    # Load images
    ldr = cv2.imread('test/1.png')
    ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)

    hdr = cv2.imread('test/1.hdr', cv2.IMREAD_ANYDEPTH)
    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)

    hdr_reconstructed = itm(model, ldr, tau=0.95)

    # Save result
    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(ldr)
    plt.axis('off')
    plt.title('Input')

    fig.add_subplot(1, 3, 2)
    plt.imshow(hdr_reconstructed ** (1 / 2))
    plt.axis('off')
    plt.title('Reconstruction')

    fig.add_subplot(1, 3, 3)
    plt.imshow(hdr ** (1 / 2))
    plt.axis('off')
    plt.title('Ground truth')

    plt.savefig('test/results/1.png', dpi=300, bbox_inches='tight')
