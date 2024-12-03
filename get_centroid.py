import torch
import tqdm
import torch.nn.functional as F

def set_centroids(ssl_model, train_loader, num_classes):
    print('Computing Centroid via SSLCon model...')
    centroids = torch.zeros(num_classes, ssl_model.dim*ssl_model.n).cuda()
    counts = torch.zeros(num_classes).cuda()
    with torch.no_grad():
        progress_bar = tqdm(train_loader)
        for idx, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description(f"[{idx + 1}/{len(train_loader)}]")
            # image = images[2]
            image, labels = image.cuda(), labels.cuda()
            features = ssl_model.encoder_k(image)
            # features = features.view(features.size(0), -1)
            for i in range(num_classes):
                mask = labels == i
                centroids[i] += features[mask].sum(dim=0)
                counts[i] += mask.sum()
    centroids /= counts.unsqueeze(1)
    
    # centroids = rearrange(centroids, 'c (n d) -> n c d', n=ssl_model.n)
    centroids = F.normalize(centroids, dim=-1)
    
    return centroids, counts