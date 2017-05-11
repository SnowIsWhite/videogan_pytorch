import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

transform = transforms.Compose([
	                            transforms.Scale(64),
	                            transforms.CenterCrop(64),
	                            transforms.ToTensor(),
	                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                           ])


def get_batch(batch_size):
	dataset = dset.ImageFolder(root='/Users/kabir/Documents/SP17/Research/pytorch_videogan',
	                           transform=transform)
	assert dataset
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(2))

	frame_size = 32
	dataiter = iter(dataloader)
	batch = []
	for v in xrange(batch_size):
		video = []
		for x in xrange(frame_size):
			images, labels = dataiter.next()
			video.append(images)
		batch.append(torch.stack(video))

	return torch.stack(batch)
