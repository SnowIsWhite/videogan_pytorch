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
	dataset_dir='/Users/kabir/Documents/SP17/Research/pytorch_videogan'
	folder_index_file='data/beach.txt'
	
	'''
	with open(folder_index_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			tokens = line.split('/')

	'''
	dataset = dset.ImageFolder(root=dataset_dir,
	                           transform=transform)
	assert dataset
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(2))

	frame_size = 32
	dataiter = iter(dataloader)
	batch = []
	for v in xrange(batch_size):
		video = []
		cached_image = None
		for x in xrange(frame_size):
			try:
				images, labels = dataiter.next()
				video.append(images)
				cached_image = images
			except StopIteration:
				video.append(cached_image)
		batch.append(torch.stack(video))

	return torch.stack(batch)
