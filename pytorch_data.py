import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
'''
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(npimg)

'''
transform =transforms.Compose([
                               transforms.Scale(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])


def get_batch(batch_size):
	dataset_dir='/Users/kabir/Documents/SP17/Research/y/videogan_pytorch/videos2/'
	folder_index_file='data/beach_copy.txt'
	roots = list()
	'''
	with open(folder_index_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			tokens = line.split('/')
			root = '/'.join(tokens[:-1])
			roots.append(root)
	for root in roots:
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
				print images
				imshow(torchvision.utils.make_grid(images))
				video.append(images)
				cached_image = images
			except StopIteration:
				video.append(cached_image)
		batch.append(torch.stack(video))
        return torch.stack(batch)


if __name__ == '__main__':
	#print get_batch(1)
	#plt.show()
	dataset = dset.ImageFolder(root='/Users/kabir/Documents/SP17/Research/y/videogan_pytorch/videos2/',
		                           transform=transform)
	assert dataset
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
	                                         shuffle=True, num_workers=int(2))
	dataiter = iter(dataloader)
	images, labels = dataiter.next()
	imshow(torchvision.utils.make_grid(images))
	print images
#	plt.show()
