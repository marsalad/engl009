import io, os
from google.cloud import vision

root = 'data'
verbose_frequency = 100
medias = ('drawings', 'engraving', 'iconography', 'painting', 'sculpture')

client = vision.ImageAnnotatorClient()

def get_labels(path):
	with io.open('%s/%s' % (root, path), 'rb') as f:
		image = vision.types.Image(content=f.read())
	labels = client.label_detection(image=image).label_annotations
	return [label.description for label in labels]

def create_label_file(source):
	i = 0
	with open('%s.txt' % source, 'w') as f:
		for media in medias:
			for file in os.listdir('%s/%s/%s' % (root, source, media)):
				path = '%s/%s/%s' % (source, media, file)
				f.write('%s\t%s\t%s\n' % (path, media, get_labels(path)))
				i += 1
				if i % verbose_frequency == 0:
					print('finished %d images' % i)

if __name__ == '__main__':
	create_label_file('musemart')
	create_label_file('dataset')